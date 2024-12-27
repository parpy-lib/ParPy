
use crate::par;
use crate::err::*;
use crate::info::*;
use crate::py_ir::ast::*;

use std::collections::HashMap;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types;

macro_rules! runtime_error {
    ($i:tt, $($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::runtime_err($i.error_msg(format!($($t)*)))))
    }
}

macro_rules! type_error {
    ($i:tt, $($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::type_err($i.error_msg(format!($($t)*)))))
    }
}

//////////////////
// PYTHON TO IR //
//////////////////

struct ConvertEnv<'py, 'a> {
    ast : Bound<'py, PyModule>,
    filepath : &'a str,
    fst_line : usize
}

fn extract_node_info<'py>(
    node : &'py Bound<'py, PyAny>
) -> PyResult<Info> {
    let l1 = node.getattr("lineno")?.extract::<usize>()?;
    let c1 = node.getattr("col_offset")?.extract::<usize>()?;
    let start = FilePos::new(l1, c1);
    let l2 = if let Ok(line) = node.getattr("end_lineno") {
        line.extract::<usize>()?
    } else {
        l1
    };
    let c2 = if let Ok(col) = node.getattr("end_col_offset") {
        col.extract::<usize>()?
    } else {
        c1
    };
    let end = FilePos::new(l2, c2);
    Ok(Info::new("", start, end))
}

fn extract_info<'py, 'a>(
    node : &'py Bound<'py, PyAny>,
    env : &'py ConvertEnv<'py, 'a>
) -> PyResult<Info> {
    let i = extract_node_info(node)?;
    Ok(i.with_file(env.filepath)
        .with_line_offset(env.fst_line))
}

fn convert_bin_op<'py, 'a>(
    binop : Bound<'py, PyAny>,
    env : &'py ConvertEnv<'py, 'a>,
    i : &'a Info
) -> PyResult<BinOp> {
    if binop.is_instance(&env.ast.getattr("Add")?)? {
        Ok(BinOp::Add)
    } else if binop.is_instance(&env.ast.getattr("Sub")?)? {
        Ok(BinOp::Sub)
    } else if binop.is_instance(&env.ast.getattr("Mult")?)? {
        Ok(BinOp::Mul)
    } else {
        runtime_error!(i, "Unsupported binary operation: {binop:?}")
    }
}

fn convert_expr<'py, 'a>(
    expr : Bound<'py, PyAny>, env : &'py ConvertEnv<'py, 'a>
) -> PyResult<Expr> {
    let i = extract_info(&expr, env)?;
    if expr.is_instance(&env.ast.getattr("Name")?)? {
        let id = expr.getattr("id")?.extract::<String>()?;
        let ty = Type::Unknown;
        Ok(Expr::Var {id, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Constant")?)? {
        let val = expr.getattr("value")?;
        if val.is_instance(&types::PyInt::type_object(val.py()))? {
            let v = val.extract::<i64>()?;
            let ty = Type::Int(IntSize::Any);
            Ok(Expr::Int {v, ty, i})
        } else if val.is_instance(&types::PyFloat::type_object(val.py()))? {
            let v = val.extract::<f64>()?;
            let ty = Type::Float(FloatSize::Any);
            Ok(Expr::Float {v, ty, i})
        } else {
            let ty = expr.get_type();
            runtime_error!(i, "Unsupported constant {val:?} of type {ty:?}")
        }
    } else if expr.is_instance(&env.ast.getattr("BinOp")?)? {
        let lhs = convert_expr(expr.getattr("left")?, env)?;
        let op = convert_bin_op(expr.getattr("op")?, env, &i)?;
        let rhs = convert_expr(expr.getattr("right")?, env)?;
        let ty = Type::Unknown;
        Ok(Expr::BinOp {lhs : Box::new(lhs), op, rhs : Box::new(rhs), ty, i})
    } else if expr.is_instance(&env.ast.getattr("Subscript")?)? {
        let target = if expr.getattr("value")?.is_instance(&env.ast.getattr("Name")?)? {
            let id = expr.getattr("value")?.getattr("id")?.extract::<String>()?;
            let ty = Type::Unknown;
            Ok(Expr::Var {id, ty, i : i.clone()})
        } else {
            runtime_error!(i, "Subscript target must be a literal value")
        }?;
        let idx = convert_expr(expr.getattr("slice")?, env)?;
        let ty = match target.get_type() {
            Type::IntTensor(sz) => Type::Int(*sz),
            Type::FloatTensor(sz) => Type::Float(*sz),
            _ => Type::Unknown
        };
        Ok(Expr::Subscript {target : Box::new(target), idx : Box::new(idx), ty, i})
    } else {
        runtime_error!(i, "Unsupported expression: {expr}")
    }
}

fn convert_stmt<'py, 'a>(
    stmt : Bound<'py, PyAny>,
    env : &'py ConvertEnv<'py, 'a>
) -> PyResult<Stmt> {
    let i = extract_info(&stmt, env)?;
    if stmt.is_instance(&env.ast.getattr("For")?)? {
        // Ensure that the for-loop only assigns to a single variable
        let target = stmt.getattr("target")?;
        let var = if target.is_instance(&env.ast.getattr("Name")?)? {
            Ok(target.getattr("id")?.extract::<String>()?)
        } else {
            runtime_error!(i, "For-loops must assign to a single variable")
        }?;

        // Ensure the for-loop iterates over the range builtin
        let iter = stmt.getattr("iter")?;
        let range_fn = if iter.is_instance(&env.ast.getattr("Call")?)? {
            let func = iter.getattr("func")?;
            if func.is_instance(&env.ast.getattr("Name")?)? {
                if func.getattr("id")?.extract::<String>()? == "range" {
                    Ok(iter)
                } else {
                    runtime_error!(i, "For-loop must iterate using the range builtin")
                }
            } else {
                runtime_error!(i, "For-loop must iterate using the range builtin")
            }
        } else {
            runtime_error!(i, "For-loops must iterate using the range builtin")
        }?;

        // Extract the lower and upper bounds of the range. We currently do not support step sizes.
        let range_args = range_fn.getattr("args")?;
        let (lo, hi) = match range_args.len()? {
            1 => {
                let lo = Expr::Int {v : 0, ty : Type::Int(IntSize::I64), i : i.clone()};
                let hi = convert_expr(range_args.get_item(0)?, env)?;
                Ok((lo, hi))
            },
            2 => {
                let lo = convert_expr(range_args.get_item(0)?, env)?;
                let hi = convert_expr(range_args.get_item(1)?, env)?;
                Ok((lo, hi))
            },
            _ => runtime_error!(i, "For-loops with a step size are not supported")
        }?;

        let body = convert_stmts(stmt.getattr("body")?, env)?;

        if stmt.getattr("orelse")?.len()? == 0 {
            Ok(Stmt::For {
                var, lo, hi, body, i
            })
        } else {
            runtime_error!(i, "For-loops with an else-clause are not supported")
        }
    } else if stmt.is_instance(&env.ast.getattr("Assign")?)? {
        let targets = stmt.getattr("targets")?;
        if targets.len()? > 1 {
            runtime_error!(i, "Cannot have more than one target of assignment")
        } else {
            let dst = convert_expr(targets.get_item(0)?, env)?;
            let e = convert_expr(stmt.getattr("value")?, env)?;
            Ok(Stmt::Assign { dst, e, i })
        }
    } else {
        runtime_error!(i, "Unsupported statement: {stmt}")
    }
}

fn merge_body_infos(
    body : &Vec<Stmt>
) -> Info {
    body.iter().fold(Info::default(), |acc, stmt| {
        Info::merge(acc, stmt.get_info())
    })
}

fn convert_stmts<'py, 'a>(
    body : Bound<'py, PyAny>,
    env : &'py ConvertEnv<'py, 'a>
) -> PyResult<Vec<Stmt>> {
    body.try_iter()?
        .map(|stmt| stmt.and_then(|s| convert_stmt(s, &env)))
        .collect::<PyResult<Vec<Stmt>>>()
}

pub fn to_untyped_ir<'py>(
    ast : Bound<'py, PyAny>,
    filepath : String,
    fst_line : usize
) -> PyResult<Ast> {
    let env = ConvertEnv {
        ast : ast.py().import("ast")?,
        filepath : &filepath,
        fst_line
    };

    let body = ast.getattr("body")?.get_item(0)?;
    let untyped_args = body.getattr("args")?.getattr("args")?.try_iter()?
        .map(|arg| {
            let id = arg?.getattr("arg")?.extract::<String>()?;
            let ty = Type::Unknown;
            let i = Info::default();
            Ok(TypedParam {id, ty, i})
        })
        .collect::<PyResult<Vec<TypedParam>>>()?;
    let id = body.getattr("name")?.extract::<String>()?;
    let ir_body = convert_stmts(body.getattr("body")?, &env)?;
    let i = merge_body_infos(&ir_body);
    Ok(vec![Def::FunDef {id, params: untyped_args, body: ir_body, i}])
}

//////////////////////
// TYPE PROPAGATION //
//////////////////////

#[derive(Debug)]
struct TypeEnv {
    vars : HashMap<String, Type>
}

impl Default for TypeEnv {
    fn default() -> TypeEnv {
        TypeEnv { vars : HashMap::new() }
    }
}

fn ir_elem_type<'py>(
    dtype : Bound<'py, PyAny>,
    id : &str
) -> PyResult<Type> {
    let torch = dtype.py().import("torch")?;
    if dtype.eq(torch.getattr("int8")?)? {
        Ok(Type::Int(IntSize::I8))
    } else if dtype.eq(torch.getattr("int16")?)? {
        Ok(Type::Int(IntSize::I16))
    } else if dtype.eq(torch.getattr("int32")?)? {
        Ok(Type::Int(IntSize::I32))
    } else if dtype.eq(torch.getattr("int64")?)? {
        Ok(Type::Int(IntSize::I64))
    } else if dtype.eq(torch.getattr("float16")?)? {
        Ok(Type::Float(FloatSize::F16))
    } else if dtype.eq(torch.getattr("float32")?)? {
        Ok(Type::Float(FloatSize::F32))
    } else if dtype.eq(torch.getattr("float64")?)? {
        Ok(Type::Float(FloatSize::F64))
    } else {
        let i = Info::default();
        runtime_error!(i, "Argument {id} has unsupported tensor type containing {dtype:?}")
    }
}

fn ir_type<'py>(
    e : &Bound<'py, PyAny>,
    id : &str
) -> PyResult<Type> {
    let torch = e.py().import("torch")?;
    if e.is_instance(&types::PyInt::type_object(e.py()))? {
        Ok(Type::Int(IntSize::Any))
    } else if e.is_instance(&types::PyFloat::type_object(e.py()))? {
        Ok(Type::Float(FloatSize::Any))
    } else if e.is_instance(&torch.getattr("Tensor")?)? {
        let elem_ty = ir_elem_type(e.getattr("dtype")?, id)?;
        Ok(match elem_ty {
            Type::Int(sz) => Type::IntTensor(sz),
            Type::Float(sz) => Type::FloatTensor(sz),
            _ => panic!("Reached impossible case in 'ir_type'")
        })
    } else {
        let i = Info::default();
        let ty = e.get_type();
        runtime_error!(i, "Argument {id} has unsupported type {ty:?}")
    }
}

fn lookup_type(
    env : &TypeEnv,
    id : &str,
    i : &Info
) -> PyResult<Type> {
    match env.vars.get(id) {
        Some(ty) => Ok(ty.clone()),
        None => runtime_error!(i ,"Unknown type of variable {id} (env={env:?})")
    }
}

fn unify_int_sizes(
    lsz : &IntSize,
    rsz : &IntSize,
    i : &Info
) -> PyResult<IntSize> {
    match (lsz, rsz) {
        (IntSize::Any, _) => Ok(rsz.clone()),
        (_, IntSize::Any) => Ok(lsz.clone()),
        (IntSize::I8, IntSize::I8) => Ok(IntSize::I8),
        (IntSize::I16, IntSize::I16) => Ok(IntSize::I16),
        (IntSize::I32, IntSize::I32) => Ok(IntSize::I32),
        (IntSize::I64, IntSize::I64) => Ok(IntSize::I64),
        _ => type_error!(i, "Integer type size mismatch: {lsz:?} != {rsz:?}")
    }
}

fn unify_float_sizes(
    lsz : &FloatSize,
    rsz : &FloatSize,
    i : &Info
) -> PyResult<FloatSize> {
    match (lsz, rsz) {
        (FloatSize::Any, _) => Ok(rsz.clone()),
        (_, FloatSize::Any) => Ok(lsz.clone()),
        (FloatSize::F16, FloatSize::F16) => Ok(FloatSize::F16),
        (FloatSize::F32, FloatSize::F32) => Ok(FloatSize::F32),
        (FloatSize::F64, FloatSize::F64) => Ok(FloatSize::F64),
        _ => type_error!(i, "Float type size mismatch: {lsz:?} != {rsz:?}")
    }
}

fn typed_binop(
    op : &BinOp,
    lhs : &Expr,
    rhs : &Expr,
    i : &Info
) -> PyResult<Type> {
    let lty = lhs.get_type();
    let rty = rhs.get_type();
    // Assumes the operator is an arithmetic operator on either integers or floats.
    match (lty, rty) {
        (Type::Int(lsz), Type::Int(rsz)) => {
            let sz = unify_int_sizes(lsz, rsz, i)?;
            Ok(Type::Int(sz))
        },
        (Type::Float(lsz), Type::Float(rsz)) => {
            let sz = unify_float_sizes(lsz, rsz, i)?;
            Ok(Type::Float(sz))
        },
        _ => type_error!(i, "Invalid types of arguments used with binary operator {op:?}: {lhs:?} and {rhs:?}")
    }
}

fn typed_expr(
    env : &TypeEnv,
    e : Expr
) -> PyResult<Expr> {
    match e {
        Expr::Var {id, i, ..} => {
            let ty = lookup_type(env, &id, &i)?;
            Ok(Expr::Var {id, ty, i})
        },
        Expr::Int {v, ty, i} => {
            let sz = if let Type::Int(sz) = ty { sz } else { IntSize::Any };
            let ty = Type::Int(sz);
            Ok(Expr::Int {v, ty, i})
        },
        Expr::Float {v, ty, i} => {
            let sz = if let Type::Float(sz) = ty { sz } else { FloatSize::Any };
            let ty = Type::Float(sz);
            Ok(Expr::Float {v, ty, i})
        },
        Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = typed_expr(&env, *lhs)?;
            let rhs = typed_expr(&env, *rhs)?;
            let ty = typed_binop(&op, &lhs, &rhs, &i)?;
            let lhs = Box::new(lhs.with_type(ty.clone()));
            let rhs = Box::new(rhs.with_type(ty.clone()));
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        Expr::Subscript {target, idx, i, ..} => {
            let target = typed_expr(&env, *target)?;
            let idx = typed_expr(&env, *idx)?;
            let ty = match target.get_type() {
                Type::IntTensor(sz) => Ok(Type::Int(*sz)),
                Type::FloatTensor(sz) => Ok(Type::Float(*sz)),
                _ => type_error!(i, "Invalid type of subscript operation")
            }?;
            Ok(Expr::Subscript {
                target : Box::new(target),
                idx : Box::new(idx),
                ty, i
            })
        }
    }
}

fn typed_stmt(
    mut env : TypeEnv,
    stmt : Stmt
) -> PyResult<(TypeEnv, Stmt)> {
    match stmt {
        Stmt::Assign {dst, e, i} => {
            let e = typed_expr(&env, e)?;
            if let Expr::Var {ref id, ..} = dst {
                env.vars.insert(id.clone(), e.get_type().clone())
            } else {
                None
            };
            let dst = typed_expr(&env, dst)?;
            Ok((env, Stmt::Assign {dst, e, i}))
        },
        Stmt::For {var, lo, hi, body, i} => {
            let lo = typed_expr(&env, lo)?;
            let hi = typed_expr(&env, hi)?;
            env.vars.insert(var.clone(), Type::Int(IntSize::I64));
            let (env, body) = typed_stmts(env, body)?;
            Ok((env, Stmt::For {var, lo, hi, body, i}))
        },
    }
}

fn typed_stmts(
    env : TypeEnv,
    body : Vec<Stmt>
) -> PyResult<(TypeEnv, Vec<Stmt>)> {
    body.into_iter()
        .fold(Ok((env, vec![])), |acc, s| {
            let (env, mut stmts) = acc?;
            let (env, s) = typed_stmt(env, s)?;
            stmts.push(s);
            Ok((env, stmts))
        })
}

pub fn to_typed_ir<'py>(
    ast : &Ast,
    args : Vec<Bound<'py, PyAny>>,
    par : HashMap<String, par::ParSpec>
) -> PyResult<Ast> {
    let (id, params, body, i) = match &ast[..] {
        [Def::FunDef {id, params, body, i}] => {
            Ok((id, params, body, i))
        },
        _ => {
            let i = Info::default();
            runtime_error!(i, "Compiler expected Python IR AST to consist of one function definition")
        }
    }?;
    let fst_line = i.get_start().line;
    let typed_params = if args.len() == params.len() {
        args.into_iter()
            .zip(params.into_iter())
            .map(|(a, TypedParam {id, ..})| {
                let i = i.clone().with_line_offset(fst_line);
                let ty = ir_type(&a, &id)?;
                Ok(TypedParam {id : id.clone(), ty, i : i.clone()})
            })
            .collect::<PyResult<Vec<TypedParam>>>()
    } else {
        let i = Info::default();
        let l1 = params.len();
        let l2 = args.len();
        runtime_error!(i, "Function {id} expected {l1} arguments but received {l2}")
    }?;
    let tyvars = typed_params.iter()
        .map(|TypedParam {id, ty, ..}| (id.clone(), ty.clone()))
        .collect::<HashMap<String, Type>>();
    let env = TypeEnv { vars : tyvars };
    let (_, body) = typed_stmts(env, body.clone())?;

    Ok(vec![
        Def::FunDef {id : id.clone(), params : typed_params, body, i : i.clone()},
        Def::ParFunInst {id : id.clone(), par, i : i.clone()}
    ])
}

#[cfg(test)]
mod test {
    use super::*;

    fn int_ty(sz : IntSize) -> Type {
        Type::Int(sz)
    }

    fn int(v : i64, ty : Type) -> Expr {
        Expr::Int {v, ty, i : Info::default()}
    }

    fn add(lhs : Expr, rhs : Expr) -> Expr {
        Expr::BinOp {
            lhs : Box::new(lhs),
            op : BinOp::Add,
            rhs : Box::new(rhs),
            ty : Type::Unknown,
            i : Info::default()
        }
    }

    #[test]
    fn type_propagate_binop_type() {
        let e = add(int(1, int_ty(IntSize::Any)), int(2, int_ty(IntSize::I32)));
        let env = TypeEnv::default();
        let expected_ty = Type::Int(IntSize::I32);
        match typed_expr(&env, e) {
            Ok(Expr::BinOp {lhs, rhs, ty, ..}) => {
                assert_eq!(lhs.get_type(), &expected_ty);
                assert_eq!(rhs.get_type(), &expected_ty);
                assert_eq!(ty, expected_ty);
            },
            _ => assert!(false)
        }
    }

    #[test]
    fn var_type_lookup() {
        let mut env = TypeEnv::default();
        let id = "x".to_string();
        let ty = Type::Float(FloatSize::F32);
        env.vars.insert(id.clone(), ty.clone());
        assert_eq!(lookup_type(&env, &id, &Info::default()).unwrap(), ty);
    }

    #[test]
    fn type_propagate_assign() {
        let expected_ty = Type::Float(FloatSize::F32);
        let s = Stmt::Assign {
            dst : Expr::Var {id : "x".to_string(), ty : Type::Unknown, i : Info::default()},
            e : Expr::Float {v : 3.14, ty : expected_ty.clone(), i : Info::default()},
            i : Info::default()
        };
        let env = TypeEnv::default();
        match typed_stmt(env, s) {
            Ok((_, Stmt::Assign {dst, e, ..})) => {
                assert_eq!(dst.get_type(), &expected_ty);
                assert_eq!(e.get_type(), &expected_ty);
            },
            _ => assert!(false)
        }
    }
}
