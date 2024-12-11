use crate::ir;

use std::collections::HashMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use pyo3::exceptions;
use pyo3::types;
use pyo3::types::PyCapsule;

struct ConvertEnv<'py> {
    ast : Bound<'py, PyModule>,
}

fn runtime_error<T>(msg : String) -> PyResult<T> {
    Err(exceptions::PyRuntimeError::new_err(msg))
}

fn type_error<T>(msg : String) -> PyResult<T> {
    Err(exceptions::PyTypeError::new_err(msg))
}

fn convert_bin_op<'py>(binop : Bound<'py, PyAny>, env : &'py ConvertEnv<'py>) -> PyResult<ir::BinOp> {
    if binop.is_instance(&env.ast.getattr("Add")?)? {
        Ok(ir::BinOp::Add)
    } else if binop.is_instance(&env.ast.getattr("Mult")?)? {
        Ok(ir::BinOp::Mul)
    } else {
        runtime_error(format!("Unsupported binary operation: {binop:?}"))
    }
}

fn convert_expr<'py>(
    expr : Bound<'py, PyAny>, env : &'py ConvertEnv<'py>
) -> PyResult<ir::Expr> {
    if expr.is_instance(&env.ast.getattr("Name")?)? {
        let id = expr.getattr("id")?.extract::<String>()?;
        let ty = ir::Type::Unknown;
        Ok(ir::Expr::Var {id, ty})
    } else if expr.is_instance(&env.ast.getattr("Constant")?)? {
        let val = expr.getattr("value")?;
        if val.is_instance(&types::PyInt::type_object(val.py()))? {
            let value = val.extract::<i64>()?;
            let ty = ir::Type::Int(ir::IntSize::Any);
            Ok(ir::Expr::LiteralInt {value, ty})
        } else if val.is_instance(&types::PyFloat::type_object(val.py()))? {
            let value = val.extract::<f64>()?;
            let ty = ir::Type::Float(ir::FloatSize::Any);
            Ok(ir::Expr::LiteralFloat {value, ty})
        } else {
            let ty = expr.get_type();
            runtime_error(format!("Unsupported constant {val:?} of type {ty:?}"))
        }
    } else if expr.is_instance(&env.ast.getattr("BinOp")?)? {
        let lhs = convert_expr(expr.getattr("left")?, env)?;
        let op = convert_bin_op(expr.getattr("op")?, env)?;
        let rhs = convert_expr(expr.getattr("right")?, env)?;
        let ty = ir::Type::Unknown;
        Ok(ir::Expr::BinOp {lhs : Box::new(lhs), op, rhs : Box::new(rhs), ty})
    } else if expr.is_instance(&env.ast.getattr("Subscript")?)? {
        let target = if expr.getattr("value")?.is_instance(&env.ast.getattr("Name")?)? {
            let id = expr.getattr("value")?.getattr("id")?.extract::<String>()?;
            let ty = ir::Type::Unknown;
            Ok(ir::Expr::Var {id, ty})
        } else {
            runtime_error(format!("Subscript target must be a literal value"))
        }?;
        let idx = convert_expr(expr.getattr("slice")?, env)?;
        let ty = match target.get_type() {
            ir::Type::Array(ty) => *ty.clone(),
            _ => ir::Type::Unknown
        };
        Ok(ir::Expr::ArrayAccess {target : Box::new(target), idx : Box::new(idx), ty})
    } else {
        runtime_error(format!("not implemented yet"))
    }
}

fn convert_stmt<'py>(stmt : Bound<'py, PyAny>, env : &'py ConvertEnv<'py>) -> PyResult<ir::Stmt> {
    if stmt.is_instance(&env.ast.getattr("For")?)? {
        // Ensure that the for-loop only assigns to a single variable
        let target = stmt.getattr("target")?;
        let var = if target.is_instance(&env.ast.getattr("Name")?)? {
            Ok(target.getattr("id")?.extract()?)
        } else {
            runtime_error(format!("For-loops must assign to a single variable"))
        }?;

        // Ensure the for-loop iterates over the range builtin
        let iter = stmt.getattr("iter")?;
        let range_fn = if iter.is_instance(&env.ast.getattr("Call")?)? {
            let func = iter.getattr("func")?;
            if func.is_instance(&env.ast.getattr("Name")?)? {
                if func.getattr("id")?.extract::<String>()? == "range" {
                    Ok(iter)
                } else {
                    runtime_error(format!("For-loop must iterate using the range builtin"))
                }
            } else {
                runtime_error(format!("For-loop must iterate using the range builtin"))
            }
        } else {
            runtime_error(format!("For-loops must iterate using the range builtin"))
        }?;

        // Extract the lower and upper bounds of the range. We currently do not support step sizes.
        let range_args = range_fn.getattr("args")?;
        let (lo, hi) = match range_args.len()? {
            1 => {
                let lo = ir::Expr::LiteralInt {value : 0, ty : ir::Type::Int(ir::IntSize::Any)};
                let hi = convert_expr(range_args.get_item(0)?, env)?;
                Ok((lo, hi))
            },
            2 => {
                let lo = convert_expr(range_args.get_item(0)?, env)?;
                let hi = convert_expr(range_args.get_item(1)?, env)?;
                Ok((lo, hi))
            },
            _ => runtime_error(format!("For-loop range cannot specify a step size"))
        }?;

        let body = convert_stmts(stmt.getattr("body")?, env)?;

        if stmt.getattr("orelse")?.len()? == 0 {
            Ok(ir::Stmt::For {var, lo, hi, body})
        } else {
            runtime_error(format!("For-loop with an else-clause are not supported"))
        }
    } else if stmt.is_instance(&env.ast.getattr("Assign")?)? {
        let targets = stmt.getattr("targets")?;
        if targets.len()? > 1 {
            runtime_error(format!("Cannot have more than one target of assignment"))
        } else {
            let dst = convert_expr(targets.get_item(0)?, env)?;
            let e = convert_expr(stmt.getattr("value")?, env)?;
            Ok(ir::Stmt::Assign { dst, e })
        }
    } else {
        runtime_error(format!("Unsupported statement: {stmt}"))
    }
}

fn convert_stmts<'py>(
    body : Bound<'py, PyAny>, env : &'py ConvertEnv<'py>
) -> PyResult<Vec<ir::Stmt>> {
    body.try_iter()?
        .map(|stmt| stmt.and_then(|s| convert_stmt(s, &env)))
        .collect::<PyResult<Vec<ir::Stmt>>>()
}

pub fn to_untyped_ir<'py>(
    ast : Bound<'py, PyAny>
) -> PyResult<Bound<'py, types::PyCapsule>> {
    let env = ConvertEnv {
        ast : ast.py().import("ast")?
    };

    let body = ast.getattr("body")?.get_item(0)?;
    let untyped_args = body.getattr("args")?.getattr("args")?.try_iter()?
        .map(|arg| {
            let id = arg?.getattr("arg")?.extract::<String>()?;
            let ty = ir::Type::Unknown;
            Ok(ir::TypedParam {id, ty})
        })
        .collect::<PyResult<Vec<ir::TypedParam>>>()?;
    let id = body.getattr("name")?.extract::<String>()?;
    let ir_body = convert_stmts(body.getattr("body")?, &env)?;
    let par_fun = ir::Def::ParFun {id, params: untyped_args, body: ir_body};

    // Build a capsule object using which we can pass along the IR AST to Python without having to
    // convert it first.
    let name = CString::new("IR AST")?;
    Ok(PyCapsule::new::<ir::Program>(ast.py(), vec![par_fun], Some(name))?)
}

//////////////////////
// TYPE PROPAGATION //
//////////////////////

fn ir_elem_type<'py>(
    dtype : Bound<'py, PyAny>, id : &str
) -> PyResult<ir::Type> {
    let torch = dtype.py().import("torch")?;
    if dtype.eq(torch.getattr("int8")?)? {
        Ok(ir::Type::Int(ir::IntSize::I8))
    } else if dtype.eq(torch.getattr("int16")?)? {
        Ok(ir::Type::Int(ir::IntSize::I16))
    } else if dtype.eq(torch.getattr("int32")?)? {
        Ok(ir::Type::Int(ir::IntSize::I32))
    } else if dtype.eq(torch.getattr("int64")?)? {
        Ok(ir::Type::Int(ir::IntSize::I64))
    } else if dtype.eq(torch.getattr("float16")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F16))
    } else if dtype.eq(torch.getattr("float32")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F32))
    } else if dtype.eq(torch.getattr("float64")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F64))
    } else {
        runtime_error(format!("Argument {id} has unsupported tensor type containing {dtype:?}"))
    }
}

fn ir_type<'py>(
    e : &Bound<'py, PyAny>, id : &str
) -> PyResult<ir::Type> {
    let torch = e.py().import("torch")?;
    if e.is_instance(&types::PyInt::type_object(e.py()))? {
        Ok(ir::Type::Int(ir::IntSize::Any))
    } else if e.is_instance(&types::PyFloat::type_object(e.py()))? {
        Ok(ir::Type::Float(ir::FloatSize::Any))
    } else if e.is_instance(&torch.getattr("Tensor")?)? {
        let elem_ty = ir_elem_type(e.getattr("dtype")?, id)?;
        Ok(ir::Type::Array(Box::new(elem_ty)))
    } else {
        let ty = e.get_type();
        runtime_error(format!("Argument {id} has unsupported type {ty:?}"))
    }
}

#[derive(Debug)]
struct TypeEnv {
    vars : HashMap<String, ir::Type>
}

fn lookup_type(
    env : &TypeEnv,
    id : &str
) -> PyResult<ir::Type> {
    match env.vars.get(id) {
        Some(ty) => Ok(ty.clone()),
        None => runtime_error(format!("Unknown type of variable {id} (env={env:?})"))
    }
}

fn unify_int_sizes(
    lsz : &ir::IntSize,
    rsz : &ir::IntSize
) -> PyResult<ir::IntSize> {
    match (lsz, rsz) {
        (ir::IntSize::Any, _) => Ok(rsz.clone()),
        (_, ir::IntSize::Any) => Ok(lsz.clone()),
        (ir::IntSize::I8, ir::IntSize::I8) => Ok(ir::IntSize::I8),
        (ir::IntSize::I16, ir::IntSize::I16) => Ok(ir::IntSize::I16),
        (ir::IntSize::I32, ir::IntSize::I32) => Ok(ir::IntSize::I32),
        (ir::IntSize::I64, ir::IntSize::I64) => Ok(ir::IntSize::I64),
        _ => type_error(format!("Integer type size mismatch: {lsz:?} != {rsz:?}"))
    }
}

fn unify_float_sizes(
    lsz : &ir::FloatSize,
    rsz : &ir::FloatSize
) -> PyResult<ir::FloatSize> {
    match (lsz, rsz) {
        (ir::FloatSize::Any, _) => Ok(rsz.clone()),
        (_, ir::FloatSize::Any) => Ok(lsz.clone()),
        (ir::FloatSize::F16, ir::FloatSize::F16) => Ok(ir::FloatSize::F16),
        (ir::FloatSize::F32, ir::FloatSize::F32) => Ok(ir::FloatSize::F32),
        (ir::FloatSize::F64, ir::FloatSize::F64) => Ok(ir::FloatSize::F64),
        _ => type_error(format!("Float type size mismatch: {lsz:?} != {rsz:?}"))
    }
}

fn typed_binop(
    op : &ir::BinOp,
    lhs : &ir::Expr,
    rhs : &ir::Expr
) -> PyResult<ir::Type> {
    let lty = lhs.get_type();
    let rty = rhs.get_type();
    // Assumes the operator is an arithmetic operator on either integers or floats.
    match (lty, rty) {
        (ir::Type::Int(lsz), ir::Type::Int(rsz)) => {
            let sz = unify_int_sizes(lsz, rsz)?;
            Ok(ir::Type::Int(sz))
        },
        (ir::Type::Float(lsz), ir::Type::Float(rsz)) => {
            let sz = unify_float_sizes(lsz, rsz)?;
            Ok(ir::Type::Float(sz))
        },
        _ => type_error(format!("Invalid types of arguments used with binary operator {op:?}: {lhs:?} and {rhs:?}"))
    }
}

fn typed_expr(
    env : &TypeEnv,
    e : ir::Expr
) -> PyResult<ir::Expr> {
    match e {
        ir::Expr::Var {id, ..} => {
            let ty = lookup_type(env, &id)?;
            Ok(ir::Expr::Var {id, ty})
        },
        ir::Expr::LiteralInt {..} => Ok(e),
        ir::Expr::LiteralFloat {value, ..} => {
            let ty = ir::Type::Float(ir::FloatSize::Any);
            Ok(ir::Expr::LiteralFloat {value, ty})
        },
        ir::Expr::BinOp {lhs, op, rhs, ..} => {
            let lhs = typed_expr(&env, *lhs)?;
            let rhs = typed_expr(&env, *rhs)?;
            let ty = typed_binop(&op, &lhs, &rhs)?;
            let lhs = Box::new(lhs.with_type(ty.clone()));
            let rhs = Box::new(rhs.with_type(ty.clone()));
            Ok(ir::Expr::BinOp {lhs, op, rhs, ty})
        },
        ir::Expr::ArrayAccess {target, idx, ..} => {
            let target = typed_expr(&env, *target)?;
            let idx = typed_expr(&env, *idx)?;
            let ty = match target.get_type() {
                ir::Type::Array(ty) => Ok(ty),
                _ => type_error(format!("Invalid type of subscript operation"))
            }?.clone();
            Ok(ir::Expr::ArrayAccess {
                target : Box::new(target),
                idx : Box::new(idx),
                ty : *ty
            })
        }
    }
}

fn typed_stmt(
    mut env : TypeEnv,
    stmt : ir::Stmt
) -> PyResult<(TypeEnv, ir::Stmt)> {
    match stmt {
        ir::Stmt::Assign {dst, e} => {
            let e = typed_expr(&env, e)?;
            if let ir::Expr::Var {ref id, ..} = dst {
                env.vars.insert(id.clone(), e.get_type().clone())
            } else {
                None
            };
            let dst = typed_expr(&env, dst)?;
            Ok((env, ir::Stmt::Assign {dst, e}))
        },
        ir::Stmt::For {var, lo, hi, body} => {
            let lo = typed_expr(&env, lo)?;
            let hi = typed_expr(&env, hi)?;
            env.vars.insert(var.clone(), lo.get_type().clone());
            let (env, body) = typed_stmts(env, body)?;
            Ok((env, ir::Stmt::For {var, lo, hi, body}))
        }
    }
}

fn typed_stmts(
    env : TypeEnv,
    body : Vec<ir::Stmt>
) -> PyResult<(TypeEnv, Vec<ir::Stmt>)> {
    body.into_iter()
        .fold(Ok((env, vec![])), |acc, s| {
            let (env, mut stmts) = acc?;
            let (env, s) = typed_stmt(env, s)?;
            stmts.push(s);
            Ok((env, stmts))
        })
}

pub fn to_typed_ir<'py>(
    ast : ir::Program,
    args : Vec<Bound<'py, PyAny>>,
    par : Vec<ir::ParSpec>
) -> PyResult<ir::Program> {
    let (id, params, body) = if ast.len() == 1 {
        match ast.into_iter().nth(0) {
            Some(ir::Def::ParFun {id, params, body}) => Ok((id, params, body)),
            _ => runtime_error(format!("Compiler expected IR AST to consist of one function definition"))
        }
    } else {
        runtime_error(format!("Compiler expected IR AST to consist of one function definition"))
    }?;
    let typed_params = if args.len() == params.len() {
        args.into_iter()
            .zip(params.into_iter())
            .map(|(a, ir::TypedParam {id, ..})| {
                let ty = ir_type(&a, &id)?;
                Ok(ir::TypedParam {id: id.clone(), ty})
            })
            .collect::<PyResult<Vec<ir::TypedParam>>>()
    } else {
        let l1 = params.len();
        let l2 = args.len();
        runtime_error(format!("Function {id} expects {l1} arguments, but received {l2} arguments"))
    }?;

    let tyenv = TypeEnv {
        vars : typed_params.clone()
            .into_iter()
            .map(|ir::TypedParam {id, ty}| (id, ty))
            .collect::<HashMap<String, ir::Type>>()
    };
    let (_, body) = typed_stmts(tyenv, body)?;

    Ok(vec![
        ir::Def::ParFun {id: id.clone(), params: typed_params, body},
        ir::Def::FunInst {id, par}
    ])
}
