use crate::ir;

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use pyo3::exceptions;
use pyo3::types;

struct ConvertEnv<'py> {
    ast : Bound<'py, PyModule>,
    torch : Bound<'py, PyModule>,
    vars : HashMap<String, ir::Type>
}

fn runtime_error<T>(msg : String) -> PyResult<T> {
    Err(exceptions::PyRuntimeError::new_err(msg))
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
        let ty = match env.vars.get(&id) {
            Some(ty) => ty.clone(),
            None => ir::Type::Unknown
        };
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
            let ty = match env.vars.get(&id) {
                Some(ty) => Ok(ty.clone()),
                None => runtime_error(format!("Could not find type of variable {id}"))
            }?;
            Ok(ir::Expr::Var {id, ty})
        } else {
            runtime_error(format!("Subscript target must be a literal value"))
        }?;
        let idx = convert_expr(expr.getattr("slice")?, env)?;
        let ty = match target.get_type() {
            ir::Type::Array(ty) => *ty.clone(),
            ty => ir::Type::Unknown
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

        // Extract the lower and upper bound of the range. We currently do not support step sizes.
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

fn ir_elem_type<'py>(
    dtype : Bound<'py, PyAny>, id : &str, env : &'py ConvertEnv<'py>
) -> PyResult<ir::Type> {
    if dtype.eq(env.torch.getattr("int8")?)? {
        Ok(ir::Type::Int(ir::IntSize::I8))
    } else if dtype.eq(env.torch.getattr("int16")?)? {
        Ok(ir::Type::Int(ir::IntSize::I16))
    } else if dtype.eq(env.torch.getattr("int32")?)? {
        Ok(ir::Type::Int(ir::IntSize::I32))
    } else if dtype.eq(env.torch.getattr("int64")?)? {
        Ok(ir::Type::Int(ir::IntSize::I64))
    } else if dtype.eq(env.torch.getattr("float16")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F16))
    } else if dtype.eq(env.torch.getattr("float32")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F32))
    } else if dtype.eq(env.torch.getattr("float64")?)? {
        Ok(ir::Type::Float(ir::FloatSize::F64))
    } else {
        runtime_error(format!("Urgument {id} has unsupported tensor type containing {dtype:?}"))
    }
}

fn ir_type<'py>(
    e : &Bound<'py, PyAny>, id : &str, env : &'py ConvertEnv<'py>
) -> PyResult<ir::Type> {
    if e.is_instance(&types::PyInt::type_object(e.py()))? {
        Ok(ir::Type::Int(ir::IntSize::Any))
    } else if e.is_instance(&types::PyFloat::type_object(e.py()))? {
        Ok(ir::Type::Float(ir::FloatSize::Any))
    } else if e.is_instance(&env.torch.getattr("Tensor")?)? {
        let elem_ty = ir_elem_type(e.getattr("dtype")?, id, env)?;
        Ok(ir::Type::Array(Box::new(elem_ty)))
    } else {
        let ty = e.get_type();
        runtime_error(format!("Argument {id} has unsupported type {ty:?}"))
    }
}

pub fn to_ir<'py>(
    ast : Bound<'py, PyAny>,
    args : Vec<Bound<'py, PyAny>>,
    par : Vec<ir::ParSpec>,
    ast_module : Bound<'py, PyModule>
) -> PyResult<ir::Program> {
    let env = ConvertEnv {
        ast : ast_module,
        torch : ast.py().import("torch")?,
        vars : HashMap::new()
    };

    // Produce a list of names and types of the arguments passed to the function.
    let body = ast.getattr("body")?.get_item(0)?;
    let args = body.getattr("args")?.getattr("args")?.try_iter()?
        .zip(args.iter())
        .map(|(a1, a2)| {
            let id = a1?.getattr("arg")?.extract::<String>()?;
            let ty = ir_type(&a2, &id, &env)?;
            Ok(ir::TypedParam {id, ty})
        })
        .collect::<PyResult<Vec<ir::TypedParam>>>()?;

    let vars = args.clone()
        .into_iter()
        .map(|ir::TypedParam {id, ty}| (id, ty))
        .collect::<HashMap<String, ir::Type>>();
    let env = ConvertEnv {
        ast : env.ast,
        torch : env.torch,
        vars : vars
    };

    // Convert the statements of the function body to the IR AST.
    let id = body.getattr("name")?.extract::<String>()?;
    let body = convert_stmts(body.getattr("body")?, &env)?;

    // TODO: perform some kind of type unification/propagation, to ensure all left- and right-hand
    // sides are given a proper type (i.e., not Unknown).

    // Produce an IR consisting of the function and a function instantiation, describing how to
    // parallelize the function based on the provided argument.
    Ok(vec![
        ir::Def::ParFun {id : id.clone(), params : args.clone(), body},
        ir::Def::FunInst {id, par}
    ])
}
