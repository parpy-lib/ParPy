use crate::py_runtime_error;
use crate::err::*;
use crate::info::*;
use super::ast::*;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types;

use std::ffi::CString;

struct ConvertEnv<'py, 'a> {
    ast: Bound<'py, PyModule>,
    filepath: &'a str,
    fst_line: usize
}

fn extract_node_info<'py>(
    node: &'py Bound<'py, PyAny>
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
    node: &'py Bound<'py, PyAny>,
    env: &'py ConvertEnv<'py, 'a>
) -> PyResult<Info> {
    let i = extract_node_info(node)?;
    Ok(i.with_file(env.filepath)
        .with_line_offset(env.fst_line))
}

fn convert_unary_op<'py, 'a>(
    unop: Bound<'py, PyAny>,
    env: &'py ConvertEnv<'py, 'a>,
    i: &'a Info
) -> PyResult<UnOp> {
    if unop.is_instance(&env.ast.getattr("USub")?)? {
        Ok(UnOp::Sub)
    } else {
        py_runtime_error!(i, "Unsupported unary expression {unop:?}")
    }
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
    } else if binop.is_instance(&env.ast.getattr("Div")?)? {
        Ok(BinOp::Div)
    } else if binop.is_instance(&env.ast.getattr("FloorDiv")?)? {
        Ok(BinOp::FloorDiv)
    } else if binop.is_instance(&env.ast.getattr("Mod")?)? {
        Ok(BinOp::Mod)
    } else if binop.is_instance(&env.ast.getattr("BitAnd")?)? {
        Ok(BinOp::BitAnd)
    } else if binop.is_instance(&env.ast.getattr("Eq")?)? {
        Ok(BinOp::Eq)
    } else if binop.is_instance(&env.ast.getattr("Lt")?)? {
        Ok(BinOp::Lt)
    } else if binop.is_instance(&env.ast.getattr("Gt")?)? {
        Ok(BinOp::Gt)
    } else {
        py_runtime_error!(i, "Unsupported binary operation: {binop:?}")
    }
}

fn lookup_builtin<'py>(expr: &Bound<'py, PyAny>, i: &Info) -> PyResult<Expr> {
    let py = expr.py();
    let ast = py.import("ast")?;
    let parir = py.import("parir")?;
    let s = ast.call_method1("unparse", types::PyTuple::new(py, vec![expr])?)?
        .extract::<String>()?;
    let globals = types::PyDict::new(py);
    globals.set_item("parir", parir.clone())?;
    match py.eval(&CString::new(s)?, Some(&globals), None) {
        Ok(e) => {
            let func = if e.eq(parir.getattr("exp")?)? {
                Ok(Builtin::Exp)
            } else if e.eq(parir.getattr("inf")?)? {
                Ok(Builtin::Inf)
            } else if e.eq(parir.getattr("log")?)? {
                Ok(Builtin::Log)
            } else if e.eq(parir.getattr("min")?)? {
                Ok(Builtin::Min)
            } else if e.eq(parir.getattr("max")?)? {
                Ok(Builtin::Max)
            } else if e.eq(parir.getattr("sum")?)? {
                Ok(Builtin::Sum)
            } else {
                py_runtime_error!(i, "Unknown built-in operator {expr}")
            }?;
            Ok(Expr::Builtin {func, args: vec![], ty: Type::Unknown, i: i.clone()})
        },
        Err(py_err) => {
            py_runtime_error!(i, "Failed to identify built-in operator {expr} (error: {py_err})")
        },
    }
}

fn convert_expr<'py, 'a>(
    expr: Bound<'py, PyAny>, env: &'py ConvertEnv<'py, 'a>
) -> PyResult<Expr> {
    let i = extract_info(&expr, env)?;
    let ty = Type::Unknown;
    if expr.is_instance(&env.ast.getattr("Name")?)? {
        if let Ok(e) = lookup_builtin(&expr, &i) {
            Ok(e)
        } else {
            let id = expr.getattr("id")?.extract::<String>()?;
            Ok(Expr::Var {id, ty, i})
        }
    } else if expr.is_instance(&env.ast.getattr("Constant")?)? {
        let val = expr.getattr("value")?;
        if val.is_instance(&types::PyInt::type_object(val.py()))? {
            let v = val.extract::<i64>()?;
            Ok(Expr::Int {v, ty, i})
        } else if val.is_instance(&types::PyFloat::type_object(val.py()))? {
            let v = val.extract::<f64>()?;
            Ok(Expr::Float {v, ty, i})
        } else if val.is_instance(&types::PyString::type_object(val.py()))? {
            let v = val.extract::<String>()?;
            Ok(Expr::String {v, ty, i})
        } else {
            let ty = expr.get_type();
            py_runtime_error!(i, "Unsupported constant {val:?} of type {ty:?}")
        }
    } else if expr.is_instance(&env.ast.getattr("UnaryOp")?)? {
        let op = convert_unary_op(expr.getattr("op")?, env, &i)?;
        let arg = convert_expr(expr.getattr("operand")?, env)?;
        Ok(Expr::UnOp {op, arg: Box::new(arg), ty, i})
    } else if expr.is_instance(&env.ast.getattr("BinOp")?)? {
        let lhs = convert_expr(expr.getattr("left")?, env)?;
        let op = convert_bin_op(expr.getattr("op")?, env, &i)?;
        let rhs = convert_expr(expr.getattr("right")?, env)?;
        Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
    } else if expr.is_instance(&env.ast.getattr("Compare")?)? {
        let lhs = convert_expr(expr.getattr("left")?, env)?;
        let ops = expr.getattr("ops")?;
        let comps = expr.getattr("comparators")?;
        if ops.len()? == 1 && comps.len()? == 1 {
            let op = convert_bin_op(ops.try_iter()?.next().unwrap()?, env, &i)?;
            let rhs = convert_expr(comps.try_iter()?.next().unwrap()?, env)?;
            Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
        } else {
            py_runtime_error!(i, "Compare nodes with multiple comparisons in sequence are not supported")
        }
    } else if expr.is_instance(&env.ast.getattr("Subscript")?)? {
        let target = convert_expr(expr.getattr("value")?, env)?;
        let idx = convert_expr(expr.getattr("slice")?, env)?;
        Ok(Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i})
    } else if expr.is_instance(&env.ast.getattr("Attribute")?)? {
        lookup_builtin(&expr, &i)
    } else if expr.is_instance(&env.ast.getattr("Tuple")?)? {
        let elts = expr.getattr("elts")?
            .try_iter()?
            .map(|elem| convert_expr(elem?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(Expr::Tuple {elems: elts, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Dict")?)? {
        let keys = expr.getattr("keys")?
            .try_iter()?
            .map(|k| convert_expr(k?, env))
            .map(|k| match k? {
                Expr::String {v, ..} => Ok(v),
                _ => py_runtime_error!(i, "Expected dictionary of string keys")
            })
            .collect::<PyResult<Vec<String>>>()?;
        let values = expr.getattr("values")?
            .try_iter()?
            .map(|v| convert_expr(v?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(Expr::Record {keys, values, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Call")?)? {
        let args = expr.getattr("args")?
            .try_iter()?
            .map(|arg| convert_expr(arg?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        match convert_expr(expr.getattr("func")?, env)? {
            Expr::Builtin {func, args: a, ..} if a.len() == 0 => {
                Ok(Expr::Builtin {func, args, ty, i})
            },
            _ => py_runtime_error!(i, "Function calls are only supported on built-in functions")
        }
    } else {
        py_runtime_error!(i, "Unsupported expression: {expr}")
    }
}

fn convert_stmt<'py, 'a>(
    stmt: Bound<'py, PyAny>,
    env: &'py ConvertEnv<'py, 'a>
) -> PyResult<Stmt> {
    let i = extract_info(&stmt, env)?;
    if stmt.is_instance(&env.ast.getattr("For")?)? {
        // Ensure that the for-loop only assigns to a single variable
        let target = stmt.getattr("target")?;
        let var = if target.is_instance(&env.ast.getattr("Name")?)? {
            Ok(target.getattr("id")?.extract::<String>()?)
        } else {
            py_runtime_error!(i, "For-loops must assign to a single variable")
        }?;

        // Ensure the for-loop iterates over the range builtin
        let iter = stmt.getattr("iter")?;
        let range_fn = if iter.is_instance(&env.ast.getattr("Call")?)? {
            let func = iter.getattr("func")?;
            if func.is_instance(&env.ast.getattr("Name")?)? {
                if func.getattr("id")?.extract::<String>()? == "range" {
                    Ok(iter)
                } else {
                    py_runtime_error!(i, "For-loop must iterate using the range builtin")
                }
            } else {
                py_runtime_error!(i, "For-loop must iterate using the range builtin")
            }
        } else {
            py_runtime_error!(i, "For-loops must iterate using the range builtin")
        }?;

        // Extract the lower and upper bounds of the range. We currently do not support step sizes.
        let range_args = range_fn.getattr("args")?;
        let (lo, hi) = match range_args.len()? {
            1 => {
                let lo = Expr::Int {v: 0, ty: Type::Unknown, i: i.clone()};
                let hi = convert_expr(range_args.get_item(0)?, env)?;
                Ok((lo, hi))
            },
            2 => {
                let lo = convert_expr(range_args.get_item(0)?, env)?;
                let hi = convert_expr(range_args.get_item(1)?, env)?;
                Ok((lo, hi))
            },
            _ => py_runtime_error!(i, "For-loops with a step size are not supported")
        }?;

        let body = convert_stmts(stmt.getattr("body")?, env)?;

        if stmt.getattr("orelse")?.len()? == 0 {
            Ok(Stmt::For {
                var, lo, hi, body, i
            })
        } else {
            py_runtime_error!(i, "For-loops with an else-clause are not supported")
        }
    } else if stmt.is_instance(&env.ast.getattr("If")?)? {
        let cond = convert_expr(stmt.getattr("test")?, env)?;
        let thn = convert_stmts(stmt.getattr("body")?, env)?;
        let els = convert_stmts(stmt.getattr("orelse")?, env)?;
        Ok(Stmt::If {cond, thn, els, i})
    } else if stmt.is_instance(&env.ast.getattr("Assign")?)? {
        let targets = stmt.getattr("targets")?;
        if targets.len()? > 1 {
            py_runtime_error!(i, "Cannot have more than one target of assignment")
        } else {
            let dst = convert_expr(targets.get_item(0)?, env)?;
            let e = convert_expr(stmt.getattr("value")?, env)?;
            match (dst, e) {
                (dst @ (Expr::Var {..} | Expr::Subscript {..}), e) => {
                    Ok(Stmt::Assign {dst: vec![dst], exprs: vec![e], i})
                },
                _ => py_runtime_error!(i, "Unsupported form of assignment")
            }
        }
    } else {
        py_runtime_error!(i, "Unsupported statement: {stmt}")
    }
}

fn merge_body_infos(body: &Vec<Stmt>) -> Info {
    body.iter().fold(Info::default(), |acc, stmt| {
        Info::merge(acc, stmt.get_info())
    })
}

fn convert_stmts<'py, 'a>(
    body: Bound<'py, PyAny>,
    env: &'py ConvertEnv<'py, 'a>
) -> PyResult<Vec<Stmt>> {
    body.try_iter()?
        .map(|stmt| stmt.and_then(|s| convert_stmt(s, &env)))
        .collect::<PyResult<Vec<Stmt>>>()
}

pub fn to_untyped_ir<'py>(
    ast: Bound<'py, PyAny>,
    filepath: String,
    fst_line: usize
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
            let i = Info::default();
            let ty = Type::Unknown;
            Ok(Param {id, ty, i})
        })
        .collect::<PyResult<Vec<Param>>>()?;
    let id = body.getattr("name")?.extract::<String>()?;
    let ir_body = convert_stmts(body.getattr("body")?, &env)?;
    let i = merge_body_infos(&ir_body);
    Ok(vec![FunDef {id, params: untyped_args, body: ir_body, i}])
}
