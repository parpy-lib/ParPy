use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;

use std::collections::BTreeMap;
use std::ffi::CString;

struct ConvertEnv<'py, 'a> {
    ast: Bound<'py, PyModule>,
    globals: Bound<'py, PyDict>,
    locals: Bound<'py, PyDict>,
    tops: &'a BTreeMap<String, Bound<'py, PyCapsule>>,
    filepath: &'a str,
    line_ofs: usize,
    col_ofs: usize
}

fn extract_node_info<'py>(node: &Bound<'py, PyAny>) -> PyResult<Info> {
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
    node: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> Info {
    if let Ok(i) = extract_node_info(node) {
        i.with_file(env.filepath)
            .with_line_offset(env.line_ofs)
            .with_column_offset(env.col_ofs)
    } else {
        Info::default()
    }
}

fn convert_unary_op<'py, 'a>(
    unop: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<UnOp> {
    if unop.is_instance(&env.ast.getattr("USub")?)? {
        Ok(UnOp::Sub)
    } else if unop.is_instance(&env.ast.getattr("Not")?)? {
        Ok(UnOp::Not)
    } else if unop.is_instance(&env.ast.getattr("Invert")?)? {
        Ok(UnOp::BitNeg)
    } else {
        py_runtime_error!(i, "Unsupported unary expression {unop:?}")
    }
}

fn convert_bin_op<'py, 'a>(
    binop: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
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
        Ok(BinOp::Rem)
    } else if binop.is_instance(&env.ast.getattr("Pow")?)? {
        Ok(BinOp::Pow)
    } else if binop.is_instance(&env.ast.getattr("BitAnd")?)? {
        Ok(BinOp::BitAnd)
    } else if binop.is_instance(&env.ast.getattr("BitOr")?)? {
        Ok(BinOp::BitOr)
    } else if binop.is_instance(&env.ast.getattr("BitXor")?)? {
        Ok(BinOp::BitXor)
    } else if binop.is_instance(&env.ast.getattr("LShift")?)? {
        Ok(BinOp::BitShl)
    } else if binop.is_instance(&env.ast.getattr("RShift")?)? {
        Ok(BinOp::BitShr)
    } else if binop.is_instance(&env.ast.getattr("Eq")?)? {
        Ok(BinOp::Eq)
    } else if binop.is_instance(&env.ast.getattr("NotEq")?)? {
        Ok(BinOp::Neq)
    } else if binop.is_instance(&env.ast.getattr("LtE")?)? {
        Ok(BinOp::Leq)
    } else if binop.is_instance(&env.ast.getattr("GtE")?)? {
        Ok(BinOp::Geq)
    } else if binop.is_instance(&env.ast.getattr("Lt")?)? {
        Ok(BinOp::Lt)
    } else if binop.is_instance(&env.ast.getattr("Gt")?)? {
        Ok(BinOp::Gt)
    } else {
        py_runtime_error!(i, "Unsupported binary operation: {binop:?}")
    }
}

fn convert_bool_op<'py, 'a>(
    boolop: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<BinOp> {
    if boolop.is_instance(&env.ast.getattr("And")?)? {
        Ok(BinOp::And)
    } else if boolop.is_instance(&env.ast.getattr("Or")?)? {
        Ok(BinOp::Or)
    } else {
        py_runtime_error!(i, "Unsupported boolean operator: {boolop:?}")
    }
}

fn eval_name<'py, 'a>(
    s: String,
    env: &ConvertEnv<'py, 'a>,
    py: Python<'py>
) -> PyResult<Bound<'py, PyAny>> {
    py.eval(&CString::new(s)?, Some(&env.globals), Some(&env.locals))
}

fn eval_node<'py, 'a>(
    e: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    py: Python<'py>
) -> PyResult<Bound<'py, PyAny>> {
    let s = env.ast.call_method1("unparse", (e,))?.extract::<String>()?;
    eval_name(s, env, py)
}

fn lookup_builtin<'py, 'a>(
    expr: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<Builtin> {
    let py = expr.py();
    let parpy = py.import("parpy")?;
    let parpy_ops = parpy.getattr("operators")?;
    match eval_node(expr, &env, py) {
        Ok(e) => {
            if e.eq(parpy_ops.getattr("exp")?)? {
                Ok(Builtin::Exp)
            } else if e.eq(parpy_ops.getattr("inf")?)? {
                Ok(Builtin::Inf)
            } else if e.eq(parpy_ops.getattr("log")?)? {
                Ok(Builtin::Log)
            } else if e.eq(parpy_ops.getattr("min")?)? {
                Ok(Builtin::Min)
            } else if e.eq(parpy_ops.getattr("max")?)? {
                Ok(Builtin::Max)
            } else if e.eq(parpy_ops.getattr("abs")?)? {
                Ok(Builtin::Abs)
            } else if e.eq(parpy_ops.getattr("cos")?)? {
                Ok(Builtin::Cos)
            } else if e.eq(parpy_ops.getattr("sin")?)? {
                Ok(Builtin::Sin)
            } else if e.eq(parpy_ops.getattr("sqrt")?)? {
                Ok(Builtin::Sqrt)
            } else if e.eq(parpy_ops.getattr("tanh")?)? {
                Ok(Builtin::Tanh)
            } else if e.eq(parpy_ops.getattr("atan2")?)? {
                Ok(Builtin::Atan2)
            } else if e.eq(parpy_ops.getattr("sum")?)? {
                Ok(Builtin::Sum)
            } else if e.eq(parpy_ops.getattr("prod")?)? {
                Ok(Builtin::Prod)
            } else if e.eq(parpy_ops.getattr("float16")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F16})
            } else if e.eq(parpy_ops.getattr("float32")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F32})
            } else if e.eq(parpy_ops.getattr("float64")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F64})
            } else if e.eq(parpy_ops.getattr("int8")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I8})
            } else if e.eq(parpy_ops.getattr("int16")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I16})
            } else if e.eq(parpy_ops.getattr("int32")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I32})
            } else if e.eq(parpy_ops.getattr("int64")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I64})
            } else if e.eq(parpy_ops.getattr("uint8")?)? {
                Ok(Builtin::Convert {sz: ElemSize::U8})
            } else if e.eq(parpy_ops.getattr("uint16")?)? {
                Ok(Builtin::Convert {sz: ElemSize::U16})
            } else if e.eq(parpy_ops.getattr("uint32")?)? {
                Ok(Builtin::Convert {sz: ElemSize::U32})
            } else if e.eq(parpy_ops.getattr("uint64")?)? {
                Ok(Builtin::Convert {sz: ElemSize::U64})
            } else if e.eq(parpy_ops.getattr("label")?)? {
                Ok(Builtin::Label)
            } else if e.eq(parpy_ops.getattr("gpu")?)? {
                Ok(Builtin::GpuContext)
            } else {
                py_runtime_error!(i, "Unknown built-in operator {expr}")
            }
        },
        Err(e) => {
            py_runtime_error!(i, "Failed to identify built-in operator {expr} (reason: {e})")
        },
    }
}

fn lookup_builtin_expr<'py, 'a>(
    expr: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<Expr> {
    let func = lookup_builtin(expr, env, &i)?;
    Ok(Expr::Builtin {func, args: vec![], axis: None, ty: Type::Unknown, i: i.clone()})
}

fn try_extract_type_annotation<'py, 'a>(
    annot: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<Type> {
    let py = annot.py();
    let parpy = py.import("parpy")?;
    let parpy_tys = parpy.getattr("types")?;
    let as_ptr_ty = |sz: Bound<'py, PyAny>| parpy_tys.call_method1("pointer", (sz,));
    match eval_node(&annot, &env, py) {
        Ok(ty) => {
            if ty.eq(parpy_tys.getattr("Bool")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::Bool})
            } else if ty.eq(parpy_tys.getattr("I8")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::I8})
            } else if ty.eq(parpy_tys.getattr("I16")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::I16})
            } else if ty.eq(parpy_tys.getattr("I32")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::I32})
            } else if ty.eq(parpy_tys.getattr("I64")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::I64})
            } else if ty.eq(parpy_tys.getattr("U8")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::U8})
            } else if ty.eq(parpy_tys.getattr("U16")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::U16})
            } else if ty.eq(parpy_tys.getattr("U32")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::U32})
            } else if ty.eq(parpy_tys.getattr("U64")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::U64})
            } else if ty.eq(parpy_tys.getattr("F16")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::F16})
            } else if ty.eq(parpy_tys.getattr("F32")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::F32})
            } else if ty.eq(parpy_tys.getattr("F64")?)? {
                Ok(Type::Tensor {shape: vec![], sz: ElemSize::F64})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("Bool")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::Bool})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("I8")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::I8})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("I16")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::I16})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("I32")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::I32})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("I64")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::I64})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("U8")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::U8})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("U16")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::U16})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("U32")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::U32})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("U64")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::U64})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("F16")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::F16})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("F32")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::F32})
            } else if ty.eq(as_ptr_ty(parpy_tys.getattr("F64")?)?)? {
                Ok(Type::Pointer {sz: ElemSize::F64})
            } else {
                py_runtime_error!(i, "Unsupported parameter type annotation")
            }
        },
        Err(_) => py_runtime_error!(i, "Unsupported parameter type annotation")
    }
}

fn extract_integer_literal_value(e: Expr) -> Option<i64> {
    match e {
        Expr::Int {v, ..} => Some(v as i64),
        Expr::UnOp {op: UnOp::Sub, arg, ..} => match *arg {
            Expr::Int {v, ..} => Some(-v as i64),
            _ => None
        },
        _ => None
    }
}

/// Currently, the only supported keyword argument is the 'axis' keyword, used to determine which
/// axis should be reduced in the supported reduction builtins. If any other keyword arguments are
/// provided, we report an error.
fn extract_axis_kwarg<'py, 'a>(
    acc: PyResult<Option<i64>>,
    kw: Bound<'py, PyAny>,
    i: &Info,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Option<i64>> {
    let kw_str = kw.getattr("arg")?.extract::<String>()?;
    match acc? {
        None if kw_str == "axis" => {
            let kw_val = kw.getattr("value")?;
            match extract_integer_literal_value(convert_expr(kw_val, env)?) {
                Some(v) => Ok(Some(v)),
                None => py_runtime_error!(i, "Expected integer literal value \
                                              for 'axis' keyword argument")
            }
        },
        None | Some(_) => {
            py_runtime_error!(i, "Unsupported keyword argument in call: {kw_str}")
        }
    }
}

fn convert_expr<'py, 'a>(
    expr: Bound<'py, PyAny>, env: &ConvertEnv<'py, 'a>
) -> PyResult<Expr> {
    let i = extract_info(&expr, env);
    let ty = Type::Unknown;
    if expr.is_instance(&env.ast.getattr("Name")?)? {
        if let Ok(e) = lookup_builtin_expr(&expr, env, &i) {
            Ok(e)
        } else {
            let id = Name::new(expr.getattr("id")?.extract::<String>()?);
            Ok(Expr::Var {id, ty, i})
        }
    } else if expr.is_instance(&env.ast.getattr("Constant")?)? {
        let val = expr.getattr("value")?;
        if val.is_instance(&PyBool::type_object(val.py()))? {
            let v = val.extract::<bool>()?;
            Ok(Expr::Bool {v, ty, i})
        } else if val.is_instance(&PyInt::type_object(val.py()))? {
            let v = val.extract::<i128>()?;
            Ok(Expr::Int {v, ty, i})
        } else if val.is_instance(&PyFloat::type_object(val.py()))? {
            let v = val.extract::<f64>()?;
            Ok(Expr::Float {v, ty, i})
        } else if val.is_instance(&PyString::type_object(val.py()))? {
            let v = val.extract::<String>()?;
            Ok(Expr::String {v, ty, i})
        } else {
            py_runtime_error!(i, "Unsupported literal value {val:?}")
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
    } else if expr.is_instance(&env.ast.getattr("BoolOp")?)? {
        let op = convert_bool_op(expr.getattr("op")?, env, &i)?;
        let mut values = expr.getattr("values")?
            .try_iter()?
            .map(|v| convert_expr(v?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        let tail = values.split_off(1);
        let head = values.remove(0);
        Ok(tail.into_iter()
            .fold(head, |acc, v| Expr::BinOp {
                lhs: Box::new(acc), op: op.clone(), rhs: Box::new(v),
                ty: Type::Unknown, i: i.clone()
            }))
    } else if expr.is_instance(&env.ast.getattr("Compare")?)? {
        let lhs = convert_expr(expr.getattr("left")?, env)?;
        let ops = expr.getattr("ops")?;
        let comps = expr.getattr("comparators")?;
        if ops.len()? == 1 && comps.len()? == 1 {
            let op = convert_bin_op(ops.try_iter()?.next().unwrap()?, env, &i)?;
            let rhs = convert_expr(comps.try_iter()?.next().unwrap()?, env)?;
            Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
        } else {
            py_runtime_error!(i, "Sequences of comparisons are not supported")
        }
    } else if expr.is_instance(&env.ast.getattr("IfExp")?)? {
        let cond = Box::new(convert_expr(expr.getattr("test")?, env)?);
        let thn = Box::new(convert_expr(expr.getattr("body")?, env)?);
        let els = Box::new(convert_expr(expr.getattr("orelse")?, env)?);
        Ok(Expr::IfExpr {cond, thn, els, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Subscript")?)? {
        let target = convert_expr(expr.getattr("value")?, env)?;
        let idx = convert_expr(expr.getattr("slice")?, env)?;
        Ok(Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i})
    } else if expr.is_instance(&env.ast.getattr("Slice")?)? {
        let lo = expr.getattr("lower")?;
        let lo = if lo.is_none() {
            None
        } else {
            Some(Box::new(convert_expr(lo, env)?))
        };
        let hi = expr.getattr("upper")?;
        let hi = if hi.is_none() {
            None
        } else {
            Some(Box::new(convert_expr(hi, env)?))
        };
        if !expr.getattr("step")?.is_none() {
            py_runtime_error!(i, "Slices with a step size are not supported")?
        };
        Ok(Expr::Slice {lo, hi, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Attribute")?)? {
        lookup_builtin_expr(&expr, env, &i)
    } else if expr.is_instance(&env.ast.getattr("Tuple")?)? {
        let elts = expr.getattr("elts")?
            .try_iter()?
            .map(|elem| convert_expr(elem?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(Expr::Tuple {elems: elts, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Call")?)? {
        let args = expr.getattr("args")?
            .try_iter()?
            .map(|arg| convert_expr(arg?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        let axis = expr.getattr("keywords")?
            .try_iter()?
            .fold(Ok(None), |acc, kw| extract_axis_kwarg(acc, kw?, &i, env))?;
        match convert_expr(expr.getattr("func")?, env)? {
            Expr::Var {id, ..} => {
                if env.tops.contains_key(&id.to_string()) {
                    if axis.is_none() {
                        Ok(Expr::Call {id, args, ty, i})
                    } else {
                        py_runtime_error!(i, "Unsupported keyword argument in call: axis")
                    }
                } else {
                    py_runtime_error!(i, "Call to unknown function {0}", id.to_string())
                }
            },
            Expr::Builtin {func, args: a, ..} if a.len() == 0 => {
                Ok(Expr::Builtin {func, args, axis, ty, i})
            },
            _ => py_runtime_error!(i, "Unsupported call target type")
        }
    } else {
        py_runtime_error!(i, "Unsupported expression: {expr}")
    }
}

fn extract_step(e: Expr) -> PyResult<i64> {
    let fail = || {
        py_runtime_error!(e.get_info(), "Range step size must be an integer literal")
    };
    let fail_zero = |i: &Info| {
        py_runtime_error!(i, "Range step size must be non-zero")
    };
    match &e {
        Expr::Int {v, ..} if *v != 0 => Ok(*v as i64),
        Expr::UnOp {op: UnOp::Sub, arg, ..} => match arg.as_ref() {
            Expr::Int {v, ..} if *v != 0 => Ok(-*v as i64),
            Expr::Int {i, ..} => fail_zero(&i),
            _ => fail()
        },
        Expr::Int {i, ..} => fail_zero(&i),
        _ => fail()
    }
}

fn construct_expr_stmt(
    value: Expr,
    i: &Info
) -> PyResult<Stmt> {
    let extract_label = |arg| match arg {
        Expr::String {v, ..} => Ok(v),
        _ => {
            let msg = concat!(
                "First argument of parpy.label should be a string literal ",
                "representing the label name"
            );
            py_runtime_error!(i, "{}", msg)
        }
    };
    match value {
        Expr::Builtin {func: Builtin::Label, mut args, ..} => {
            let label = match args.len() {
                1 => extract_label(args.remove(0)),
                _ => py_runtime_error!(i, "Label statement expects one argument")
            }?;
            Ok(Stmt::Label {label, i: i.clone()})
        },
        Expr::Call {id, args, ..} => {
            Ok(Stmt::Call {func: id.to_string(), args, i: i.clone()})
        },
        _ => py_runtime_error!(i, "Unsupported expression statement")
    }
}

fn convert_stmt<'py, 'a>(
    stmt: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Stmt> {
    let i = extract_info(&stmt, env);
    if stmt.is_instance(&env.ast.getattr("For")?)? {
        // Ensure that the for-loop only assigns to a single variable
        let target = stmt.getattr("target")?;
        let var = if target.is_instance(&env.ast.getattr("Name")?)? {
            let s = target.getattr("id")?.extract::<String>()?;
            Ok(Name::new(s))
        } else {
            py_runtime_error!(i, "For-loops must assign to a single variable")
        }?;

        // Ensure the for-loop iterates over the range builtin
        let iter = stmt.getattr("iter")?;
        let range_fn = if iter.is_instance(&env.ast.getattr("Call")?)? {
            let func = iter.getattr("func")?;
            if func.is_instance(&env.ast.getattr("Name")?)? {
                let fun_id = func.getattr("id")?.extract::<String>()?;
                let py = stmt.py();
                let builtins = py.import("builtins")?;
                match eval_name(fun_id, env, py) {
                    Ok(e) if e.eq(builtins.getattr("range")?)? => Ok(iter),
                    _ => py_runtime_error!(i, "For-loop must iterate using the range builtin")
                }
            } else {
                py_runtime_error!(i, "For-loop must iterate using the range builtin")
            }
        } else {
            py_runtime_error!(i, "For-loop must iterate using the range builtin")
        }?;

        // Extract the bounds and the step size of the range.
        let range_args = range_fn.getattr("args")?;
        let (lo, hi, step) = match range_args.len()? {
            1 => {
                let lo = Expr::Int {v: 0, ty: Type::Unknown, i: i.clone()};
                let hi = convert_expr(range_args.get_item(0)?, env)?;
                Ok((lo, hi, 1))
            },
            2 => {
                let lo = convert_expr(range_args.get_item(0)?, env)?;
                let hi = convert_expr(range_args.get_item(1)?, env)?;
                Ok((lo, hi, 1))
            },
            3 => {
                let lo = convert_expr(range_args.get_item(0)?, env)?;
                let hi = convert_expr(range_args.get_item(1)?, env)?;
                let step = extract_step(convert_expr(range_args.get_item(2)?, env)?)?;
                Ok((lo, hi, step))
            }
            _ => py_runtime_error!(i, "Invalid number of arguments passed to range")
        }?;

        let body = convert_stmts(stmt.getattr("body")?, env)?;

        if stmt.getattr("orelse")?.len()? == 0 {
            Ok(Stmt::For {var, lo, hi, step, body, labels: vec![], i})
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
            let expr = convert_expr(stmt.getattr("value")?, env)?;
            match (dst, expr) {
                (dst @ (Expr::Var {..} | Expr::Subscript {..}), expr) => {
                    Ok(Stmt::Assign {dst, expr, labels: vec![], i})
                },
                _ => py_runtime_error!(i, "Unsupported form of assignment")
            }
        }
    } else if stmt.is_instance(&env.ast.getattr("AugAssign")?)? {
        let dst = convert_expr(stmt.getattr("target")?, env)?;
        let op = convert_bin_op(stmt.getattr("op")?, env, &i)?;
        let value = convert_expr(stmt.getattr("value")?, env)?;
        let expr = Expr::BinOp {
            lhs: Box::new(dst.clone()),
            op,
            rhs: Box::new(value),
            ty: Type::Unknown,
            i: i.clone()
        };
        Ok(Stmt::Assign {dst, expr, labels: vec![], i})
    } else if stmt.is_instance(&env.ast.getattr("While")?)? {
        let cond = convert_expr(stmt.getattr("test")?, env)?;
        let body = convert_stmts(stmt.getattr("body")?, env)?;
        if stmt.getattr("orelse")?.len()? == 0 {
            Ok(Stmt::While {cond, body, i})
        } else {
            py_runtime_error!(i, "While-loops with an else-clause are not supported")
        }
    } else if stmt.is_instance(&env.ast.getattr("Expr")?)? {
        let value = convert_expr(stmt.getattr("value")?, env)?;
        construct_expr_stmt(value, &i)
    } else if stmt.is_instance(&env.ast.getattr("With")?)? {
        let items = stmt.getattr("items")?;
        if items.len()? == 1 {
            let fst = items.get_item(0)?;
            if fst.is_instance(&env.ast.getattr("withitem")?)? {
                if !fst.getattr("optional_vars")?.is_none() {
                    py_runtime_error!(i, "With statements using the 'as' keyword are not supported")?
                }
                match lookup_builtin(&fst.getattr("context_expr")?, env, &i) {
                    Ok(Builtin::GpuContext) => {
                        let body = convert_stmts(stmt.getattr("body")?, env)?;
                        Ok(Stmt::WithGpuContext {body, i})
                    },
                    _ => py_runtime_error!(i, "With statements are only supported for 'parpy.gpu'")
                }
            } else {
                let msg = concat!(
                    "Unexpected shape of the AST definition.\n",
                    "This issue may arise because the AST format used by the ",
                    "'ast' module of Python is different from what the ParPy ",
                    "compiler expects. Try using Python version 3.10."
                );
                py_runtime_error!(i, "{}", msg)
            }
        } else {
            py_runtime_error!(i, "With statements using multiple items is not supported")
        }
    } else if stmt.is_instance(&env.ast.getattr("Return")?)? {
        let value = stmt.getattr("value")?;
        if value.is_none() {
            py_runtime_error!(i, "Empty return statements are not supported")
        } else {
            let value = convert_expr(value, env)?;
            Ok(Stmt::Return {value, i})
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
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Vec<Stmt>> {
    body.try_iter()?
        .map(|stmt| stmt.and_then(|s| convert_stmt(s, &env)))
        .collect::<PyResult<Vec<Stmt>>>()
}

fn convert_arg<'py, 'a>(
    arg: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Param> {
    let id = Name::new(arg.getattr("arg")?.extract::<String>()?);
    let i = extract_info(&arg, &env);
    let annot = arg.getattr("annotation")?;
    let ty = if annot.is_none() {
        Ok(Type::Unknown)
    } else {
        try_extract_type_annotation(annot, &env, &i)
    }?;
    Ok(Param {id, ty, i})
}

fn convert_args<'py, 'a>(
    body: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Vec<Param>> {
    body.getattr("args")?.getattr("args")?.try_iter()?
        .map(|arg| convert_arg(arg?, env))
        .collect::<PyResult<Vec<Param>>>()
}

fn convert_fun_def<'py, 'a>(
    ast: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<FunDef> {
    let body = ast.getattr("body")?.get_item(0)?;
    let params = convert_args(&body, env)?;
    let id = Name::new(body.getattr("name")?.extract::<String>()?);
    let ir_body = convert_stmts(body.getattr("body")?, &env)?;
    let i = merge_body_infos(&ir_body);
    Ok(FunDef {id, params, body: ir_body, res_ty: Type::Unknown, i})
}

pub fn to_untyped_ir<'py>(
    ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>)
) -> PyResult<FunDef> {
    let (filepath, line_ofs, col_ofs) = info;
    let (globals, locals) = vars;
    let env = ConvertEnv {
        ast: ast.py().import("ast")?,
        globals,
        locals,
        tops,
        filepath: &filepath,
        line_ofs,
        col_ofs,
    };
    convert_fun_def(ast, &env)
}

fn assert_known_param_types(params: &Vec<Param>) -> PyResult<()> {
    for Param {id: _id, ty, i} in params {
        if let Type::Unknown = ty {
            py_runtime_error!(i, "External declaration parameters must be \
                                  annotated with types.")?
        }
    }
    Ok(())
}

fn convert_returns<'py>(
    ast: Bound<'py, PyAny>,
    env: &ConvertEnv,
    i: &Info
) -> PyResult<Type> {
    let returns = ast.getattr("returns")?;
    try_extract_type_annotation(returns, env, &i)
        .or_else(|_| py_runtime_error!(i, "Unsupported return type annotation \
                                           on external function"))
}

pub fn convert_external<'py>(
    ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    ext_id: String,
    target: Target,
    header: Option<String>,
    par: LoopPar,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>)
) -> PyResult<Top> {
    let (filepath, line_ofs, col_ofs) = info;
    let (globals, locals) = vars;
    let env = ConvertEnv {
        ast: ast.py().import("ast")?,
        globals,
        locals,
        tops: &BTreeMap::new(),
        filepath: &filepath,
        line_ofs,
        col_ofs,
    };
    let body = ast.getattr("body")?.get_item(0)?;
    let i = extract_info(&body, &env);
    let params = convert_args(&body, &env)?;
    let id = Name::new(body.getattr("name")?.extract::<String>()?);
    assert_known_param_types(&params)?;
    let res_ty = convert_returns(body, &env, &i)?;
    Ok(Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    use strum::IntoEnumIterator;

    fn make_env<'py, 'a>(
        py: Python<'py>,
        tops: &'a BTreeMap<String, Bound<'py, PyCapsule>>,
        custom_globals: Option<Bound<'py, PyDict>>
    ) -> PyResult<ConvertEnv<'py, 'a>> {
        let globals = match custom_globals {
            Some(g) => g,
            None => {
                let parpy = py.import("parpy")?;
                let ops = parpy.getattr("operators")?.downcast_into::<PyModule>()?;
                let types = parpy.getattr("types")?.downcast_into::<PyModule>()?;
                vec![
                    ("parpy", parpy),
                    ("parpy.operators", ops),
                    ("parpy.types", types),
                ].into_py_dict(py)?
            }
        };
        let locals = PyDict::new(py);
        Ok(ConvertEnv {
            ast: py.import("ast")?, tops: &tops, globals, locals,
            filepath: "<test>", line_ofs: 0, col_ofs: 0
        })
    }

    fn parse_str<'py>(
        py: Python<'py>,
        s: &str,
        as_expr: bool
    ) -> PyResult<Bound<'py, PyAny>> {
        let ast_module = py.import("ast")?;
        let py_str = PyString::new(py, s);
        let py_args = PyTuple::new(py, vec![py_str])?;
        let py_kwargs = PyDict::new(py);
        if as_expr {
            py_kwargs.set_item("mode", PyString::new(py, "eval"))?;
        }
        ast_module.call_method("parse", py_args, Some(&py_kwargs))
    }

    fn parse_str_fun_def<'py>(
        py: Python<'py>,
        s: &str
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_str(py, s, false)
    }

    fn parse_str_stmts<'py>(
        py: Python<'py>,
        s: &str
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_str_fun_def(py, s)?.getattr("body")
    }

    fn parse_str_stmt<'py>(
        py: Python<'py>,
        s: &str
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_str_stmts(py, s)?.get_item(0)
    }

    fn parse_str_expr<'py>(
        py: Python<'py>,
        s: &str
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_str(py, s, true)?.getattr("body")
    }

    fn lookup_builtin<'py>(
        py: Python<'py>,
        s: &str,
        globals: Option<Bound<'py, PyDict>>
    ) -> PyResult<Expr> {
        let ast = parse_str_expr(py, s)?;
        let tops = BTreeMap::new();
        let env = make_env(py, &tops, globals)?;
        lookup_builtin_expr(&ast, &env, &Info::default())
    }

    fn lookup_builtin_ok(
        s: &str,
        expected_func: Builtin
    ) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = lookup_builtin(py, s, None)?;
            if let Expr::Builtin {func, ..} = e {
                assert_eq!(func, expected_func);
            } else {
                assert!(false)
            }
            Ok(())
        })
    }

    fn lookup_builtin_fail(s: &str) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            assert!(lookup_builtin(py, s, None).is_err());
            Ok(())
        })
    }

    #[test]
    fn lookup_builtin_exp() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.exp", Builtin::Exp)?;
        lookup_builtin_fail("torch.exp")?;
        lookup_builtin_fail("exp")
    }

    #[test]
    fn lookup_builtin_inf() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.inf", Builtin::Inf)?;
        lookup_builtin_ok("float('inf')", Builtin::Inf)?;
        lookup_builtin_fail("torch.inf")?;
        lookup_builtin_fail("inf")
    }

    #[test]
    fn lookup_builtin_log() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.log", Builtin::Log)?;
        lookup_builtin_fail("torch.log")
    }

    #[test]
    fn lookup_builtin_max() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.max", Builtin::Max)
    }

    #[test]
    fn lookup_builtin_min() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.min", Builtin::Min)
    }

    #[test]
    fn lookup_builtin_reduce_ops() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.sum", Builtin::Sum)?;
        lookup_builtin_ok("parpy.operators.prod", Builtin::Prod)
    }

    #[test]
    fn lookup_builtin_abs() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.abs", Builtin::Abs)
    }

    #[test]
    fn lookup_builtin_conversion() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.int8", Builtin::Convert {sz: ElemSize::I8})?;
        lookup_builtin_ok("parpy.operators.int16", Builtin::Convert {sz: ElemSize::I16})?;
        lookup_builtin_ok("parpy.operators.int32", Builtin::Convert {sz: ElemSize::I32})?;
        lookup_builtin_ok("parpy.operators.int64", Builtin::Convert {sz: ElemSize::I64})?;
        lookup_builtin_ok("parpy.operators.uint8", Builtin::Convert {sz: ElemSize::U8})?;
        lookup_builtin_ok("parpy.operators.uint16", Builtin::Convert {sz: ElemSize::U16})?;
        lookup_builtin_ok("parpy.operators.uint32", Builtin::Convert {sz: ElemSize::U32})?;
        lookup_builtin_ok("parpy.operators.uint64", Builtin::Convert {sz: ElemSize::U64})?;
        lookup_builtin_ok("parpy.operators.float16", Builtin::Convert {sz: ElemSize::F16})?;
        lookup_builtin_ok("parpy.operators.float32", Builtin::Convert {sz: ElemSize::F32})?;
        lookup_builtin_ok("parpy.operators.float64", Builtin::Convert {sz: ElemSize::F64})
    }

    #[test]
    fn lookup_builtin_sqrt() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.sqrt", Builtin::Sqrt)?;
        lookup_builtin_fail("torch.sqrt")
    }

    #[test]
    fn lookup_builtin_trigonometry() -> PyResult<()> {
        lookup_builtin_ok("parpy.operators.cos", Builtin::Cos)?;
        lookup_builtin_fail("torch.cos")?;
        lookup_builtin_ok("parpy.operators.sin", Builtin::Sin)?;
        lookup_builtin_fail("torch.sin")?;
        lookup_builtin_ok("parpy.operators.tanh", Builtin::Tanh)?;
        lookup_builtin_fail("torch.tanh")?;
        lookup_builtin_ok("parpy.operators.atan2", Builtin::Atan2)?;
        lookup_builtin_fail("torch.atan2")
    }

    #[test]
    fn lookup_builtin_torch_prefix_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.log")
    }

    #[test]
    fn lookup_builtin_torch_max_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.max")
    }

    #[test]
    fn lookup_builtin_torch_sum_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.sum")
    }

    #[test]
    fn lookup_builtin_torch_prod_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.prod")
    }

    #[test]
    fn lookup_builtin_torch_any_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.any")
    }

    #[test]
    fn lookup_builtin_torch_all_fail() -> PyResult<()> {
        lookup_builtin_fail("torch.all")
    }

    #[test]
    fn lookup_builtin_sum_custom_globals() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let parpy = py.import("parpy")?;
            let globals = vec![
                ("pops", parpy.getattr("operators")?.downcast_into::<PyModule>()?)
            ].into_py_dict(py)?;
            let e = lookup_builtin(py, "pops.sum", Some(globals))?;
            assert!(matches!(e, Expr::Builtin {func: Builtin::Sum, ..}));
            Ok(())
        })
    }

    fn convert_expr_wrap_helper(s: &str, ir_ast_def: Vec<String>) -> PyResult<Expr> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let expr = parse_str_expr(py, s)?;
            let tops = ir_ast_def.into_iter()
                .map(|id| (id, PyCapsule::new(py, 0, None).unwrap()))
                .collect::<BTreeMap<String, Bound<_>>>();
            let env = make_env(py, &tops, None)?;
            convert_expr(expr, &env)
        })
    }

    fn convert_expr_wrap(s: &str) -> PyResult<Expr> {
        convert_expr_wrap_helper(s, vec![])
    }

    fn mkinfo(line1: usize, col1: usize, line2: usize, col2: usize) -> Info {
        Info::new("<test>", FilePos::new(line1, col1), FilePos::new(line2, col2))
    }

    #[test]
    fn convert_expr_variable() {
        let expr = convert_expr_wrap("a").unwrap();
        assert_eq!(expr, Expr::Var {
            id: id("a"),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 1)
        });
    }

    #[test]
    fn convert_expr_int_literal() {
        let expr = convert_expr_wrap("3").unwrap();
        assert_eq!(expr, Expr::Int {
            v: 3,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 1)
        });
    }

    #[test]
    fn convert_expr_float_literal() {
        let expr = convert_expr_wrap("2.718").unwrap();
        assert_eq!(expr, Expr::Float {
            v: 2.718,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_string_literal() {
        let expr = convert_expr_wrap("'hello'").unwrap();
        assert_eq!(expr, Expr::String {
            v: "hello".to_string(),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_expr_bool_literal() {
        let expr = convert_expr_wrap("True").unwrap();
        assert_eq!(expr, Expr::Bool {v: true, ty: Type::Unknown, i: mkinfo(1, 0, 1, 4)});
    }

    #[test]
    fn convert_expr_bytes_literal() {
        let e = convert_expr_wrap("b'123'");
        assert_py_error_matches(e, r"Unsupported literal value b'123'");
    }

    #[test]
    fn convert_expr_unop_int_negation() {
        let expr = convert_expr_wrap("-2").unwrap();
        assert_eq!(expr, Expr::UnOp {
            op: UnOp::Sub,
            arg: Box::new(Expr::Int {
                v: 2,
                ty: Type::Unknown,
                i: mkinfo(1, 1, 1, 2)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 2)
        });
    }

    #[test]
    fn convert_expr_unop_float_negation() {
        let expr = convert_expr_wrap("-3.14").unwrap();
        assert_eq!(expr, Expr::UnOp {
            op: UnOp::Sub,
            arg: Box::new(Expr::Float {
                v: 3.14,
                ty: Type::Unknown,
                i: mkinfo(1, 1, 1, 5)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_unop_not() {
        let expr = convert_expr_wrap("not a").unwrap();
        assert_eq!(expr, Expr::UnOp {
            op: UnOp::Not,
            arg: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 4, 1, 5)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_unop_inv() {
        let expr = convert_expr_wrap("~a").unwrap();
        assert_eq!(expr, Expr::UnOp {
            op: UnOp::BitNeg,
            arg: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 1, 1, 2)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 2)
        });
    }

    fn binop_info(op: BinOp) -> Expr {
        Expr::BinOp {
            lhs: Box::new(Expr::Int {
                v: 1,
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            op,
            rhs: Box::new(Expr::Int {
                v: 2,
                ty: Type::Unknown,
                i: mkinfo(1, 4, 1, 5)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        }
    }

    #[test]
    fn convert_expr_binop_add() {
        let expr = convert_expr_wrap("1 + 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Add));
    }

    #[test]
    fn convert_expr_binop_sub() {
        let expr = convert_expr_wrap("1 - 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Sub));
    }

    #[test]
    fn convert_expr_binop_mul() {
        let expr = convert_expr_wrap("1 * 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Mul));
    }

    #[test]
    fn convert_expr_binop_div() {
        let expr = convert_expr_wrap("1 / 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Div));
    }

    #[test]
    fn convert_expr_binop_floor_div() {
        let expr = convert_expr_wrap("1// 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::FloorDiv));
    }

    #[test]
    fn convert_expr_binop_mod() {
        let expr = convert_expr_wrap("1 % 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Rem));
    }

    #[test]
    fn convert_expr_binop_pow() {
        let expr = convert_expr_wrap("1** 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Pow));
    }

    #[test]
    fn convert_expr_binop_eq() {
        let expr = convert_expr_wrap("1== 2").unwrap();
        assert_eq!(expr, binop_info(BinOp::Eq));
    }

    #[test]
    fn convert_expr_binop_equality_sequence() {
        let e = convert_expr_wrap("a == b == c");
        assert_py_error_matches(e, r"Sequences of comparisons are not supported");
    }

    #[test]
    fn convert_expr_binop_and() {
        let expr = convert_expr_wrap("a and b").unwrap();
        assert_eq!(expr, Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            op: BinOp::And,
            rhs: Box::new(Expr::Var {
                id: id("b"),
                ty: Type::Unknown,
                i: mkinfo(1, 6, 1, 7)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_if_expr() {
        let expr = convert_expr_wrap("0 if x else 1").unwrap();
        assert_eq!(expr, Expr::IfExpr {
            cond: Box::new(Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 5, 1, 6)
            }),
            thn: Box::new(Expr::Int {
                v: 0,
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            els: Box::new(Expr::Int {
                v: 1,
                ty: Type::Unknown,
                i: mkinfo(1, 12, 1, 13)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 13)
        });
    }

    #[test]
    fn convert_expr_string_lookup() {
        let expr = convert_expr_wrap("a['x']").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::String {
                v: "x".to_string(),
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 5)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 6)
        });
    }

    #[test]
    fn convert_expr_int_lookup() {
        let expr = convert_expr_wrap("a[0]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Int {
                v: 0,
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 3)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 4)
        });
    }

    #[test]
    fn convert_expr_multi_dim_lookup() {
        let expr = convert_expr_wrap("a[x, y]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Tuple {
                elems: vec![
                    Expr::Var {id: id("x"), ty: Type::Unknown, i: mkinfo(1, 2, 1, 3)},
                    Expr::Var {id: id("y"), ty: Type::Unknown, i: mkinfo(1, 5, 1, 6)}
                ],
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 6)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_expr_slice() {
        let expr = convert_expr_wrap("a[3:10]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Slice {
                lo: Some(Box::new(Expr::Int {
                    v: 3, ty: Type::Unknown, i: mkinfo(1, 2, 1, 3)
                })),
                hi: Some(Box::new(Expr::Int {
                    v: 10, ty: Type::Unknown, i: mkinfo(1, 4, 1, 6)
                })),
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 6)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_expr_slice_no_lower_bound() {
        let expr = convert_expr_wrap("a[:5]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Slice {
                lo: None,
                hi: Some(Box::new(Expr::Int {
                    v: 5, ty: Type::Unknown, i: mkinfo(1, 3, 1, 4)
                })),
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 4)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_slice_no_upper_bound() {
        let expr = convert_expr_wrap("a[2:]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Slice {
                lo: Some(Box::new(Expr::Int {
                    v: 2, ty: Type::Unknown, i: mkinfo(1, 2, 1, 3)
                })),
                hi: None,
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 4)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_slice_no_bounds() {
        let expr = convert_expr_wrap("a[:]").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Slice {
                lo: None,
                hi: None,
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 3)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 4)
        });
    }

    #[test]
    fn convert_expr_slice_with_step_size() {
        let e = convert_expr_wrap("a[1:10:2]");
        assert_py_error_matches(e, r"Slices with a step size.*not supported");
    }

    #[test]
    fn convert_expr_label() {
        let e = convert_expr_wrap("parpy.operators.label('a')").unwrap();
        assert_eq!(e, Expr::Builtin {
            func: Builtin::Label,
            args: vec![Expr::String {
                v: "a".to_string(),
                ty: Type::Unknown,
                i: mkinfo(1, 14, 1, 17)
            }],
            axis: None,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 18)
        });
    }

    #[test]
    fn convert_expr_sum_no_axis() {
        let e = convert_expr_wrap("parpy.operators.sum(x[:])").unwrap();
        assert_eq!(e, Expr::Builtin {
            func: Builtin::Sum,
            args: vec![Expr::Subscript {
                target: Box::new(Expr::Var {
                    id: id("x"),
                    ty: Type::Unknown,
                    i: mkinfo(1, 12, 1, 13)
                }),
                idx: Box::new(Expr::Slice {
                    lo: None, hi: None, ty: Type::Unknown, i: mkinfo(1, 14, 1, 15)
                }),
                ty: Type::Unknown,
                i: mkinfo(1, 12, 1, 17)
            }],
            axis: None,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 17)
        });
    }

    #[test]
    fn convert_expr_sum_axis() {
        let e = convert_expr_wrap("parpy.operators.sum(x[:,:], axis=1)").unwrap();
        assert_eq!(e, Expr::Builtin {
            func: Builtin::Sum,
            args: vec![Expr::Subscript {
                target: Box::new(Expr::Var {
                    id: id("x"),
                    ty: Type::Unknown,
                    i: mkinfo(1, 12, 1, 13)
                }),
                idx: Box::new(Expr::Tuple {
                    elems: vec![
                        Expr::Slice {
                            lo: None, hi: None, ty: Type::Unknown, i: mkinfo(1, 14, 1, 15)
                        },
                        Expr::Slice {
                            lo: None, hi: None, ty: Type::Unknown, i: mkinfo(1, 16, 1, 17)
                        }
                    ],
                    ty: Type::Unknown,
                    i: mkinfo(1, 14, 1, 17)
                }),
                ty: Type::Unknown,
                i: mkinfo(1, 12, 1, 18)
            }],
            axis: Some(1),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 27)
        });
    }

    #[test]
    fn convert_expr_sum_invalid_axis_form() {
        let e = convert_expr_wrap("parpy.operators.sum(x[:], axis='x')");
        assert_py_error_matches(e, r"Expected integer literal value.*");
    }

    #[test]
    fn convert_expr_call_to_undefined() {
        let e = convert_expr_wrap("f(2, 3)");
        assert_py_error_matches(e, r"Call to unknown function.*");
    }

    #[test]
    fn convert_expr_call_no_args() {
        let e = convert_expr_wrap_helper("f()", vec!["f".to_string()]).unwrap();
        assert_eq!(e, Expr::Call {
            id: Name::new("f".to_string()),
            args: vec![],
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 3)
        });
    }

    #[test]
    fn convert_expr_call_multi_args() {
        let e = convert_expr_wrap_helper("f(2, 3)", vec!["f".to_string()]).unwrap();
        assert_eq!(e, Expr::Call {
            id: Name::new("f".to_string()),
            args: vec![
                Expr::Int {
                    v: 2,
                    ty: Type::Unknown,
                    i: mkinfo(1, 2, 1, 3)
                },
                Expr::Int {
                    v: 3,
                    ty: Type::Unknown,
                    i: mkinfo(1, 5, 1, 6)
                }
            ],
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_expr_call_keyword_args() {
        let e = convert_expr_wrap_helper("f(2, y=3)", vec!["f".to_string()]);
        assert_py_error_matches(e, r"Unsupported keyword argument in call.*y");
    }

    #[test]
    fn convert_expr_call_axis_keyword_arg() {
        let e = convert_expr_wrap_helper("f(2, axis=3)", vec!["f".to_string()]);
        assert_py_error_matches(e, r"Unsupported keyword argument in call.*axis");
    }

    fn convert_stmt_wrap(s: &str) -> PyResult<Stmt> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stmt = parse_str_stmt(py, s)?;
            let tops = BTreeMap::new();
            let env = make_env(py, &tops, None)?;
            convert_stmt(stmt, &env)
        })
    }

    #[test]
    fn convert_stmt_label_invalid_arg() {
        let e = convert_stmt_wrap("parpy.label(4)");
        assert_py_error_matches(e, r"First argument.*should be a string literal");
    }

    #[test]
    fn convert_stmt_label_multi_args() {
        let e = convert_stmt_wrap("parpy.label('a', 'b')");
        assert_py_error_matches(e, r"Label statement expects one argument");
    }

    #[test]
    fn convert_stmt_assignment() {
        let stmt = convert_stmt_wrap("a = 2").unwrap();
        assert_eq!(stmt, Stmt::Assign {
            dst: Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            },
            expr: Expr::Int {
                v: 2,
                ty: Type::Unknown,
                i: mkinfo(1, 4, 1, 5)
            },
            labels: vec![],
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_stmt_assign_multi_target() {
        let e = convert_stmt_wrap("a = b = 2");
        assert_py_error_matches(e, r"Cannot have more than one target of assignment");
    }

    #[test]
    fn convert_stmt_assign_tuple_target() {
        let e = convert_stmt_wrap("a, b = 2, 3");
        assert_py_error_matches(e, r"Unsupported form of assignment");
    }

    #[test]
    fn convert_stmt_aug_assignment() {
        let stmt = convert_stmt_wrap("a += 2").unwrap();
        assert_eq!(stmt, Stmt::Assign {
            dst: Expr::Var {
                id: id("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            },
            expr: Expr::BinOp {
                lhs: Box::new(Expr::Var {
                    id: id("a"),
                    ty: Type::Unknown,
                    i: mkinfo(1, 0, 1, 1)
                }),
                op: BinOp::Add,
                rhs: Box::new(Expr::Int {
                    v: 2,
                    ty: Type::Unknown,
                    i: mkinfo(1, 5, 1, 6)
                }),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 6)
            },
            labels: vec![],
            i: mkinfo(1, 0, 1, 6)
        });
    }

    #[test]
    fn convert_stmt_for_loop_range() {
        let stmt = convert_stmt_wrap("for i in range(1, 10):\n  x[i] = i").unwrap();
        assert_eq!(stmt, Stmt::For {
            var: id("i"),
            lo: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(1, 15, 1, 16)},
            hi: Expr::Int {v: 10, ty: Type::Unknown, i: mkinfo(1, 18, 1, 20)},
            step: 1,
            body: vec![
                Stmt::Assign {
                    dst: Expr::Subscript {
                        target: Box::new(Expr::Var {
                            id: id("x"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 2, 2, 3)
                        }),
                        idx: Box::new(Expr::Var {
                            id: id("i"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 4, 2, 5)
                        }),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 6)
                    },
                    expr: Expr::Var {
                        id: id("i"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 9, 2, 10)
                    },
                    labels: vec![],
                    i: mkinfo(2, 2, 2, 10)
                }
            ],
            labels: vec![],
            i: mkinfo(1, 0, 2, 10)
        })
    }

    #[test]
    fn convert_stmt_for_range_negative_step() {
        let stmt = convert_stmt_wrap("for i in range(10, 1, -2):\n  x[i] = i").unwrap();
        assert_eq!(stmt, Stmt::For {
            var: id("i"),
            lo: Expr::Int {v: 10, ty: Type::Unknown, i: mkinfo(1, 15, 1, 17)},
            hi: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(1, 19, 1, 20)},
            step: -2,
            body: vec![
                Stmt::Assign {
                    dst: Expr::Subscript {
                        target: Box::new(Expr::Var {
                            id: id("x"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 2, 2, 3)
                        }),
                        idx: Box::new(Expr::Var {
                            id: id("i"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 4, 2, 5)
                        }),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 6)
                    },
                    expr: Expr::Var {
                        id: id("i"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 9, 2, 10)
                    },
                    labels: vec![],
                    i: mkinfo(2, 2, 2, 10)
                }
            ],
            labels: vec![],
            i: mkinfo(1, 0, 2, 10)
        });
    }

    #[test]
    fn convert_stmt_for_in_loop_fail() {
        let e = convert_stmt_wrap("for x in s:\n  x = x + 1");
        assert_py_error_matches(e, r".*must iterate using the range builtin");
    }

    #[test]
    fn convert_stmt_for_invalid_range_call() {
        let e = convert_stmt_wrap("for x in range(1, 2, 3, 4):\n  a = 1");
        assert_py_error_matches(e, r"Invalid number of arguments passed to range");
    }

    #[test]
    fn convert_stmt_for_with_else_clause() {
        let e = convert_stmt_wrap("for x in range(1, 2):\n  a = 1\nelse:\n  a = 2");
        assert_py_error_matches(e, r"else-clause.*not supported.*");
    }

    #[test]
    fn convert_stmt_if_cond() {
        let stmt = convert_stmt_wrap("if x:\n  y = 1\nelse:\n  y = 2").unwrap();
        assert_eq!(stmt, Stmt::If {
            cond: Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 3, 1, 4)
            },
            thn: vec![
                Stmt::Assign {
                    dst: Expr::Var {
                        id: id("y"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 3)
                    },
                    expr: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(2, 6, 2, 7)},
                    labels: vec![],
                    i: mkinfo(2, 2, 2, 7)
                }
            ],
            els: vec![
                Stmt::Assign {
                    dst: Expr::Var {
                        id: id("y"),
                        ty: Type::Unknown,
                        i: mkinfo(4, 2, 4, 3)
                    },
                    expr: Expr::Int {v: 2, ty: Type::Unknown, i: mkinfo(4, 6, 4, 7)},
                    labels: vec![],
                    i: mkinfo(4, 2, 4, 7)
                }
            ],
            i: mkinfo(1, 0, 4, 7)
        });
    }

    #[test]
    fn convert_while_stmt() {
        let stmt = convert_stmt_wrap("while True:\n  y = 1").unwrap();
        assert_eq!(stmt, Stmt::While {
            cond: Expr::Bool {v: true, ty: Type::Unknown, i: mkinfo(1, 6, 1, 10)},
            body: vec![
                Stmt::Assign {
                    dst: Expr::Var {
                        id: id("y"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 3)
                    },
                    expr: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(2, 6, 2, 7)},
                    labels: vec![],
                    i: mkinfo(2, 2, 2, 7)
                }
            ],
            i: mkinfo(1, 0, 2, 7)
        });
    }

    #[test]
    fn convert_while_else_stmt() {
        let e = convert_stmt_wrap("while True:\n  y = 1\nelse:\n  y = 2");
        assert_py_error_matches(e, r".*else-clause.*");
    }

    #[test]
    fn convert_return_stmt() {
        let stmt = convert_stmt_wrap("return 3").unwrap();
        assert_eq!(stmt, Stmt::Return {
            value: Expr::Int {v: 3, ty: Type::Unknown, i: mkinfo(1, 7, 1, 8)},
            i: mkinfo(1, 0, 1, 8)
        });
    }

    #[test]
    fn convert_empty_return_stmt() {
        let e = convert_stmt_wrap("return");
        assert_py_error_matches(e, r"Empty return statement.*");
    }

    #[test]
    fn convert_with_gpu_context_stmt() {
        let stmt = convert_stmt_wrap("with parpy.gpu:\n  a = 2").unwrap();
        assert_eq!(stmt, Stmt::WithGpuContext {
            body: vec![Stmt::Assign {
                dst: Expr::Var {
                    id: id("a"),
                    ty: Type::Unknown,
                    i: mkinfo(2, 2, 2, 3)
                },
                expr: Expr::Int {v: 2, ty: Type::Unknown, i: mkinfo(2, 6, 2, 7)},
                labels: vec![],
                i: mkinfo(2, 2, 2, 7)
            }],
            i: mkinfo(1, 0, 2, 7)
        });
    }

    #[test]
    fn convert_with_as_err() {
        let e = convert_stmt_wrap("with parpy.gpu as x:\n  a = 2");
        assert_py_error_matches(e, r"With statements.*'as' keyword.*");
    }

    #[test]
    fn convert_with_unknown_context() {
        let e = convert_stmt_wrap("with ctx:\n  a = 2");
        assert_py_error_matches(e, r"With statements are only supported for.*");
    }

    #[test]
    fn convert_with_multi_items() {
        let e = convert_stmt_wrap("with a, b:\n  c = 2");
        assert_py_error_matches(e, r"With statements using multiple items.*");
    }

    fn convert_stmts_wrap(s: &str) -> PyResult<Vec<Stmt>> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stmt = parse_str_stmts(py, s)?;
            let tops = BTreeMap::new();
            let env = make_env(py, &tops, None)?;
            convert_stmts(stmt, &env)
        })
    }

    #[test]
    fn convert_for_overloaded_range() {
        let res = convert_stmts_wrap("range = 3\n for x in range(1, 2):\n  x = x + 1");
        assert!(res.is_err());
    }

    fn convert_arg_type_annot(annot: &str) -> PyResult<Type> {
        let s = format!("def f(x: {annot}):\n  return x");
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stmts = parse_str_fun_def(py, &s)?;
            let tops = BTreeMap::new();
            let env = make_env(py, &tops, None)?;
            let FunDef {params, ..} = convert_fun_def(stmts, &env)?;
            match &params[..] {
                [Param {ty, ..}] => Ok(ty.clone()),
                _ => panic!("Invalid form of parameters in converted function definition")
            }
        })
    }

    #[test]
    fn try_extract_scalar_type_annot() -> PyResult<()> {
        for sz in ElemSize::iter() {
            let id = format!("parpy.types.{:?}", sz);
            assert_eq!(convert_arg_type_annot(&id)?, scalar(sz));
        }
        Ok(())
    }

    #[test]
    fn try_extract_pointer_type_annot() {
        let ty = convert_arg_type_annot("parpy.types.pointer(parpy.types.I8)").unwrap();
        assert_eq!(ty, Type::Pointer {sz: ElemSize::I8});
    }

    #[test]
    fn try_extract_int_type_annot() {
        assert_py_error_matches(
            convert_arg_type_annot("int"),
            "Unsupported parameter type annotation"
        )
    }
}
