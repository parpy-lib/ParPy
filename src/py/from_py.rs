use super::ast::*;
use super::for_loops;
use crate::py_runtime_error;
use crate::py_internal_error;
use crate::ext::types::{ExtType, TypeVar};
use crate::option::CompileBackend;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;

use std::collections::BTreeMap;
use std::ffi::CString;

#[derive(Debug)]
pub struct ConvertEnv<'py, 'a> {
    pub ast: Bound<'py, PyModule>,
    pub globals: Bound<'py, PyDict>,
    pub locals: Bound<'py, PyDict>,
    pub tops: &'a BTreeMap<String, Bound<'py, PyCapsule>>,
    pub filepath: &'a str,
    pub line_ofs: usize,
    pub col_ofs: usize
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

pub fn eval_node<'py, 'a>(
    e: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    py: Python<'py>
) -> PyResult<Bound<'py, PyAny>> {
    let s = env.ast.call_method1("unparse", (e,))?.extract::<String>()?;
    eval_name(s, env, py)
}

fn convert_binary_extrema_builtin<'py, 'a>(
    op: BinOp,
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 2 {
        let rhs = convert_expr(args.pop().unwrap(), env)?;
        let lhs = convert_expr(args.pop().unwrap(), env)?;
        Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty: Type::Unknown, i})
    } else {
        let op = op.pprint_default();
        py_runtime_error!(i, "{n} arguments were provided to binary builtin {op}")
    }
}

fn convert_reduction_builtin<'py, 'a>(
    op: ReduceOp,
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 1 {
        let arg = convert_expr(args.pop().unwrap(), env)?;
        Ok(Expr::ReduceOp {op, arg: Box::new(arg), ty: Type::Unknown, i})
    } else {
        let op = op.pprint_default();
        py_runtime_error!(i, "{n} arguments were provided to unary reduction builtin {op}")
    }
}

fn convert_type_conversion_builtin<'py, 'a>(
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 2 {
        let ty = match try_extract_type_annotation(args.pop().unwrap(), env, &i) {
            Ok(ty) => Ok(ty),
            Err(_) => py_runtime_error!(i, "Second argument of conversion builtin \
                                            must be a ParPy type.")
        }?;
        let e = match convert_expr(args.pop().unwrap(), env) {
            Ok(e) => Ok(e),
            Err(_) => py_runtime_error!(i, "First argument of conversion builtin \
                                            must be an expression.")
        }?;
        Ok(Expr::Convert {e: Box::new(e), ty, i})
    } else {
        py_runtime_error!(i, "Type conversion expects two arguments but found {n}")
    }
}

fn convert_builtin_string_arg<'py, 'a>(
   mut args: Vec<Bound<'py, PyAny>>,
   env: &ConvertEnv<'py, 'a>,
   i: &Info,
   id: &str
) -> PyResult<String> {
    let n = args.len();
    if n == 1 {
        match convert_expr(args.pop().unwrap(), env)? {
            Expr::String {v, ..} => Ok(v),
            _ => py_runtime_error!(i, "{id} builtin expects a string argument")
        }
    } else {
        py_runtime_error!(i, "{id} builtin expects one argument but found {n}")
    }
}

fn convert_label_builtin<'py, 'a>(
    args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let label = convert_builtin_string_arg(args, &env, &i, "Label")?;
    Ok(Expr::Label {label, ty: Type::Unknown, i})
}

fn convert_fail_builtin<'py, 'a>(
    args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let msg = convert_builtin_string_arg(args, &env, &i, "Fail")?;
    Ok(Expr::StaticFail {msg, ty: Type::Unknown, i})
}

fn convert_inline_builtin<'py, 'a>(
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 1 {
        match convert_expr(args.pop().unwrap(), env)? {
            e @ Expr::Call {..} => Ok(Expr::Inline {e: Box::new(e), ty: Type::Unknown, i}),
            _ => py_runtime_error!(i, "Inline expects a call expression argument")
        }
    } else {
        py_runtime_error!(i, "Inline builtin expects one argument but found {n}")
    }
}

fn try_extract_literal_shape<'py, 'a>(
    arg: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<Vec<Expr>> {
    match convert_expr(arg, env) {
        Ok(Expr::Tuple {elems, ..}) => Ok(elems),
        Ok(_) => py_runtime_error!(i, "Expected tuple of dimensions describing the shape"),
        Err(e) => py_runtime_error!(i, "Failed to parse literal shape: {e}"),
    }
}

fn convert_shared_alloc_builtin<'py, 'a>(
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    // The first argument is the shape of the allocated memory, represented as a list/tuple of
    // statically known integers corresponding to its dimensions. The second argument is the type
    // of the elements in the array.
    if n == 2 {
        let sz = match try_extract_type_annotation(args.pop().unwrap(), env, &i) {
            Ok(Type::Tensor {ref shape, sz}) if shape.is_empty() => Ok(sz),
            _ => py_runtime_error!(i, "Second argument of shared allocation \
                                       builtin must be a scalar ParPy type.")
        }?;
        let lit_shape = try_extract_literal_shape(args.pop().unwrap(), env, &i)?;
        Ok(Expr::AllocShared {shape: lit_shape, sz, ty: Type::Unknown, i})
    } else {
        py_runtime_error!(i, "Shared allocation builtin expects two arguments but found {n}")
    }
}

fn convert_static_backend_equality<'py, 'a>(
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 1 {
        let arg = args.pop().unwrap();
        let py = arg.py();
        match eval_node(&arg, &env, py) {
            Ok(v) => {
                if let Ok(backend) = v.extract::<CompileBackend>() {
                    Ok(Expr::StaticBackendEq {backend, ty: Type::Unknown, i})
                } else {
                    py_runtime_error!(i, "Static backend equality expects a \
                                          CompileBackend argument")
                }
            },
            Err(e) => py_runtime_error!(i, "Failed to resolve backend: {e}")
        }
    } else {
        py_runtime_error!(i, "Static backend equality expects one argument but found {n}")
    }
}

fn convert_static_types_equality<'py, 'a>(
    mut args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Expr> {
    let n = args.len();
    if n == 2 {
        let rhs = match try_extract_type_annotation(args.pop().unwrap(), env, &i) {
            Ok(Type::Tensor {sz, ..}) => Ok(sz),
            _ => py_runtime_error!(i, "Second argument of static type \
                                       equality must be a ParPy type.")
        }?;
        let lhs = match try_extract_type_annotation(args.pop().unwrap(), env, &i) {
            Ok(Type::Tensor {sz, ..}) => Ok(sz),
            _ => py_runtime_error!(i, "First argument of static type \
                                       equality must be a ParPy type.")
        }?;
        Ok(Expr::StaticTypesEq {lhs, rhs, ty: Type::Unknown, i})
    } else {
        py_runtime_error!(i, "Static types equality expects two arguments but found {n}")
    }
}

fn convert_builtin<'py, 'a>(
    func: &Bound<'py, PyAny>,
    args: Vec<Bound<'py, PyAny>>,
    env: &ConvertEnv<'py, 'a>,
    i: Info
) -> PyResult<Option<Expr>> {
    let py = func.py();
    let parpy = py.import("parpy")?;
    let parpy_builtins = parpy.getattr("builtin")?;
    let parpy_reduce = parpy.getattr("reduce")?;
    match eval_node(&func, &env, py) {
        Ok(e) => {
            // Constants
            let res = if e.eq(parpy_builtins.getattr("inf")?)? {
                Some(Expr::Float {v: f64::INFINITY, ty: Type::Unknown, i})

            // Binary operations for computing max/min
            } else if e.eq(parpy_builtins.getattr("maximum")?)? {
                Some(convert_binary_extrema_builtin(BinOp::Max, args, env, i)?)
            } else if e.eq(parpy_builtins.getattr("minimum")?)? {
                Some(convert_binary_extrema_builtin(BinOp::Min, args, env, i)?)

            // Reduction operators
            } else if e.eq(parpy_reduce.getattr("max")?)? {
                Some(convert_reduction_builtin(ReduceOp::Max, args, env, i)?)
            } else if e.eq(parpy_reduce.getattr("min")?)? {
                Some(convert_reduction_builtin(ReduceOp::Min, args, env, i)?)
            } else if e.eq(parpy_reduce.getattr("prod")?)? {
                Some(convert_reduction_builtin(ReduceOp::Prod, args, env, i)?)
            } else if e.eq(parpy_reduce.getattr("sum")?)? {
                Some(convert_reduction_builtin(ReduceOp::Sum, args, env, i)?)

            // Type conversion
            } else if e.eq(parpy_builtins.getattr("convert")?)? {
                Some(convert_type_conversion_builtin(args, env, i)?)

            // GPU context (only usable in a 'with' statement)
            } else if e.eq(parpy_builtins.getattr("gpu")?)? {
                Some(Expr::GpuContext {ty: Type::Unknown, i})

            // Labeling (only usable as a statement)
            } else if e.eq(parpy_builtins.getattr("label")?)? {
                Some(convert_label_builtin(args, env, i)?)

            // Inlining of calls (only usable as a statement)
            } else if e.eq(parpy_builtins.getattr("inline")?)? {
                Some(convert_inline_builtin(args, env, i)?)

            // Manual allocation of shared memory (only usable as a statement)
            } else if e.eq(parpy_builtins.getattr("alloc_shared")?)? {
                Some(convert_shared_alloc_builtin(args, env, i)?)

            // Statically evaluated nodes used for compile-time specialization
            } else if e.eq(parpy_builtins.getattr("static_backend_eq")?)? {
                Some(convert_static_backend_equality(args, env, i)?)
            } else if e.eq(parpy_builtins.getattr("static_types_eq")?)? {
                Some(convert_static_types_equality(args, env, i)?)
            } else if e.eq(parpy_builtins.getattr("static_fail")?)? {
                Some(convert_fail_builtin(args, env, i)?)
            } else {
                None
            };
            Ok(res)
        },
        Err(_) => Ok(None)
    }
}

fn try_extract_type_annotation<'py, 'a>(
    annot: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>,
    i: &Info
) -> PyResult<Type> {
    let fail = || py_runtime_error!(i, "Unsupported parameter type annotation");
    let py = annot.py();
    match eval_node(&annot, &env, py) {
        Ok(ty) => match ty.extract::<ElemSize>() {
            Ok(sz) => Ok(Type::fixed_scalar(sz)),
            Err(_) => match ty.extract::<TypeVar>() {
                Ok(var) => {
                    let sz = TensorElemSize::Variable {id: var.id};
                    Ok(Type::Tensor {sz, shape: vec![]})
                },
                Err(_) => match ty.extract::<ExtType>() {
                    Ok(ExtType::Buffer(sz, syms)) => {
                        let sz = TensorElemSize::Fixed {sz};
                        let shape = syms.into_iter()
                            .map(|sym| TensorShape::Symbol {id: sym.id})
                            .collect::<Vec<TensorShape>>();
                        Ok(Type::Tensor {sz, shape})
                    },
                    Ok(ExtType::VarBuffer(tyvar, syms)) => {
                        let sz = TensorElemSize::Variable {id: tyvar.id};
                        let shape = syms.into_iter()
                            .map(|sym| TensorShape::Symbol {id: sym.id})
                            .collect::<Vec<TensorShape>>();
                        Ok(Type::Tensor {sz, shape})
                    },
                    Err(_) => fail()
                }
            }
        },
        Err(_) => fail()
    }
}

fn ensure_no_keyword_arguments<'py>(e: &Bound<'py, PyAny>, i: &Info) -> PyResult<()> {
    if e.getattr("keywords")?.try_iter()?.count() > 0 {
        py_runtime_error!(i, "Keyword arguments are not supported in call nodes")
    } else {
        Ok(())
    }
}

fn try_get_qualified_name<'py>(
    e: &Bound<'py, PyAny>
) -> PyResult<String> {
    let py = e.py();
    let inspect = py.import("inspect")?;
    let module = inspect.call_method("getmodule", (e,), None)?;
    let module_id = module.getattr("__name__")?;
    let func_id = e.getattr("__name__")?;
    Ok(format!("{module_id}.{func_id}"))
}

pub fn convert_expr<'py, 'a>(
    expr: Bound<'py, PyAny>, env: &ConvertEnv<'py, 'a>
) -> PyResult<Expr> {
    let i = extract_info(&expr, env);
    let ty = Type::Unknown;
    if expr.is_instance(&env.ast.getattr("Name")?)? {
        if let Ok(Some(e)) = convert_builtin(&expr, vec![], env, i.clone()) {
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
        match convert_builtin(&expr, vec![], env, i.clone()) {
            Ok(Some(e)) => Ok(e),
            Ok(None) => py_runtime_error!(i, "Unknown attribute {expr}"),
            Err(e) => Err(e)
        }
    } else if expr.is_instance(&env.ast.getattr("Tuple")?)? {
        let elts = expr.getattr("elts")?
            .try_iter()?
            .map(|elem| convert_expr(elem?, env))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(Expr::Tuple {elems: elts, ty, i})
    } else if expr.is_instance(&env.ast.getattr("Call")?)? {
        ensure_no_keyword_arguments(&expr, &i)?;
        let func = expr.getattr("func")?;
        let args = expr.getattr("args")?
            .try_iter()?
            .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?;
        if let Some(builtin) = convert_builtin(&func, args, &env, i.clone())? {
            Ok(builtin)
        } else {
            let py = expr.py();
            let fun = eval_node(&func, &env, py)?;
            match try_get_qualified_name(&fun) {
                Ok(qualified_name) => {
                    if env.tops.contains_key(&qualified_name) {
                        let id = Name::sym_str(&qualified_name);
                        let args = expr.getattr("args")?
                            .try_iter()?
                            .map(|arg| convert_expr(arg?, env))
                            .collect::<PyResult<Vec<Expr>>>()?;
                        Ok(Expr::Call {id, args, ty, i})
                    } else {
                        py_runtime_error!(i, "Call to unknown ParPy function \
                                              {qualified_name}.")
                    }
                },
                Err(e) => {
                    py_internal_error!(i, "Failed to find qualified name of \
                                           function: {e}")
                }
            }
        }
    } else {
        py_runtime_error!(i, "Unsupported expression: {expr}")
    }
}

fn construct_expr_stmt(
    value: Expr,
    i: &Info
) -> PyResult<Stmt> {
    match value {
        Expr::Inline {..} | Expr::Label {..} | Expr::StaticFail {..} |
        Expr::Call {..} => {
            Ok(Stmt::Expr {e: value, i: i.clone()})
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
        for_loops::convert_for_loop(stmt, i, env)
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
                match convert_builtin(&fst.getattr("context_expr")?, vec![], env, i.clone())? {
                    Some(Expr::GpuContext {..}) => {
                        let body = convert_stmts(stmt.getattr("body")?, env)?;
                        Ok(Stmt::WithGpuContext {body, i})
                    },
                    _ => {
                        py_runtime_error!(i, "With statements are only supported for 'parpy.gpu'")
                    }
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

pub fn convert_stmts<'py, 'a>(
    body: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Vec<Stmt>> {
    body.try_iter()?
        .map(|stmt| stmt.and_then(|s| convert_stmt(s, &env)))
        .collect::<PyResult<Vec<Stmt>>>()
}

fn convert_param<'py, 'a>(
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

fn convert_params<'py, 'a>(
    body: &Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Vec<Param>> {
    body.getattr("args")?.getattr("args")?.try_iter()?
        .map(|arg| convert_param(arg?, env))
        .collect::<PyResult<Vec<Param>>>()
}

fn strip_docstring<'py, 'a>(
    body: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<Bound<'py, PyAny>> {
    let py = body.py();
    let stmts = body.getattr("body")?;
    let fst_stmt = stmts.get_item(0)?;
    if fst_stmt.is_instance(&env.ast.getattr("Expr")?)? {
        let expr = fst_stmt.getattr("value")?;
        if expr.is_instance(&env.ast.getattr("Constant")?)? {
            let value = expr.getattr("value")?;
            if value.is_instance(&PyString::type_object(py))? {
                let body = stmts.try_iter()?
                    .skip(1)
                    .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?;
                Ok(body.into_pyobject(py)?)
            } else {
                Ok(stmts)
            }
        } else {
            Ok(stmts)
        }
    } else {
        Ok(stmts)
    }
}

fn convert_fun_def<'py, 'a>(
    ast: Bound<'py, PyAny>,
    env: &ConvertEnv<'py, 'a>
) -> PyResult<FunDef> {
    let body = ast.getattr("body")?.get_item(0)?;
    let params = convert_params(&body, &env)?;
    let id = Name::new(body.getattr("name")?.extract::<String>()?);
    let body = strip_docstring(body, &env)?;
    let ir_body = convert_stmts(body, &env)?;
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
        .or(Ok(Type::Void))
}

pub fn convert_callback<'py>(
    ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
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
        col_ofs
    };
    let body = ast.getattr("body")?.get_item(0)?;
    let id = Name::new(body.getattr("name")?.extract::<String>()?);
    let params = convert_params(&body, &env)?;
    let i = extract_info(&body, &env);
    Ok(Top::CallbackDecl {id, params, i})
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
    let params = convert_params(&body, &env)?;
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

    fn make_env<'py, 'a>(
        py: Python<'py>,
        tops: &'a BTreeMap<String, Bound<'py, PyCapsule>>,
        custom_globals: Option<Bound<'py, PyDict>>
    ) -> PyResult<ConvertEnv<'py, 'a>> {
        let globals = match custom_globals {
            Some(g) => g,
            None => {
                let parpy = py.import("parpy")?;
                let ops = parpy.getattr("builtin")?.downcast_into::<PyModule>()?;
                let types = parpy.getattr("types")?.downcast_into::<PyModule>()?;
                vec![
                    ("parpy", parpy),
                    ("parpy.builtin", ops),
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
        let (func, args) = if ast.hasattr("func")? {
            let func = ast.getattr("func")?;
            let args = ast.getattr("args")?
                .try_iter()?
                .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?;
            (func, args)
        } else {
            (ast, vec![])
        };
        match convert_builtin(&func, args, &env, i())? {
            Some(e) => Ok(e),
            None => py_runtime_error!(i(), "Failed to find built-in")
        }
    }

    fn lookup_builtin_ok(
        s: &str,
        expected: Expr
    ) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let e = lookup_builtin(py, s, None)?;
            assert_eq!(e, expected);
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
    fn lookup_builtin_inf() -> PyResult<()> {
        let expected = Expr::Float {
            v: f64::INFINITY,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 19)
        };
        lookup_builtin_ok("parpy.builtin.inf", expected)?;
        lookup_builtin_fail("torch.inf")?;
        lookup_builtin_fail("inf")
    }

    #[test]
    fn lookup_max_reduce() -> PyResult<()> {
        let expected = Expr::ReduceOp {
            op: ReduceOp::Max,
            arg: Box::new(Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 20, 1, 21)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 22)
        };
        lookup_builtin_ok("parpy.reduce.max(x)", expected)
    }

    #[test]
    fn lookup_min_reduce() -> PyResult<()> {
        let expected = Expr::ReduceOp {
            op: ReduceOp::Min,
            arg: Box::new(Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 20, 1, 21)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 22)
        };
        lookup_builtin_ok("parpy.reduce.min(x)", expected)
    }

    #[test]
    fn lookup_sum_reduce() -> PyResult<()> {
        let expected = Expr::ReduceOp {
            op: ReduceOp::Sum,
            arg: Box::new(Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 20, 1, 21)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 22)
        };
        lookup_builtin_ok("parpy.reduce.sum(x)", expected)
    }

    #[test]
    fn lookup_prod_reduce() -> PyResult<()> {
        let expected = Expr::ReduceOp {
            op: ReduceOp::Prod,
            arg: Box::new(Expr::Var {
                id: id("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 21, 1, 22)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 23)
        };
        lookup_builtin_ok("parpy.reduce.prod(x)", expected)
    }

    #[test]
    fn lookup_sum_reduce_custom_globals() -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let parpy = py.import("parpy")?;
            let globals = vec![
                ("parpy_reduce", parpy.getattr("reduce")?.downcast_into::<PyModule>()?)
            ].into_py_dict(py)?;
            let e = lookup_builtin(py, "parpy_reduce.sum(x)", Some(globals))?;
            assert!(matches!(e, Expr::ReduceOp {op: ReduceOp::Sum, ..}));
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
        let e = convert_expr_wrap("parpy.builtin.label('a')").unwrap();
        assert_eq!(e, Expr::Label {
            label: "a".to_string(),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 18)
        });
    }

    #[test]
    fn convert_expr_sum() {
        let e = convert_expr_wrap("parpy.reduce.sum(x[:])").unwrap();
        assert_eq!(e, Expr::ReduceOp {
            op: ReduceOp::Sum,
            arg: Box::new(Expr::Subscript {
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
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 17)
        });
    }

    #[test]
    fn convert_expr_sum_kwarg() {
        let e = convert_expr_wrap("parpy.builtin.sum(x[:,:], axis=1)");
        assert_py_error_matches(e, "Keyword arguments are not supported.*");
    }

    #[test]
    fn convert_expr_sum_invalid_axis_form() {
        let e = convert_expr_wrap("parpy.builtin.sum(x[:], axis='x')");
        assert_py_error_matches(e, "Keyword arguments are not supported.*");
    }

    #[test]
    fn convert_expr_call_keyword_args() {
        let e = convert_expr_wrap_helper("f(2, y=3)", vec!["f".to_string()]);
        assert_py_error_matches(e, "Keyword arguments are not supported.*");
    }

    #[test]
    fn convert_expr_call_axis_keyword_arg() {
        let e = convert_expr_wrap_helper("f(2, axis=3)", vec!["f".to_string()]);
        assert_py_error_matches(e, "Keyword arguments are not supported.*");
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
        assert_py_error_matches(e, r"Label builtin expects a string argument");
    }

    #[test]
    fn convert_stmt_label_multi_args() {
        let e = convert_stmt_wrap("parpy.label('a', 'b')");
        assert_py_error_matches(e, r"Label builtin expects one argument but found 2");
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
            step: Expr::Int {v: 1, ty: Type::Unknown, i: Info::default()},
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
            step: Expr::UnOp {
                op: UnOp::Sub,
                arg: Box::new(Expr::Int {v: 2, ty: Type::Unknown, i: mkinfo(1, 22, 1, 23)}),
                ty: Type::Unknown,
                i: mkinfo(1, 22, 1, 24)
            },
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
        assert_py_error_matches(e, r".*must iterate using.*");
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

    fn convert_param_type_annot(annot: &str) -> PyResult<Type> {
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
    fn try_extract_int_type_annot() {
        assert_py_error_matches(
            convert_param_type_annot("int"),
            "Unsupported parameter type annotation"
        )
    }
}
