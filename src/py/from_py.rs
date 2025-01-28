use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use super::ast::*;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types;

use std::collections::BTreeMap;
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
) -> Info {
    if let Ok(i) = extract_node_info(node) {
        i.with_file(env.filepath)
            .with_line_offset(env.fst_line)
    } else {
        Info::default()
    }
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

fn eval_name<'py>(s: String, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let globals = types::PyDict::new(py);
    globals.set_item("math", py.import("math")?)?;
    globals.set_item("parir", py.import("parir")?)?;
    py.eval(&CString::new(s)?, Some(&globals), None)
}

fn lookup_builtin<'py>(expr: &Bound<'py, PyAny>, i: &Info) -> PyResult<Expr> {
    let py = expr.py();
    let ast = py.import("ast")?;
    let parir = py.import("parir")?;
    let s = ast.call_method1("unparse", types::PyTuple::new(py, vec![expr])?)?
        .extract::<String>()?;
    match eval_name(s, py) {
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
            } else if e.eq(parir.getattr("float16")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F16})
            } else if e.eq(parir.getattr("float32")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F32})
            } else if e.eq(parir.getattr("float64")?)? {
                Ok(Builtin::Convert {sz: ElemSize::F64})
            } else if e.eq(parir.getattr("int8")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I8})
            } else if e.eq(parir.getattr("int16")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I16})
            } else if e.eq(parir.getattr("int32")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I32})
            } else if e.eq(parir.getattr("int64")?)? {
                Ok(Builtin::Convert {sz: ElemSize::I64})
            } else {
                py_runtime_error!(i, "Unknown built-in operator {expr}")
            }?;
            Ok(Expr::Builtin {func, args: vec![], ty: Type::Unknown, i: i.clone()})
        },
        Err(_) => {
            py_runtime_error!(i, "Failed to identify built-in operator {expr}")
        },
    }
}

fn convert_expr<'py, 'a>(
    expr: Bound<'py, PyAny>, env: &'py ConvertEnv<'py, 'a>
) -> PyResult<Expr> {
    let i = extract_info(&expr, env);
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
        if val.is_instance(&types::PyBool::type_object(val.py()))? {
            let v = val.extract::<bool>()?;
            Ok(Expr::Bool {v, ty, i})
        } else if val.is_instance(&types::PyInt::type_object(val.py()))? {
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
        let fields = keys.into_iter()
            .zip(values.into_iter())
            .collect::<BTreeMap<String, Expr>>();
        Ok(Expr::Dict {fields, ty, i})
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
    let i = extract_info(&stmt, env);
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
                let fun_id = func.getattr("id")?.extract::<String>()?;
                let py = stmt.py();
                let builtins = py.import("builtins")?;
                match eval_name(fun_id, py) {
                    Ok(e) if e.eq(builtins.getattr("range")?)? => Ok(iter),
                    _ => py_runtime_error!(i, "For-loop must iterate using the range builtin")
                }
            } else {
                py_runtime_error!(i, "For-loop must iterate using the range builtin")
            }
        } else {
            py_runtime_error!(i, "For-loop must iterate using the range builtin")
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
            let expr = convert_expr(stmt.getattr("value")?, env)?;
            match (dst, expr) {
                (dst @ (Expr::Var {..} | Expr::Subscript {..}), expr) => {
                    Ok(Stmt::Assign {dst, expr, i})
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
) -> PyResult<FunDef> {
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
    Ok(FunDef {id, params: untyped_args, body: ir_body, i})
}

#[cfg(test)]
mod test {
    use super::*;

    use pyo3::types::*;

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
        let ast = ast_module.call_method("parse", py_args, Some(&py_kwargs))?;
        ast.getattr("body")
    }

    fn parse_str_stmts<'py>(
        py: Python<'py>,
        s: &str
    ) -> PyResult<Bound<'py, PyAny>> {
        parse_str(py, s, false)
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
        parse_str(py, s, true)
    }

    fn lookup_builtin_ok(
        s: &str,
        expected_func: Builtin
    ) -> PyResult<()> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let ast = parse_str_expr(py, s)?;
            let e = lookup_builtin(&ast, &Info::default())?;
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
            let ast = parse_str_expr(py, s)?;
            assert!(lookup_builtin(&ast, &Info::default()).is_err());
            Ok(())
        })
    }

    #[test]
    fn lookup_builtin_exp() -> PyResult<()> {
        lookup_builtin_ok("parir.exp", Builtin::Exp)?;
        lookup_builtin_ok("math.exp", Builtin::Exp)?;
        lookup_builtin_fail("exp")
    }

    #[test]
    fn lookup_builtin_inf() -> PyResult<()> {
        lookup_builtin_ok("parir.inf", Builtin::Inf)?;
        lookup_builtin_ok("math.inf", Builtin::Inf)?;
        lookup_builtin_ok("float('inf')", Builtin::Inf)?;
        lookup_builtin_fail("inf")
    }

    #[test]
    fn lookup_builtin_log() -> PyResult<()> {
        lookup_builtin_ok("parir.log", Builtin::Log)?;
        lookup_builtin_ok("math.log", Builtin::Log)?;
        lookup_builtin_fail("log")
    }

    #[test]
    fn lookup_builtin_max() -> PyResult<()> {
        lookup_builtin_ok("parir.max", Builtin::Max)?;
        lookup_builtin_ok("max", Builtin::Max)
    }

    #[test]
    fn lookup_builtin_min() -> PyResult<()> {
        lookup_builtin_ok("parir.min", Builtin::Min)?;
        lookup_builtin_ok("min", Builtin::Min)
    }

    #[test]
    fn lookup_builtin_conversion() -> PyResult<()> {
        lookup_builtin_ok("parir.int8", Builtin::Convert {sz: ElemSize::I8})?;
        lookup_builtin_ok("parir.int16", Builtin::Convert {sz: ElemSize::I16})?;
        lookup_builtin_ok("parir.int32", Builtin::Convert {sz: ElemSize::I32})?;
        lookup_builtin_ok("parir.int64", Builtin::Convert {sz: ElemSize::I64})?;
        lookup_builtin_ok("parir.float16", Builtin::Convert {sz: ElemSize::F16})?;
        lookup_builtin_ok("parir.float32", Builtin::Convert {sz: ElemSize::F32})?;
        lookup_builtin_ok("parir.float64", Builtin::Convert {sz: ElemSize::F64})
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

    fn convert_expr_wrap(s: &str) -> PyResult<Expr> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let expr = parse_str_expr(py, s)?;
            let env = ConvertEnv {
                ast : py.import("ast")?,
                filepath: &String::from("<test>"),
                fst_line: 0
            };
            convert_expr(expr, &env)
        })
    }

    fn mkinfo(line1: usize, col1: usize, line2: usize, col2: usize) -> Info {
        Info::new("<test>", FilePos::new(line1, col1), FilePos::new(line2, col2))
    }

    fn var(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn convert_expr_variable() {
        let expr = convert_expr_wrap("a").unwrap();
        assert_eq!(expr, Expr::Var {
            id: var("a"),
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
    fn convert_expr_binop_add() {
        let expr = convert_expr_wrap("1 + 1").unwrap();
        assert_eq!(expr, Expr::BinOp {
            lhs: Box::new(Expr::Int {
                v: 1,
                ty : Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            op: BinOp::Add,
            rhs: Box::new(Expr::Int {
                v: 1,
                ty: Type::Unknown,
                i: mkinfo(1, 4, 1, 5)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_expr_binop_pow_fail() {
        let result = convert_expr_wrap("2 ** 4");
        assert!(result.is_err());
    }

    #[test]
    fn convert_expr_equality() {
        let expr = convert_expr_wrap("a == b").unwrap();
        assert_eq!(expr, Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: var("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            op: BinOp::Eq,
            rhs: Box::new(Expr::Var {
                id: var("b"),
                ty: Type::Unknown,
                i: mkinfo(1, 5, 1, 6)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 6)
        });
    }

    #[test]
    fn convert_expr_string_lookup() {
        let expr = convert_expr_wrap("a['x']").unwrap();
        assert_eq!(expr, Expr::Subscript {
            target: Box::new(Expr::Var {
                id: var("a"),
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
                id: var("a"),
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
                id: var("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            }),
            idx: Box::new(Expr::Tuple {
                elems: vec![
                    Expr::Var {id: var("x"), ty: Type::Unknown, i: mkinfo(1, 2, 1, 3)},
                    Expr::Var {id: var("y"), ty: Type::Unknown, i: mkinfo(1, 5, 1, 6)}
                ],
                ty: Type::Unknown,
                i: mkinfo(1, 2, 1, 6)
            }),
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 7)
        });
    }

    #[test]
    fn convert_expr_dict() {
        let expr = convert_expr_wrap("{'a': 2, 'b': 3.14}").unwrap();
        let fields = vec![
            ("a", Expr::Int {v: 2, ty: Type::Unknown, i: mkinfo(1, 6, 1, 7)}),
            ("b", Expr::Float {v: 3.14, ty: Type::Unknown, i: mkinfo(1, 14, 1, 18)})
        ];
        let fields = fields.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<BTreeMap<String, Expr>>();
        assert_eq!(expr, Expr::Dict {
            fields,
            ty: Type::Unknown,
            i: mkinfo(1, 0, 1, 19)
        });
    }

    fn convert_stmt_wrap(s: &str) -> PyResult<Stmt> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stmt = parse_str_stmt(py, s)?;
            let env = ConvertEnv {
                ast : py.import("ast")?,
                filepath: &String::from("<test>"),
                fst_line: 0
            };
            convert_stmt(stmt, &env)
        })
    }

    #[test]
    fn convert_stmt_assignment() {
        let stmt = convert_stmt_wrap("a = 2").unwrap();
        assert_eq!(stmt, Stmt::Assign {
            dst: Expr::Var {
                id: var("a"),
                ty: Type::Unknown,
                i: mkinfo(1, 0, 1, 1)
            },
            expr: Expr::Int {
                v: 2,
                ty: Type::Unknown,
                i: mkinfo(1, 4, 1, 5)
            },
            i: mkinfo(1, 0, 1, 5)
        });
    }

    #[test]
    fn convert_stmt_for_loop_range() {
        let stmt = convert_stmt_wrap("for i in range(1, 10):\n  x[i] = i").unwrap();
        assert_eq!(stmt, Stmt::For {
            var: var("i"),
            lo: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(1, 15, 1, 16)},
            hi: Expr::Int {v: 10, ty: Type::Unknown, i: mkinfo(1, 18, 1, 20)},
            body: vec![
                Stmt::Assign {
                    dst: Expr::Subscript {
                        target: Box::new(Expr::Var {
                            id: var("x"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 2, 2, 3)
                        }),
                        idx: Box::new(Expr::Var {
                            id: var("i"),
                            ty: Type::Unknown,
                            i: mkinfo(2, 4, 2, 5)
                        }),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 6)
                    },
                    expr: Expr::Var {
                        id: var("i"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 9, 2, 10)
                    },
                    i: mkinfo(2, 2, 2, 10)
                }
            ],
            i: mkinfo(1, 0, 2, 10)
        })
    }

    #[test]
    fn convert_stmt_for_in_loop_fail() {
        let result = convert_stmt_wrap("for x in s:\n  x = x + 1");
        assert!(result.is_err());
    }

    #[test]
    fn convert_stmt_if_cond() {
        let stmt = convert_stmt_wrap("if x:\n  y = 1\nelse:\n  y = 2").unwrap();
        assert_eq!(stmt, Stmt::If {
            cond: Expr::Var {
                id: var("x"),
                ty: Type::Unknown,
                i: mkinfo(1, 3, 1, 4)
            },
            thn: vec![
                Stmt::Assign {
                    dst: Expr::Var {
                        id: var("y"),
                        ty: Type::Unknown,
                        i: mkinfo(2, 2, 2, 3)
                    },
                    expr: Expr::Int {v: 1, ty: Type::Unknown, i: mkinfo(2, 6, 2, 7)},
                    i: mkinfo(2, 2, 2, 7)
                }
            ],
            els: vec![
                Stmt::Assign {
                    dst: Expr::Var {
                        id: var("y"),
                        ty: Type::Unknown,
                        i: mkinfo(4, 2, 4, 3)
                    },
                    expr: Expr::Int {v: 2, ty: Type::Unknown, i: mkinfo(4, 6, 4, 7)},
                    i: mkinfo(4, 2, 4, 7)
                }
            ],
            i: mkinfo(1, 0, 4, 7)
        });
    }

    fn convert_stmts_wrap(s: &str) -> PyResult<Vec<Stmt>> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let stmt = parse_str_stmts(py, s)?;
            let env = ConvertEnv {
                ast : py.import("ast")?,
                filepath: &String::from("<test>"),
                fst_line: 0
            };
            convert_stmts(stmt, &env)
        })
    }

    #[test]
    fn convert_for_overloaded_range() {
        let res = convert_stmts_wrap("range = 3\n for x in range(1, 2):\n  x = x + 1");
        assert!(res.is_err());
    }
}
