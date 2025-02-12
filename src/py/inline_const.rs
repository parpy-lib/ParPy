/// All scalar parameters are treated as constants. In this file, we specialize the generated code
/// with respect to the actual value of these parameters by replacing references to the parameter
/// with its value.

use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;

use std::collections::BTreeMap;

use pyo3::prelude::*;

fn replace_constants_expr(
    consts: &BTreeMap<Expr, Expr>,
    e: Expr
) -> Expr {
    match e {
        Expr::Var {ref i, ..} => {
            match consts.get(&e) {
                Some(e) => e.clone().with_info(i.clone()),
                None => e.clone(),
            }
        },
        Expr::Subscript {ref target, ref idx, ref ty, ref i} => {
            if let Some(v) = consts.get(&e) {
                v.clone()
            } else {
                let target = Box::new(replace_constants_expr(consts, *target.clone()));
                let idx = Box::new(replace_constants_expr(consts, *idx.clone()));
                Expr::Subscript {target, idx, ty: ty.clone(), i: i.clone()}
            }
        },
        Expr::String {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} => e,
        Expr::UnOp {op, arg, ty, i} => {
            let arg = Box::new(replace_constants_expr(consts, *arg));
            Expr::UnOp {op, arg, ty, i}
        },
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = Box::new(replace_constants_expr(consts, *lhs));
            let rhs = Box::new(replace_constants_expr(consts, *rhs));
            Expr::BinOp {lhs, op, rhs, ty, i}
        },
        Expr::IfExpr {cond, thn, els, ty, i} => {
            let cond = Box::new(replace_constants_expr(consts, *cond));
            let thn = Box::new(replace_constants_expr(consts, *thn));
            let els = Box::new(replace_constants_expr(consts, *els));
            Expr::IfExpr {cond, thn, els, ty, i}
        },
        Expr::Tuple {elems, ty, i} => {
            let elems = replace_constants_exprs(consts, elems);
            Expr::Tuple {elems, ty, i}
        },
        Expr::Dict {fields, ty, i} => {
            let fields = fields.into_iter()
                .map(|(id, e)| (id, replace_constants_expr(consts, e)))
                .collect::<BTreeMap<String, Expr>>();
            Expr::Dict {fields, ty, i}
        },
        Expr::Builtin {func, args, ty, i} => {
            let args = replace_constants_exprs(consts, args);
            Expr::Builtin {func, args, ty, i}
        },
        Expr::Convert {e, ty} => {
            let e = Box::new(replace_constants_expr(consts, *e));
            Expr::Convert {e, ty}
        }
    }
}

fn replace_constants_exprs(
    consts: &BTreeMap<Expr, Expr>,
    exprs: Vec<Expr>
) -> Vec<Expr> {
    exprs.into_iter()
        .map(|e| replace_constants_expr(consts, e))
        .collect::<Vec<Expr>>()
}

fn replace_constants_stmt(
    consts: &BTreeMap<Expr, Expr>,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::Definition {ty, id, expr, i} => {
            let expr = replace_constants_expr(consts, expr);
            Stmt::Definition {ty, id, expr, i}
        },
        Stmt::Assign {dst, expr, i} => {
            let dst = replace_constants_expr(consts, dst);
            let expr = replace_constants_expr(consts, expr);
            Stmt::Assign {dst, expr, i}
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = replace_constants_expr(consts, lo);
            let hi = replace_constants_expr(consts, hi);
            let body = replace_constants_stmts(consts, body);
            Stmt::For {var, lo, hi, step, body, i}
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = replace_constants_expr(consts, cond);
            let thn = replace_constants_stmts(consts, thn);
            let els = replace_constants_stmts(consts, els);
            Stmt::If {cond, thn, els, i}
        },
        Stmt::While {cond, body, i} => {
            let cond = replace_constants_expr(consts, cond);
            let body = replace_constants_stmts(consts, body);
            Stmt::While {cond, body, i}
        },
        Stmt::Label {label, assoc, i} => Stmt::Label {label, assoc, i}
    }
}

fn replace_constants_stmts(
    consts: &BTreeMap<Expr, Expr>,
    stmts: Vec<Stmt>
) -> Vec<Stmt> {
    stmts.into_iter()
        .map(|s| replace_constants_stmt(consts, s))
        .collect::<Vec<Stmt>>()
}

fn extract_scalar_value<'py>(
    arg: &Bound<'py, PyAny>,
    i: &Info,
    sz: ElemSize
) -> PyResult<Expr> {
    if sz.is_signed_integer() {
        let v = arg.extract::<i64>()?;
        Ok(Expr::Int {v, ty: Type::Tensor {sz, shape: vec![]}, i: i.clone()})
    } else if sz.is_floating_point() {
        let v = arg.extract::<f64>()?;
        Ok(Expr::Float {v, ty: Type::Tensor {sz, shape: vec![]}, i: i.clone()})
    } else {
        py_runtime_error!(i, "Failed to extract literal value of type {sz}")
    }
}

fn add_scalar_constant<'py>(
    mut acc: BTreeMap<Expr, Expr>,
    target: Expr,
    arg: &Bound<'py, PyAny>
) -> PyResult<BTreeMap<Expr, Expr>> {
    let ty = target.get_type();
    let i = target.get_info();
    match ty {
        Type::Tensor {sz, shape} if shape.is_empty() => {
            if let Ok(value) = extract_scalar_value(arg, &i, sz.clone()) {
                acc.insert(target, value);
            };
            Ok(acc)
        },
        Type::Dict {fields} => {
            fields.iter()
                .fold(Ok(acc), |acc, (k, ty)| {
                    let i = target.get_info();
                    let target = Expr::Subscript {
                        target: Box::new(target.clone()),
                        idx: Box::new(Expr::String {
                            v: k.clone(), ty: Type::String, i: i.clone()
                        }),
                        ty: ty.clone(), i: i.clone()
                    };
                    let field = arg.get_item(k)?;
                    add_scalar_constant(acc?, target, &field)
                })
        },
        _ => Ok(acc)
    }
}

pub fn inline_scalar_values<'py>(
    ast: FunDef,
    args: &Vec<Bound<'py, PyAny>>
) -> PyResult<FunDef> {
    let const_map = args.iter()
        .zip(ast.params.iter())
        .fold(Ok(BTreeMap::new()), |acc, (arg, Param {id, ty, i})| {
            let target = Expr::Var {
                id: id.clone(), ty: ty.clone(), i: i.clone()
            };
            add_scalar_constant(acc?, target, arg)
        })?;
    let body = replace_constants_stmts(&const_map, ast.body);
    Ok(FunDef {body, ..ast})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::name::Name;
    use std::ffi::CString;
    use pyo3::types;

    fn eval_str<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyAny>> {
        let globals = types::PyDict::new(py);
        globals.set_item("math", py.import("math")?)?;
        globals.set_item("parir", py.import("parir")?)?;
        py.eval(&CString::new(s)?, Some(&globals), None)
    }

    fn extract_literals(s: &str, target: Expr) -> BTreeMap<Expr, Expr> {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let arg = eval_str(py, s).unwrap();
            add_scalar_constant(BTreeMap::new(), target.clone(), &arg).unwrap()
        })
    }

    fn var(s: &str) -> Name {
        Name::sym_str(s)
    }

    fn scalar_ty(sz: ElemSize) -> Type {
        Type::Tensor {sz, shape: vec![]}
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: scalar_ty(ElemSize::I64), i: Info::default()}
    }

    fn float(v: f64) -> Expr {
        Expr::Float {v, ty: scalar_ty(ElemSize::F64), i: Info::default()}
    }

    fn string(s: &str) -> Expr {
        Expr::String {v: s.to_string(), ty: Type::String, i: Info::default()}
    }

    #[test]
    fn extract_integer_literal() {
        let target = Expr::Var {
            id: var("x"), ty: scalar_ty(ElemSize::I64), i: Info::default()
        };
        let env = extract_literals("3", target.clone());
        assert_eq!(env.get(&target), Some(&int(3)));
    }

    #[test]
    fn extract_float_literal() {
        let target = Expr::Var {
            id: var("y"), ty: scalar_ty(ElemSize::F64), i: Info::default()
        };
        let env = extract_literals("2.0", target.clone());
        assert_eq!(env.get(&target), Some(&float(2.0)));
    }

    fn dict_lookup(target: &Expr, key: &str) -> Option<Expr> {
        if let Type::Dict {fields} = target.get_type() {
            let ty = fields.get(key)?;
            Some(Expr::Subscript {
                target: Box::new(target.clone()),
                idx: Box::new(string(key)),
                ty: ty.clone(),
                i: Info::default()
            })
        } else {
            None
        }
    }

    #[test]
    fn extract_dict_entry_literals() {
        let keys = ["a".to_string(), "b".to_string(), "c".to_string()];
        let types = [scalar_ty(ElemSize::I64), scalar_ty(ElemSize::Bool), scalar_ty(ElemSize::F64)];
        let fields = keys.clone()
            .into_iter()
            .zip(types.into_iter())
            .collect::<BTreeMap<String, Type>>();
        let target = Expr::Var {
            id: var("x"), ty: Type::Dict {fields}, i: Info::default()
        };
        let env = extract_literals("{'a': 3, 'b': False, 'c': 2.0}", target.clone());
        let a_expr = dict_lookup(&target, "a");
        let b_expr = dict_lookup(&target, "b");
        let c_expr = dict_lookup(&target, "c");
        assert_eq!(env.get(&a_expr.unwrap()), Some(&int(3)));
        assert_eq!(env.get(&b_expr.unwrap()), None);
        assert_eq!(env.get(&c_expr.unwrap()), Some(&float(2.0)));
    }

    #[test]
    fn replace_dict_arg_literal() {
        let a_arg = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(string("a")),
            ty: Type::Unknown,
            i: Info::default()
        };
        let env = vec![
            (a_arg.clone(), int(4))
        ].into_iter()
            .collect::<BTreeMap<Expr, Expr>>();

        assert_eq!(replace_constants_expr(&env, a_arg.clone()), int(4));
    }
}
