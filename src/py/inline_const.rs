/// All scalar parameters are treated as constants. In this file, we specialize the generated code
/// with respect to the actual value of these parameters by replacing references to the parameter
/// with its value.

use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::smap::SMapAccum;

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
        Expr::String {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::UnOp {..} | Expr::BinOp {..} | Expr::IfExpr {..} |
        Expr::Slice {..} | Expr::Tuple {..} | Expr::Call {..} |
        Expr::NeutralElement {..} | Expr::Builtin {..} | Expr::Convert {..} =>
            e.smap(|e| replace_constants_expr(consts, e))
    }
}

fn replace_constants_stmt(
    consts: &BTreeMap<Expr, Expr>,
    s: Stmt
) -> Stmt {
    s.smap(|s| replace_constants_stmt(consts, s))
        .smap(|e| replace_constants_expr(consts, e))
}

fn replace_constants_stmts(
    consts: &BTreeMap<Expr, Expr>,
    stmts: Vec<Stmt>
) -> Vec<Stmt> {
    stmts.smap(|s| replace_constants_stmt(consts, s))
}

fn replace_constants_def(
    consts: &BTreeMap<Expr, Expr>,
    def: FunDef
) -> FunDef {
    let body = replace_constants_stmts(consts, def.body);
    FunDef {body, ..def}
}

fn extract_scalar_value<'py>(
    arg: &Bound<'py, PyAny>,
    i: &Info,
    sz: ElemSize
) -> PyResult<Expr> {
    if sz == ElemSize::Bool {
        let v = arg.extract::<bool>()?;
        Ok(Expr::Bool {v, ty: Type::Tensor {sz, shape: vec![]}, i: i.clone()})
    } else if sz.is_signed_integer() {
        let v = arg.extract::<i128>()?;
        Ok(Expr::Int {v, ty: Type::Tensor {sz, shape: vec![]}, i: i.clone()})
    } else if sz.is_floating_point() {
        let v = arg.extract::<f64>()?;
        Ok(Expr::Float {v, ty: Type::Tensor {sz, shape: vec![]}, i: i.clone()})
    } else {
        py_runtime_error!(i, "Invalid scalar of element type {sz}")
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
            match extract_scalar_value(arg, &i, sz.clone()) {
                Ok(value) => {
                    acc.insert(target, value);
                    Ok(acc)
                },
                Err(e) => {
                    py_runtime_error!(i, "Extracting scalar value failed: {e}")
                }
            }
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
    def: FunDef,
    args: &Vec<Bound<'py, PyAny>>
) -> PyResult<FunDef> {
    let const_map = args.iter()
        .zip(def.params.iter())
        .fold(Ok(BTreeMap::new()), |acc, (arg, Param {id, ty, i})| {
            let target = Expr::Var {
                id: id.clone(), ty: ty.clone(), i: i.clone()
            };
            add_scalar_constant(acc?, target, arg)
        })?;
    Ok(replace_constants_def(&const_map, def))
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
        Expr::Int {v: v as i128, ty: scalar_ty(ElemSize::I64), i: Info::default()}
    }

    fn float(v: f64) -> Expr {
        Expr::Float {v, ty: scalar_ty(ElemSize::F64), i: Info::default()}
    }

    fn bool(v: bool) -> Expr {
        Expr::Bool {v, ty: scalar_ty(ElemSize::Bool), i: Info::default()}
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
        assert_eq!(env.get(&b_expr.unwrap()), Some(&bool(false)));
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
