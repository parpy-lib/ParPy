use super::ast::*;
use super::constant_fold;
use super::type_check;
use crate::py_runtime_error;
use crate::option::CompileOptions;
use crate::utils::err::CompileError;
use crate::utils::info::Info;
use crate::utils::smap::*;

use pyo3::prelude::*;

#[derive(Debug)]
struct SpecEnv<'a> {
    unify_env: &'a type_check::UnifyEnv,
    opts: &'a CompileOptions
}

impl<'a> SpecEnv<'a> {
    fn new(
        unify_env: &'a type_check::UnifyEnv,
        opts: &'a CompileOptions
    ) -> SpecEnv<'a> {
        SpecEnv { unify_env, opts }
    }
}

fn specialize_tensor_elem_size<'a>(
    env: &SpecEnv<'a>,
    sz: TensorElemSize,
    i: &Info
) -> PyResult<TensorElemSize> {
    match sz {
        TensorElemSize::Fixed {..} => Ok(sz),
        TensorElemSize::Variable {id} => {
            match env.unify_env.lookup_type_variable(&id) {
                Some(sz) => Ok(TensorElemSize::Fixed {sz: sz.clone()}),
                None => py_runtime_error!(i, "Found unresolved type variable")
            }
        }
    }
}

fn specialize_type<'a>(
    env: &SpecEnv<'a>,
    ty: Type,
    i: &Info
) -> PyResult<Type> {
    match ty {
        Type::Tensor {sz, shape} => {
            let sz = specialize_tensor_elem_size(env, sz, &i)?;
            Ok(Type::Tensor {sz, shape})
        },
        _ => ty.smap_result(|ty| specialize_type(env, ty, &i))
    }
}

fn specialize_expr<'a>(
    env: &SpecEnv<'a>,
    e: Expr,
) -> PyResult<Expr> {
    match e {
        Expr::Var {id, ty, i} => {
            match env.unify_env.shape_vars.get(&id) {
                Some(v) => Ok(Expr::Int {v: *v as i128, ty, i}),
                None => Ok(Expr::Var {id, ty, i})
            }
        },
        Expr::Convert {e, ty, i} => {
            let ty = specialize_type(env, ty, &i)?;
            Ok(Expr::Convert {e, ty, i})
        },
        Expr::StaticBackendEq {backend, ty, i} => {
            let v = backend == env.opts.backend;
            Ok(Expr::Bool {v, ty, i})
        },
        Expr::StaticTypesEq {lhs, rhs, ty, i} => {
            let v = type_check::eq_tensor_elem_size(env.unify_env.clone(), &lhs, &rhs);
            Ok(Expr::Bool {v: v.is_some(), ty, i})
        },
        Expr::AllocShared {shape, sz, ty, i} => {
            let sz = specialize_tensor_elem_size(env, sz, &i)?;
            let ty = specialize_type(env, ty, &i)?;
            Ok(Expr::AllocShared {shape, sz, ty, i})
        },
        _ => e.smap_result(|e| specialize_expr(env, e))
    }
}

fn specialize_stmt<'a>(
    env: &SpecEnv<'a>,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> PyResult<Vec<Stmt>> {
    match s {
        Stmt::If {cond, thn, els, i} => {
            let cond = specialize_expr(env, cond)?;
            let cond = constant_fold::fold_expr(cond);
            if let Expr::Bool {v, ..} = cond {
                if v {
                    let mut thn = specialize_stmts(env, thn)?;
                    acc.append(&mut thn);
                } else {
                    let mut els = specialize_stmts(env, els)?;
                    acc.append(&mut els);
                };
                Ok(acc)
            } else {
                let thn = specialize_stmts(env, thn)?;
                let els = specialize_stmts(env, els)?;
                acc.push(Stmt::If {cond, thn, els, i});
                Ok(acc)
            }
        },
        // If the compiler reaches a static fail statement, it immediately produces a runtime error
        // based on the contents of the node.
        Stmt::Expr {e: Expr::StaticFail {msg, ..}, i} => py_runtime_error!(i, "{msg}"),
        _ => {
            let s = s.smap_result(|e| specialize_expr(env, e))?;
            s.sflatten_result(acc, |acc, s| specialize_stmt(env, acc, s))
        }
    }
}

fn specialize_stmts<'a>(
    env: &SpecEnv<'a>,
    stmts: Vec<Stmt>
) -> PyResult<Vec<Stmt>> {
    stmts.sflatten_result(vec![], |acc, s| specialize_stmt(env, acc, s))
}

pub fn apply<'py>(
    unify_env: &type_check::UnifyEnv,
    opts: &CompileOptions,
    body: Vec<Stmt>
) -> PyResult<Vec<Stmt>> {
    let env = SpecEnv::new(unify_env, opts);
    specialize_stmts(&env, body)
}
