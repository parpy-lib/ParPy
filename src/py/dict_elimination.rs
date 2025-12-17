use super::ast::*;
use crate::py_internal_error;
use crate::py_runtime_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use pyo3::prelude::*;
use std::collections::BTreeMap;

type DictEnv = BTreeMap<Name, BTreeMap<String, (Name, Type)>>;

fn extract_identifier(target: Expr) -> PyResult<Name> {
    let i = target.get_info();
    match target {
        Expr::Var {id, ..} => Ok(id),
        _ => py_internal_error!(i, "Invalid form of dictionary expression")
    }
}

fn extract_string(idx: Expr) -> PyResult<String> {
    let i = idx.get_info();
    match idx {
        Expr::String {v, ..} => Ok(v),
        _ => py_internal_error!(i, "Invalid form of dictionary index")
    }
}

fn flatten_dicts_argument(
    env: &DictEnv,
    mut acc: Vec<Expr>,
    arg: Expr
) -> PyResult<Vec<Expr>> {
    match arg {
        Expr::Var {id, ty: Type::Dict {..}, i} => {
            match env.get(&id) {
                Some(fields) => {
                    let mut expanded_args = fields.into_iter()
                        .map(|(_, (id, ty))| Expr::Var {
                            id: id.clone(),
                            ty: ty.clone(),
                            i: i.clone()
                        })
                        .collect::<Vec<Expr>>();
                    acc.append(&mut expanded_args);
                    Ok(acc)
                },
                None => py_internal_error!(i, "Failed to identify dictionary when flattening")
            }
        },
        _ => {
            acc.push(flatten_dicts_expr(&env, arg)?);
            Ok(acc)
        }
    }
}

fn flatten_dicts_expr(
    env: &DictEnv,
    e: Expr
) -> PyResult<Expr> {
    match e {
        Expr::Subscript {target, idx, ty, i} if *idx.get_type() == Type::String => {
            let id = extract_identifier(*target)?;
            let key = extract_string(*idx)?;
            let value_id = match env.get(&id) {
                Some(fields) => {
                    match fields.get(&key) {
                        Some((id, _)) => Ok(id.clone()),
                        None => py_runtime_error!(i, "Could not find key {key} in dictionary")
                    }
                },
                None => py_internal_error!(i, "Failed to identify dictionary when flattening")
            }?;
            Ok(Expr::Var {id: value_id, ty, i})
        },
        Expr::Call {id, args, ty, i} => {
            let args = args.sflatten_result(vec![], |acc, arg| {
                flatten_dicts_argument(&env, acc, arg)
            })?;
            Ok(Expr::Call {id, args, ty, i})
        },
        _ => e.smap_result(|e| flatten_dicts_expr(&env, e))
    }
}

fn flatten_dicts_stmt(
    env: &DictEnv,
    s: Stmt
) -> PyResult<Stmt> {
    let s = s.smap_result(|s| flatten_dicts_stmt(&env, s))?;
    s.smap_result(|e| flatten_dicts_expr(&env, e))
}

fn flatten_dict_param(
    acc: (DictEnv, Vec<Param>),
    p: Param
) -> (DictEnv, Vec<Param>) {
    let (mut env, mut params) = acc;
    match p.ty {
        Type::Dict {fields} => {
            let ids = fields.into_iter()
                .map(|(k, ty)| (k.clone(), (Name::sym_str(&format!("{0}_{k}", p.id)), ty)))
                .collect::<BTreeMap<String, (Name, Type)>>();
            let mut unrolled_params = ids.iter()
                .map(|(_, (id, ty))| Param {id: id.clone(), ty: ty.clone(), i: p.i.clone()})
                .collect::<Vec<Param>>();
            env.insert(p.id, ids);
            params.append(&mut unrolled_params);
            (env, params)
        },
        _ => {
            params.push(p);
            (env, params)
        }
    }
}

fn flatten_dicts_fun_def(def: FunDef) -> PyResult<FunDef> {
    let (env, params) = def.params.sfold_owned((BTreeMap::new(), vec![]), flatten_dict_param);
    let body = def.body.smap_result(|s| flatten_dicts_stmt(&env, s))?;
    Ok(FunDef {params, body, ..def})
}

fn apply_top(t: Top) -> PyResult<Top> {
    match t {
        Top::FunDef {v} => Ok(Top::FunDef {v: flatten_dicts_fun_def(v)?}),
        Top::CallbackDecl {..} | Top::ExtDecl {..} => Ok(t)
    }
}

pub fn apply(ast: Ast) -> PyResult<Ast> {
    let tops = ast.tops.smap_result(apply_top)?;
    let main = flatten_dicts_fun_def(ast.main)?;
    Ok(Ast {tops, main})
}
