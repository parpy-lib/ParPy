use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::smap::SFold;

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use std::collections::BTreeMap;

fn collect_called_functions_expr<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    e: &Expr
) -> PyResult<Vec<String>> {
    match e {
        Expr::Call {id, args, i, ..} => {
            let mut acc = args.sfold_result(Ok(acc), |acc, e| {
                collect_called_functions_expr(tops, acc, e)
            })?;
            if let Some(ast_ref) = tops.get(id.get_str()) {
                // If this function has not been included yet, we include it. If it is a
                // user-defined function in ParPy, we recursively consider the body of this
                // function.
                if !acc.contains(id.get_str()) {
                    acc.push(id.get_str().clone());
                    let t = unsafe { ast_ref.reference::<Top>() };
                    if let Top::FunDef {v} = t {
                        acc.append(&mut collect_called_functions(tops, v)?);
                    };
                }
                Ok(acc)
            } else {
                py_runtime_error!(i, "Call to unknown function {id}")
            }
        }
        _ => e.sfold_result(Ok(acc), |acc, e| collect_called_functions_expr(tops, acc, e))
    }
}

fn collect_called_functions_stmt<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    s: &Stmt
) -> PyResult<Vec<String>> {
    let acc = s.sfold_result(Ok(acc), |acc, s| collect_called_functions_stmt(tops, acc, s))?;
    s.sfold_result(Ok(acc), |acc, e| collect_called_functions_expr(tops, acc, e))
}

fn collect_called_functions_stmts<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    stmts: &Vec<Stmt>
) -> PyResult<Vec<String>> {
    stmts.sfold_result(Ok(acc), |acc, s| collect_called_functions_stmt(tops, acc, s))
}

fn collect_called_functions<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    def: &FunDef
) -> PyResult<Vec<String>> {
    collect_called_functions_stmts(tops, vec![], &def.body)
}

fn make_ast<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    called_funs: Vec<String>,
    main: FunDef
) -> Ast {
    let tops = called_funs.into_iter()
        .unique()
        .rev()
        .map(|id| {
            let ast_ref = tops.get(&id).unwrap();
            unsafe { ast_ref.reference::<Top>() }.clone()
        })
        .collect::<Vec<Top>>();
    Ast {tops, main}
}

pub fn apply<'py>(
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    def: FunDef
) -> PyResult<Ast> {
    let called_funs = collect_called_functions(&tops, &def)?;
    Ok(make_ast(&tops, called_funs, def))
}
