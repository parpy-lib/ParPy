use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::smap::SFold;

use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use std::collections::BTreeMap;

fn collect_called_functions_expr<'py>(
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    e: &Expr
) -> PyResult<Vec<String>> {
    match e {
        Expr::Call {id, args, i, ..} => {
            let mut acc = args.sfold_result(Ok(acc), |acc, e| {
                collect_called_functions_expr(ir_asts, acc, e)
            })?;
            if let Some(ast_ref) = ir_asts.get(id) {
                // If we have not yet considered this function, we add it to the set and
                // recursively collect functions called from this function.
                if !acc.contains(id) {
                    acc.push(id.clone());
                    let def = unsafe { ast_ref.reference::<FunDef>() };
                    acc.append(&mut collect_called_functions(ir_asts, def)?);
                }
                Ok(acc)
            } else {
                py_runtime_error!(i, "Call to unknown function {id}")
            }
        }
        _ => e.sfold_result(Ok(acc), |acc, e| collect_called_functions_expr(ir_asts, acc, e))
    }
}

fn collect_called_functions_stmt<'py>(
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    s: &Stmt
) -> PyResult<Vec<String>> {
    let acc = s.sfold_result(Ok(acc), |acc, s| collect_called_functions_stmt(ir_asts, acc, s))?;
    s.sfold_result(Ok(acc), |acc, e| collect_called_functions_expr(ir_asts, acc, e))
}

fn collect_called_functions_stmts<'py>(
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>,
    acc: Vec<String>,
    stmts: &Vec<Stmt>
) -> PyResult<Vec<String>> {
    stmts.sfold_result(Ok(acc), |acc, s| collect_called_functions_stmt(ir_asts, acc, s))
}

fn collect_called_functions<'py>(
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>,
    def: &FunDef
) -> PyResult<Vec<String>> {
    collect_called_functions_stmts(ir_asts, vec![], &def.body)
}

fn make_ast<'py>(
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>,
    called_funs: Vec<String>,
    def: FunDef
) -> Ast {
    let defs = called_funs.into_iter()
        .unique()
        .rev()
        .map(|id| {
            let ast_ref = ir_asts.get(&id).unwrap();
            unsafe { ast_ref.reference::<FunDef>() }.clone()
        })
        .chain(vec![def].into_iter())
        .collect::<Vec<FunDef>>();
    Ast {exts: vec![], defs}
}

pub fn apply<'py>(
    ir_asts: BTreeMap<String, Bound<'py, PyCapsule>>,
    def: FunDef
) -> PyResult<Ast> {
    let called_funs = collect_called_functions(&ir_asts, &def)?;
    Ok(make_ast(&ir_asts, called_funs, def))
}
