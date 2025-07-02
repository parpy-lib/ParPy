use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::CompileError;
use crate::utils::info::Info;
use crate::utils::smap::SFold;

use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use std::collections::BTreeMap;

fn collect_called_function_ids_expr<'py>(
    mut acc: BTreeMap<String, Info>,
    e: &Expr,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<BTreeMap<String, Info>> {
    match e {
        Expr::Call {id, i, ..} if ir_asts.contains_key(id.get_str()) => {
            acc.insert(id.get_str().clone(), i.clone());
            Ok(acc)
        },
        _ => e.sfold_result(Ok(acc), |acc, e| collect_called_function_ids_expr(acc, e, ir_asts))
    }
}

fn collect_called_function_ids_stmt<'py>(
    acc: BTreeMap<String, Info>,
    s: &Stmt,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<BTreeMap<String, Info>> {
    let acc = s.sfold_result(Ok(acc), |acc, s| collect_called_function_ids_stmt(acc, s, ir_asts))?;
    s.sfold_result(Ok(acc), |acc, e| collect_called_function_ids_expr(acc, e, ir_asts))
}

fn collect_called_function_ids<'py>(
    fun: &FunDef,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<BTreeMap<String, Info>> {
    let acc = Ok(BTreeMap::new());
    fun.body.sfold_result(acc, |acc, s| collect_called_function_ids_stmt(acc, s, ir_asts))
}

fn lookup_fun_def<'py>(
    id: String,
    i: Info,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<FunDef> {
    if let Some(ast_ref) = ir_asts.get(&id) {
        Ok(unsafe { ast_ref.reference::<FunDef>() }.clone())
    } else {
        py_runtime_error!(i, "Reference to unknown function definition {id}")
    }
}

pub fn include_called_functions<'py>(
    fun: FunDef,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Ast> {
    let ids = collect_called_function_ids(&fun, &ir_asts)?;
    let mut funs = ids.into_iter()
        .map(|(k, v)| lookup_fun_def(k, v, ir_asts))
        .collect::<PyResult<Ast>>()?;
    funs.push(fun);
    Ok(funs)
}
