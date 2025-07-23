use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::smap::SFold;

use pyo3::prelude::*;

fn check_no_return_in_main_function_stmt(id: &Name, _: (), s: &Stmt) -> PyResult<()> {
    match s {
        Stmt::Return {i, ..} => {
            let id = id.pprint_default();
            py_runtime_error!(i, "The called function {id} cannot return a value")
        },
        _ => {
            s.sfold_result(Ok(()), |acc, s| {
                check_no_return_in_main_function_stmt(&id, acc, s)
            })
        }
    }
}

fn check_no_return_in_main_function_body(id: &Name, stmts: &Vec<Stmt>) -> PyResult<()> {
    stmts.sfold_result(Ok(()), |acc, s| check_no_return_in_main_function_stmt(&id, acc, s))
}

pub fn check_no_return_in_main_function(ast: &Ast) -> PyResult<()> {
    let main_def = ast.last().unwrap();
    check_no_return_in_main_function_body(&main_def.id, &main_def.body)
}
