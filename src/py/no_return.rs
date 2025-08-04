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
    check_no_return_in_main_function_body(&ast.main.id, &ast.main.body)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;
    use crate::utils::info::*;

    #[test]
    fn detects_trailing_return_stmt() {
        let main = FunDef {
            id: id("f"),
            params: vec![],
            body: vec![return_stmt(int(1, None))],
            res_ty: Type::Void,
            i: Info::default()
        };
        let ast = Ast {tops: vec![], main};
        assert!(check_no_return_in_main_function(&ast).is_err());
    }

    #[test]
    fn detects_nested_return_stmt() {
        let main = FunDef {
            id: id("f"),
            params: vec![],
            body: vec![Stmt::While {
                cond: bool_expr(true, None),
                body: vec![return_stmt(int(0, None))],
                i: Info::default()
            }],
            res_ty: Type::Void,
            i: Info::default()
        };
        let ast = Ast {tops: vec![], main};
        assert!(check_no_return_in_main_function(&ast).is_err());
    }

    #[test]
    fn no_return_stmt() {
        let main = FunDef {
            id: id("f"),
            params: vec![],
            body: vec![],
            res_ty: Type::Void,
            i: Info::default()
        };
        let ast = Ast {tops: vec![], main};
        assert!(check_no_return_in_main_function(&ast).is_ok());
    }
}
