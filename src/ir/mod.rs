mod ast;
mod from_py;
mod local_vars;
mod par;
mod types;

use crate::err::CompileResult;
use crate::py::ast as py_ast;

pub fn python_to_ir(ast: py_ast::Ast) -> CompileResult<ast::Ast> {
    let ast = from_py::from_ast(ast)?;
    let ast = local_vars::insert_local_variable_declarations(ast);
    let ast = types::infer_types(ast)?;
    Ok(ast)
}
