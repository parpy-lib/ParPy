pub mod ast;
mod from_py;
mod inline_const;
mod labels;
mod pprint;
mod symbolize;
mod type_check;

use symbolize::Symbolize;

use pyo3::prelude::*;

pub use type_check::type_check;
pub use inline_const::inline_scalar_values;

pub fn parse_untyped_ast<'py>(
    ast: Bound<'py, PyAny>,
    filepath: String,
    fst_line: usize
) -> PyResult<ast::FunDef> {
    let ast = from_py::to_untyped_ir(ast, filepath, fst_line)?;
    let ast = ast.symbolize_default()?;
    labels::associate_labels(ast)
}

#[macro_export]
macro_rules! py_runtime_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::compile_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_name_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::name_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_type_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::type_err($i.error_msg(format!($($t)*)))))
    }
}
