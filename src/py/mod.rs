pub mod ast;
mod constant_fold;
mod from_py;
mod inline_calls;
mod inline_const;
mod labels;
mod par;
mod pprint;
mod slices;
mod symbolize;
mod type_check;

use crate::par::ParKind;
use symbolize::Symbolize;

use pyo3::prelude::*;

use std::collections::BTreeMap;

pub use inline_calls::inline_function_calls;

pub fn parse_untyped_ast<'py>(
    ast: Bound<'py, PyAny>,
    filepath: String,
    line_ofs: usize,
    col_ofs: usize
) -> PyResult<ast::FunDef> {
    let ast = from_py::to_untyped_ir(ast, filepath, line_ofs, col_ofs)?;
    let ast = ast.symbolize_default()?;
    labels::associate_labels(ast)
}

pub fn specialize_ast_on_arguments<'py>(
    ast: ast::FunDef,
    args: Vec<Bound<'py, PyAny>>,
    par: &BTreeMap<String, Vec<ParKind>>
) -> PyResult<ast::FunDef> {
    // Ensure the AST contains any degree of parallelism - otherwise, there is no point in using
    // this framework at all.
    par::ensure_parallelism(&ast, &par)?;

    // Perform the type-checking and inlining of literal values in an intertwined manner. First, we
    // type-check the parameters based on the corresponding arguments provided in the function
    // call. Second, once the parameters have been typed, we inline the values of scalar parameters
    // into the AST. Third, we type-check the body of the function.
    //
    // This particular order is important, because it allows the type-checker to determine the
    // exact size of all slices, and therefore reason about the correctness of the dimensions of
    // slice operations.
    let ast = type_check::type_check_params(ast, &args)?;
    let ast = inline_const::inline_scalar_values(ast, &args)?;
    let ast = type_check::type_check_body(ast)?;

    // Replace slice statements with for-loops
    let ast = slices::replace_slices_with_for_loops(ast)?;

    Ok(ast)
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
