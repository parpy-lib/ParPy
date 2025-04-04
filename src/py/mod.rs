pub mod ast;
mod constant_fold;
mod from_py;
mod indices;
mod inline_calls;
mod inline_const;
mod labels;
mod par;
mod pprint;
mod slices;
mod symbolize;
mod type_check;

use symbolize::Symbolize;
use crate::par::LoopPar;
use crate::utils::debug;

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
    par: &BTreeMap<String, LoopPar>,
    debug_env: &debug::DebugEnv
) -> PyResult<ast::FunDef> {
    // Ensure the AST contains any degree of parallelism - otherwise, there is no point in using
    // this framework at all.
    par::ensure_parallelism(&ast, &par)?;

    // Perform the type-checking and inlining of literal values in an intertwined manner. First, we
    // type-check the parameters based on the corresponding arguments provided in the function
    // call. Second, once the parameters have been typed, we inline the values of scalar parameters
    // into the AST.
    //
    // This particular order is important, because it allows us to reason about the exact sizes of
    // all slices and by extension the correctness of dimensions of slice operations.
    let ast = type_check::type_check_params(ast, &args)?;
    let ast = inline_const::inline_scalar_values(ast, &args)?;
    debug_env.print("Python-like AST after inlining", &ast);

    // As the types of slice operations involve tensors, we check that the shapes of all operations
    // are valid, ignoring the element types inside tensors. Next, we resolve indices based on the
    // inferred shapes, replacing negative and omitted indices with valid non-negative indices.
    // After this, we perform the slice transformation, replacing slice statements with scalar
    // operations. Finally, we can type-check the scalar operations of the resulting AST.
    //
    // Slice operations must be explicit, but they are allowed to perform broadcasting operations,
    // unlike regular operations. When working with slices, the dimensions are explicitly declared,
    // making it natural to specify parallelism, while regular broadcasting would make this
    // implicit and therefore ill-suited with the parallelism approach used in this library.
    let ast = type_check::check_body_shape(ast)?;
    debug_env.print("Python-like AST after shape checking", &ast);
    let ast = indices::resolve_indices(ast)?;
    debug_env.print("Python-like AST after resolving indices", &ast);
    let ast = slices::replace_slices_with_for_loops(ast)?;
    debug_env.print("Python-like AST after slice transformation", &ast);
    let ast = type_check::type_check_body(ast)?;
    debug_env.print("Python-like AST after type-checking", &ast);

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
