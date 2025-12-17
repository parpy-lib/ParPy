mod adjust_negative_indices;
pub mod ast;
mod constant_fold;
mod eliminate_duplicate_functions;
mod for_loops;
mod free_vars;
mod from_py;
mod inline_calls;
mod inline_const;
mod labels;
mod par;
mod pprint;
mod replace_callbacks;
mod shape_symbol_labels;
mod slice_transformation;
mod specialize;
mod symbolize;
mod type_check;

#[cfg(test)]
pub mod ast_builder;

use crate::py_runtime_error;
use crate::option::*;
use crate::utils::ast::ScalarSizes;
use crate::utils::debug;
use crate::utils::err::CompileError;

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use std::collections::BTreeMap;

pub use from_py::convert_callback;
pub use from_py::convert_external;
pub use inline_calls::inline_function_calls;

pub fn parse_untyped_ast<'py>(
    ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>)
) -> PyResult<ast::FunDef> {
    let ast = from_py::to_untyped_ir(ast, info, tops, vars.clone())?;
    let ast = symbolize::with_tops(tops, &vars, ast)?;
    labels::associate_labels(ast)
}

pub fn specialize_ast_on_arguments<'py>(
    t: ast::Top,
    args: Vec<Bound<'py, PyAny>>,
    opts: &CompileOptions,
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    debug_env: &debug::DebugEnv
) -> PyResult<ast::Ast> {
    // We expect the top to contain a function definition, but it could also be an external
    // declaration, in which case we report an error.
    let main = match t {
        ast::Top::FunDef {v} => Ok(v),
        ast::Top::CallbackDecl {id, i, ..} => {
            py_runtime_error!(
                i,
                "Expected {id} to be a function definition, but found a \
                 callback declaration."
            )
        },
        ast::Top::ExtDecl {id, i, ..} => {
            py_runtime_error!(
                i,
                "Expected {id} to be a function definition, but found an \
                 external function declaration."
            )
        }
    }?;

    // Adds labels to statements corresponding to the shape symbols they use, if this is enabled in
    // the compiler.
    let main = shape_symbol_labels::add_implicit_labels(&opts, main);

    let main = inline_const::inline_scalar_values(main, &args)?;
    debug_env.print("Python-like AST after scalar parameter inlining", &main);

    // Ensure the AST contains any degree of parallelism - otherwise, there is no point in using
    // this framework at all.
    par::ensure_parallelism(&main, &opts.parallelize)?;

    // Applies the type-checker, which resolves shape sizes and monomorphizes functions based on
    // the provided arguments. The result is an AST containing all monomorphized versions of
    // top-level definitions. Before printing the AST, we eliminate unnecessary duplicates of the
    // same function, as the type-checker may produce such instances.
    let scalar_sizes = ScalarSizes::from_opts(&opts);
    let ast = type_check::apply(main, &args, tops, &opts)?;
    let ast = eliminate_duplicate_functions::apply(ast)?;
    debug_env.print("Python-like AST after type-checking", &ast);

    // Adjusts indexing using negative literals to properly offset it with respect to the size of
    // the corresponding dimension, to achieve a similar behavior as in Python.
    let ast = adjust_negative_indices::apply(ast)?;

    // Transform slice statements into for-loops.
    let ast = slice_transformation::apply(ast, &scalar_sizes)?;
    debug_env.print("Python-like AST after slice transformation", &ast);

    // Replace call expressions targeting callback functions with explicit callback nodes. For
    // these nodes, we keep the Python AST details including the shapes of all subexpressions. When
    // constructing the callback functions, we need the shapes to be able to re-construct the
    // buffers in the wrapper.
    let ast = replace_callbacks::apply(ast);
    debug_env.print("Python-like AST after specializing callback calls", &ast);

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

#[macro_export]
macro_rules! py_internal_error {
    ($i:expr,$($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::internal_err($i.error_msg(format!($($t)*)))))
    }
}
