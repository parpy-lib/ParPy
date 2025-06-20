mod cuda;
mod gpu;
mod ir;
mod metal;
mod option;
mod par;
mod py;
mod utils;

use crate::utils::pprint::PrettyPrint;

use std::collections::BTreeMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast: Bound<'py, PyAny>,
    filepath: String,
    line_ofs: usize,
    col_ofs: usize,
    ir_asts: BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Bound<'py, PyCapsule>> {
    let py = py_ast.py().clone();

    // Convert the provided Python AST (parsed by the 'ast' module of Python) to a similar
    // representation of the Python AST using Rust data types.
    let def = py::parse_untyped_ast(py_ast, filepath, line_ofs, col_ofs)?;

    // Inline function calls referring to previously defined IR ASTs.
    let def = py::inline_function_calls(def, &ir_asts)?;

    // Wrap the intermediate AST in a capsule that we return to Python.
    let name = CString::new("Parir untyped Python AST")?;
    Ok(PyCapsule::new::<py::ast::FunDef>(py, def, Some(name))?)
}

#[pyfunction]
fn print_ir_ast<'py>(ir_ast_cap: Bound<'py, PyCapsule>) -> String {
    let untyped_ir_def : &py::ast::FunDef = unsafe {
        ir_ast_cap.reference()
    };
    untyped_ir_def.pprint_default()
}

#[pyfunction]
fn compile_ir<'py>(
    ir_ast_cap: Bound<'py, PyCapsule>,
    args: Vec<Bound<'py, PyAny>>,
    opts: option::CompileOptions
) -> PyResult<String> {
    // Extract a reference to the untyped AST parsed earlier.
    let untyped_ir_def : &py::ast::FunDef = unsafe {
        ir_ast_cap.reference()
    };

    let debug_env = utils::debug::init(&opts);
    debug_env.print("Untyped Python-like AST", untyped_ir_def);

    // Specialize the Python-like AST based on the provided arguments, inferring the types of all
    // expressions and inlining scalar argument values directly into the AST.
    let py_ast = untyped_ir_def.clone();
    let py_ast = py::specialize_ast_on_arguments(py_ast, args, &opts, &debug_env)?;
    debug_env.print("Specialized Python-like AST", &py_ast);

    // Converts the Python-like AST to an IR by removing or simplifying concepts from Python. For
    // example, this transformation
    // * Inserts top-level struct definitions for each Python dictionary.
    // * Replaces uses of tuples for indexing with an integer expression.
    // * Adds the parallelization arguments directly to the AST.
    let ir_ast = ir::from_python(py_ast, opts.parallelize, &debug_env)?;
    debug_env.print("IR AST", &ir_ast);

    match opts.backend {
        option::CompileBackend::Cuda => {
            let ast = cuda::codegen(ir_ast, &debug_env)?;
            debug_env.print("CUDA AST", &ast);
            Ok(ast.pprint_default())
        },
        option::CompileBackend::Metal => {
            let ast = metal::codegen(ir_ast, &debug_env)?;
            debug_env.print("Metal AST", &ast);
            Ok(ast.pprint_default())
        },
        option::CompileBackend::Auto => {
            Err(PyRuntimeError::new_err("Internal error: Auto backend should \
                                         be resolved before being passed to \
                                         the code generator."))
        },
        option::CompileBackend::Dummy => {
            Err(PyRuntimeError::new_err("Code generation is not supported \
                                         for the Dummy backend."))
        }
    }
}

#[pymodule]
fn parir(m : &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(print_ir_ast, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_function(wrap_pyfunction!(option::parallelize, m)?)?;
    m.add_class::<par::LoopPar>()?;
    m.add_class::<option::CompileBackend>()?;
    m.add_class::<option::CompileOptions>()?;
    Ok(())
}
