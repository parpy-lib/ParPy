mod cuda;
mod ir;
mod par;
mod py;
mod utils;

use crate::utils::pprint::PrettyPrint;

use std::collections::BTreeMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast: Bound<'py, PyAny>,
    filepath: String,
    fst_line: usize,
    ir_asts: BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Bound<'py, PyCapsule>> {
    let py = py_ast.py().clone();

    // Convert the provided Python AST (parsed by the 'ast' module of Python) to a similar
    // representation of the Python AST using Rust data types.
    let def = py::parse_untyped_ast(py_ast, filepath, fst_line)?;

    // Inline function calls referring to previously defined IR ASTs.
    let def = py::inline_function_calls(def, &ir_asts)?;

    // Wrap the intermediate AST in a capsule that we return to Python.
    let name = CString::new("Parir untyped Python AST")?;
    Ok(PyCapsule::new::<py::ast::FunDef>(py, def, Some(name))?)
}

#[pyfunction]
fn compile_ir<'py>(
    ir_ast_cap: Bound<'py, PyCapsule>,
    args: Vec<Bound<'py, PyAny>>,
    par: BTreeMap<String, Vec<par::ParKind>>
) -> PyResult<String> {
    // Extract a reference to the untyped AST parsed earlier.
    let untyped_ir_def : &py::ast::FunDef = unsafe {
        ir_ast_cap.reference()
    };

    // Specialize the Python-like AST based on the provided arguments, inferring the types of all
    // expressions and inlining scalar argument values in the AST.
    let py_ir_ast = py::type_check(untyped_ir_def.clone(), &args)?;
    let py_ir_ast = py::inline_scalar_values(py_ir_ast, &args)?;

    // Converts the Python-like AST to an IR by removing or simplifying concepts from Python. For
    // example, this transformation
    // * Inserts top-level struct definitions for each Python dictionary.
    // * Replaces uses of tuples for indexing with an integer expression.
    // * Adds the parallelization arguments directly to the AST.
    let ir_ast = ir::from_python(py_ir_ast, par)?;

    // Convert the IR AST to CUDA code, based on the parallel annotations on for-loops.
    let ast = cuda::codegen(ir_ast)?;

    Ok(ast.pprint_default())
}

#[pymodule]
fn parir(m : &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_class::<par::ParKind>()?;
    Ok(())
}
