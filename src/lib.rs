mod codegen;
mod err;
mod info;
mod ir;
mod par;
mod py;

use codegen::ast;

use std::collections::HashMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast : Bound<'py, PyAny>,
    filepath : String,
    fst_line : usize
) -> PyResult<Bound<'py, PyCapsule>> {
    let py = py_ast.py().clone();
    let ast = py::to_untyped_ir(py_ast, filepath, fst_line)?;

    // Wrap the intermediate AST in a capsule so we can pass it back to Python.
    let name = CString::new("Parir IR AST")?;
    Ok(PyCapsule::new::<py::ast::Ast>(py, ast, Some(name))?)
}

#[pyfunction]
fn compile_ir<'py>(
    ir_ast_cap : Bound<'py, PyCapsule>,
    args : Vec<Bound<'py, PyAny>>,
    par : HashMap<String, par::ParSpec>
) -> PyResult<(String, String)> {
    let untyped_ir_ast : &py::ast::Ast = unsafe {
        ir_ast_cap.reference()
    };
    let ir_ast = py::to_typed_ir(untyped_ir_ast, args, par)?;
    let ast = codegen::from_ir::ir_to_code(ir_ast)?;
    Ok(ast::pprint_ast(&ast))
}

#[pymodule]
fn parir(m : &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_class::<par::ParKind>()?;
    m.add_class::<par::ParSpec>()?;
    Ok(())
}
