mod codegen;
mod ir;
mod par;

use codegen::ast;

use std::collections::HashMap;
use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast : Bound<'py, PyAny>
) -> PyResult<Bound<'py, PyCapsule>> {
    let py = py_ast.py().clone();
    let ast = ir::from_py::to_untyped_ir(py_ast)?;

    // Wrap the intermediate AST in a capsule so we can pass it back to Python.
    let name = CString::new("Parir IR AST")?;
    Ok(PyCapsule::new::<ir::ast::Ast>(py, ast, Some(name))?)
}

#[pyfunction]
fn compile_ir<'py>(
    ir_ast_cap : Bound<'py, PyCapsule>,
    args : Vec<Bound<'py, PyAny>>,
    par : HashMap<String, Vec<par::ParKind>>
) -> PyResult<(String, String, String)> {
    let untyped_ir_ast : &ir::ast::Ast = unsafe {
        ir_ast_cap.reference()
    };
    let ir_ast = ir::from_py::to_typed_ir(untyped_ir_ast, args, par)?;
    let ast = codegen::from_ir::ir_to_code(ir_ast)?;
    Ok(ast::pprint_ast(&ast))
}

#[pymodule]
fn parir(m : &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_class::<par::ParKind>()?;
    Ok(())
}
