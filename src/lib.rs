mod ir;
mod python;

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast : Bound<'py, PyAny>
) -> PyResult<Bound<'py, PyCapsule>> {
    python::to_untyped_ir(py_ast)
}

#[pyfunction]
fn compile_ir<'py>(
    ir_ast_cap : Bound<'py, PyCapsule>,
    args : Vec<Bound<'py, PyAny>>,
    par : Vec<ir::ParSpec>,
) -> PyResult<String> {
    let untyped_ir_ast : &ir::Program = unsafe {
        ir_ast_cap.reference()
    };
    let ir_ast = python::to_typed_ir(untyped_ir_ast.clone(), args, par);
    Ok(format!("{ir_ast:?}"))
    //let ir_ast = python::to_ir(py_ast, args, par, ast_module);
    // TODO: translate the IR AST to a CUDA AST.
    //ir_ast.map(|ast| format!("{ast:?}"))
}

/// A Python module implemented in Rust.
#[pymodule]
fn parir(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_class::<ir::ParKind>()?;
    m.add_class::<ir::ParSpec>()?;
    Ok(())
}
