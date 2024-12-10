mod ir;
mod python;

use pyo3::prelude::*;

#[pyfunction]
fn compile_python_ast<'py>(
    py_ast : Bound<'py, PyAny>,
    args : Vec<Bound<'py, PyAny>>,
    par : Vec<ir::ParSpec>,
    ast_module : Bound<'py, PyModule>
) -> PyResult<String> {
    let ir_ast = python::to_ir(py_ast, args, par, ast_module);
    // TODO: translate the IR AST to a CUDA AST.
    Ok(format!("{ir_ast:?}"))
}

/// A Python module implemented in Rust.
#[pymodule]
fn parir(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_python_ast, m)?)?;
    m.add_class::<ir::ParKind>()?;
    m.add_class::<ir::ParSpec>()?;
    Ok(())
}
