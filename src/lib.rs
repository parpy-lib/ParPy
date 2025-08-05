mod buffer;
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
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;

#[pyfunction]
fn python_to_ir<'py>(
    py_ast: Bound<'py, PyAny>,
    filepath: String,
    line_ofs: usize,
    col_ofs: usize,
    tops: BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Bound<'py, PyCapsule>> {
    let py = py_ast.py().clone();

    // Convert the provided Python AST (parsed by the 'ast' module of Python) to a similar
    // representation of the Python AST using Rust data types.
    let def = py::parse_untyped_ast(py_ast, filepath, line_ofs, col_ofs, &tops)?;

    // Inline function calls referring to previously defined IR ASTs.
    let def = py::inline_function_calls(def, &tops)?;

    // Wrap the intermediate AST in a capsule that we return to Python.
    let t = py::ast::Top::FunDef {v: def};
    Ok(PyCapsule::new::<py::ast::Top>(py, t, None)?)
}

#[pyfunction]
fn make_external_declaration<'py>(
    id: String,
    ext_id: String,
    params: Vec<(String, py::ext::ExtType)>,
    res_ty: py::ext::ExtType,
    header: Option<String>,
    info: (String, usize, usize, usize, usize),
    py: Python<'py>
) -> PyResult<Bound<'py, PyCapsule>> {
    let t = py::ext::make_declaration(id, ext_id, params, res_ty, header, info);
    Ok(PyCapsule::new::<py::ast::Top>(py, t, None)?)
}

#[pyfunction]
fn print_ast<'py>(cap: Bound<'py, PyCapsule>) -> String {
    let untyped_def: &py::ast::Top = unsafe {
        cap.reference()
    };
    untyped_def.pprint_default()
}

#[pyfunction]
fn get_function_name<'py>(cap: Bound<'py, PyCapsule>) -> String {
    let untyped_def: &py::ast::Top = unsafe {
        cap.reference()
    };
    match untyped_def {
        py::ast::Top::ExtDecl {id, ..} |
        py::ast::Top::FunDef {v: py::ast::FunDef {id, ..}} => id.get_str().clone(),
    }
}

#[pyfunction]
fn compile_ir<'py>(
    cap: Bound<'py, PyCapsule>,
    args: Vec<Bound<'py, PyAny>>,
    opts: option::CompileOptions,
    ir_asts: BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<(String, String)> {
    // Extract a reference to the untyped AST parsed earlier.
    let t: &py::ast::Top = unsafe { cap.reference() };

    let debug_env = utils::debug::DebugEnv::new(&opts);
    debug_env.print("Untyped Python-like AST", t);

    // Specialize the Python-like AST based on the provided arguments, inferring the types of all
    // expressions and inlining scalar argument values directly into the AST.
    let py_ast = py::specialize_ast_on_arguments(t.clone(), args, &opts, ir_asts, &debug_env)?;
    debug_env.print("Specialized Python-like AST", &py_ast);

    // Converts the Python-like AST to an IR by removing or simplifying concepts from Python. For
    // example, this transformation
    // * Inserts top-level struct definitions for each Python dictionary.
    // * Replaces uses of tuples for indexing with an integer expression.
    // * Adds the parallelization arguments directly to the AST.
    let ir_ast = ir::from_python(py_ast, opts.parallelize.clone(), &debug_env)?;
    debug_env.print("IR AST", &ir_ast);

    // Compile using the backend-specific approach to code generation. In the end, we pretty-print
    // the AST with and without symbols. The latter is used as a key to the cache - if only the
    // symbols differ between two ASTs, they should be considered equivalent.
    match opts.backend {
        option::CompileBackend::Cuda => {
            let ast = cuda::codegen(ir_ast, &opts, &debug_env)?;
            debug_env.print("CUDA AST", &ast);
            Ok((ast.pprint_default(), ast.pprint_ignore_symbols()))
        },
        option::CompileBackend::Metal => {
            let ast = metal::codegen(ir_ast, &opts, &debug_env)?;
            debug_env.print("Metal AST", &ast);
            Ok((ast.pprint_default(), ast.pprint_ignore_symbols()))
        },
        option::CompileBackend::Auto => {
            Err(PyRuntimeError::new_err("Internal error: Auto backend should \
                                         be resolved before being passed to \
                                         the code generator."))
        },
    }
}

#[pymodule]
fn prickle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(make_external_declaration, m)?)?;
    m.add_function(wrap_pyfunction!(print_ast, m)?)?;
    m.add_function(wrap_pyfunction!(get_function_name, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_function(wrap_pyfunction!(option::par, m)?)?;
    m.add_function(wrap_pyfunction!(option::seq, m)?)?;
    m.add_class::<par::LoopPar>()?;
    m.add_class::<option::CompileBackend>()?;
    m.add_class::<option::CompileOptions>()?;
    m.add_class::<utils::ast::ElemSize>()?;
    m.add_class::<buffer::DataType>()?;
    m.add_class::<py::ext::ExtType>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::err::*;
    use crate::utils::info::Info;
    use pyo3::{Python, PyResult};
    use regex::Regex;
    use std::fmt;

    fn assert_error_msg_matches(err_msg: String, pat: &str) {
        let re = Regex::new(pat).unwrap();
        assert!(
            re.is_match(&err_msg),
            "Error message \"{0}\" did not match expected pattern \"{1}\"",
            err_msg, pat
        );
    }

    pub fn assert_py_error_matches<T: fmt::Debug>(r: PyResult<T>, pat: &str) {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err_msg = r.unwrap_err().value(py).to_string();
            assert_error_msg_matches(err_msg, pat)
        })
    }

    pub fn assert_error_matches<T: fmt::Debug>(r: CompileResult<T>, pat: &str) {
        let err_msg = format!("{}", r.unwrap_err());
        assert_error_msg_matches(err_msg, pat)
    }

    pub fn i() -> Info {
        Info::default()
    }
}
