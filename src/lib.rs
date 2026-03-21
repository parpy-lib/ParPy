mod callbacks;
mod cuda;
mod ext;
mod gpu;
mod ir;
mod metal;
mod option;
mod par;
mod py;
mod triton;
mod utils;

use crate::utils::pprint::PrettyPrint;

use std::collections::BTreeMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyCapsule, PyDict};

#[pyfunction]
fn python_to_ir<'py>(
    py_ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>),
    py: Python<'py>
) -> PyResult<Bound<'py, PyCapsule>> {
    // Convert the provided Python AST (parsed by the 'ast' module of Python) to a similar
    // representation of the Python AST using Rust data types.
    let def = py::parse_untyped_ast(py_ast, info, &tops, vars)?;

    // Inline function calls referring to previously defined IR ASTs.
    let def = py::inline_function_calls(def, &tops)?;

    // Wrap the intermediate AST in a capsule that we return to Python.
    let t = py::ast::Top::FunDef {v: def};
    Ok(PyCapsule::new::<py::ast::Top>(py, t, None)?)
}

#[pyfunction]
fn declare_callback<'py>(
    py_ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>),
    py: Python<'py>
) -> PyResult<Bound<'py, PyCapsule>> {
    let t = py::convert_callback(py_ast, info, vars)?;
    Ok(PyCapsule::new::<py::ast::Top>(py, t, None)?)
}

#[pyfunction]
fn declare_external<'py>(
    py_ast: Bound<'py, PyAny>,
    info: (String, usize, usize),
    ext_id: String,
    target: utils::ast::Target,
    header: Option<String>,
    par: par::LoopPar,
    vars: (Bound<'py, PyDict>, Bound<'py, PyDict>),
    py: Python<'py>
) -> PyResult<Bound<'py, PyCapsule>> {
    let t = py::convert_external(py_ast, info, ext_id, target, header, par, vars)?;
    Ok(PyCapsule::new::<py::ast::Top>(py, t, None)?)
}

#[pyfunction]
fn get_function_name<'py>(cap: Bound<'py, PyCapsule>) -> String {
    let untyped_def: &py::ast::Top = unsafe {
        cap.reference()
    };
    match untyped_def {
        py::ast::Top::CallbackDecl {id, ..} |
        py::ast::Top::ExtDecl {id, ..} |
        py::ast::Top::FunDef {v: py::ast::FunDef {id, ..}} => id.get_str().clone(),
    }
}

#[pyfunction]
fn compile_ir<'py>(
    cap: Bound<'py, PyCapsule>,
    args: Vec<Bound<'py, PyAny>>,
    opts: option::CompileOptions,
    ir_asts: BTreeMap<String, Bound<'py, PyCapsule>>,
    py: Python<'py>
) -> PyResult<(Vec<Bound<'py, PyAny>>, Vec<String>, Vec<String>, String)> {
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
    let (ir_ast, classification) = ir::from_python(py_ast, &opts, &debug_env)?;
    debug_env.print("IR AST", &ir_ast);

    // Convert the IR AST to a GPU AST. The main difference between these two ASTs is that the GPU
    // AST distinguishes between code running on the host (CPU) and on the device (GPU). Further,
    // it includes constructs exclusive to GPU programming, such as thread and block indexing and
    // statements representing the allocation of shared memory.
    let gpu_ast = gpu::from_general_ir(ir_ast, classification, &opts, &debug_env)?;
    debug_env.print("GPU AST", &gpu_ast);

    // Extracts the callback functions used in the GPU AST and produces separate ASTs for these.
    // The result consists of three parts:
    // - A complete list of the argument types (as Ctypes types) of the entry point, including the
    //   callback functions.
    // - A list of Python ASTs (in the form of a function definition) representing a wrapper to
    //   each callback function, which wraps pointer arguments as a ParPy buffer (keeping track of
    //   its type and shape) and repeatedly calls the user-provided callback function, to minimize
    //   the overhead of argument wrapping.
    // - The updated GPU AST, with callback functions as arguments to the entry point function.
    let (argtypes, callback_asts, gpu_ast) = callbacks::from_gpu_ast(&opts, gpu_ast, py)?;
    debug_env.print("GPU AST after callback conversion", &gpu_ast);

    // Compile using the backend-specific approach to code generation. In the end, we pretty-print
    // the AST with and without symbols. The latter is used as a key to the cache - if only the
    // symbols differ between two ASTs, they should be considered equivalent.
    match opts.backend {
        option::CompileBackend::Cuda => {
            let gpu_ast = gpu::transform(gpu_ast, &opts, &debug_env)?;
            let ast = cuda::codegen(gpu_ast, &opts)?;
            debug_env.print("CUDA AST", &ast);
            Ok((
                argtypes,
                callback_asts,
                vec![ast.pprint_default()],
                ast.pprint_ignore_symbols()
            ))
        },
        option::CompileBackend::Metal => {
            let gpu_ast = gpu::transform(gpu_ast, &opts, &debug_env)?;
            let ast = metal::codegen(gpu_ast)?;
            debug_env.print("Metal AST", &ast);
            Ok((
                argtypes,
                callback_asts,
                vec![ast.pprint_default()],
                ast.pprint_ignore_symbols()
            ))
        },
        option::CompileBackend::Triton => {
            let (asts, unsymb_ast) = if opts.triton_native {
                let entry = triton::generate_native_entry_point(&gpu_ast)?;
                let ast = triton::codegen(gpu_ast, &opts, &debug_env)?;
                debug_env.print("Triton AST", &ast);
                debug_env.print("Native entry point AST", &entry);
                ( vec![ast.pprint_default(), entry.pprint_default()]
                , ast.pprint_ignore_symbols() )
            } else {
                let ast = triton::codegen(gpu_ast, &opts, &debug_env)?;
                debug_env.print("Triton AST", &ast);
                (vec![ast.pprint_default()], ast.pprint_ignore_symbols())
            };
            Ok((argtypes, callback_asts, asts, unsymb_ast))
        },
        option::CompileBackend::Auto => {
            Err(PyRuntimeError::new_err("Internal error: Auto backend should \
                                         be resolved before being passed to \
                                         the code generator."))
        },
    }
}

#[pymodule]
fn parpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_to_ir, m)?)?;
    m.add_function(wrap_pyfunction!(declare_callback, m)?)?;
    m.add_function(wrap_pyfunction!(declare_external, m)?)?;
    m.add_function(wrap_pyfunction!(get_function_name, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ir, m)?)?;
    m.add_function(wrap_pyfunction!(option::par, m)?)?;
    m.add_class::<par::LoopPar>()?;
    m.add_class::<option::CompileBackend>()?;
    m.add_class::<option::CompileOptions>()?;
    m.add_class::<utils::ast::ElemSize>()?;
    m.add_class::<utils::ast::ScalarSizes>()?;
    m.add_class::<utils::ast::Target>()?;
    m.add_class::<ext::buffer::DataType>()?;
    m.add_class::<ext::types::ExtType>()?;
    m.add_class::<ext::types::Shape>()?;
    m.add_class::<ext::types::TypeVar>()?;
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
