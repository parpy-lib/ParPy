use crate::par;

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum CompileBackend {
    Auto, Cuda, Metal, Dummy
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /////////////////////
    // FRONT-END FLAGS //
    /////////////////////

    // The parallelization mapping, used to control how the code in the function is parallelized
    // based on labels associated with statements.
    #[pyo3(get, set)]
    pub parallelize: BTreeMap<String, par::LoopPar>,

    // When enabled, the compiler caches previous compilations of functions to reduce the overhead
    // of the JIT compilation on repeated runs.
    #[pyo3(get)]
    pub cache: bool,

    // When enabled, the function is not JIT compiled but instead executes sequentially using the
    // Python interpreter.
    #[pyo3(get, set)]
    pub seq: bool,

    // When enabled, the front-end compiler will print detailed information on why each disabled
    // backend is not considered to be available.
    #[pyo3(get, set)]
    pub verbose_backend_resolution: bool,

    ///////////////////
    // CODEGEN FLAGS //
    ///////////////////

    // Set the backend of the compiler, determining what kind of code it will generate for the
    // parallel regions of the function.
    #[pyo3(get, set)]
    pub backend: CompileBackend,

    // Enable to make the compiler print intermediate ASTs to standard output.
    #[pyo3(get)]
    pub debug_print: bool,

    // Enable to have to compiler report the time spent in various passes.
    #[pyo3(get)]
    pub debug_perf: bool,

    /////////////////
    // BUILD FLAGS //
    /////////////////

    // List of directories to add to the include path.
    #[pyo3(get, set)]
    pub includes: Vec<String>,

    // List of directories to add to the library path.
    #[pyo3(get, set)]
    pub libs: Vec<String>,

    // List of additional flags to be passed to the underlying compiler when building the generated
    // code. Which underlying compiler is used depends on the selected backend.
    #[pyo3(get, set)]
    pub extra_flags: Vec<String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            parallelize: BTreeMap::new(),
            cache: true,
            seq: false,
            verbose_backend_resolution: false,
            backend: CompileBackend::Auto,
            debug_print: false,
            debug_perf: false,
            includes: vec![],
            libs: vec![],
            extra_flags: vec![],
        }
    }
}

#[pymethods]
impl CompileOptions {
    #[new]
    fn _compile_options_new() -> Self {
        CompileOptions::default()
    }

    fn is_debug_enabled(&self) -> bool {
        self.debug_print || self.debug_perf
    }

    #[setter]
    fn set_cache(&mut self, v: bool) -> PyResult<()> {
        // After setting a debug flag to true, caching is automatically disabled. If we try to
        // enable caching when a debug flag is set, we get an error.
        if self.is_debug_enabled() && v {
            Err(PyRuntimeError::new_err("Caching cannot be enabled when debug flags are set"))
        } else {
            self.cache = v;
            Ok(())
        }
    }

    // When any of the debug flags are enabled, we automatically disable caching to ensure that the
    // debug output is actually printed.
    #[setter]
    fn set_debug_print(&mut self, v: bool) -> PyResult<()> {
        if v {
            self.cache = false;
        }
        self.debug_print = v;
        Ok(())
    }

    #[setter]
    fn set_debug_perf(&mut self, v: bool) -> PyResult<()> {
        if v {
            self.cache = false;
        }
        self.debug_perf = v;
        Ok(())
    }
}

// Constructs the default options object but containing the provided parallelization specification.
// This is useful when you want to parallelize while using the default options.
#[pyfunction]
pub fn parallelize(p: BTreeMap<String, par::LoopPar>) -> PyResult<CompileOptions> {
    let mut opts = CompileOptions::default();
    opts.parallelize = p;
    Ok(opts)
}

#[pyfunction]
pub fn seq() -> PyResult<CompileOptions> {
    let mut opts = CompileOptions::default();
    opts.seq = true;
    Ok(opts)
}
