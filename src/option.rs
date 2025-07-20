use crate::par::LoopPar;

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
pub enum CompileBackend {
    Auto, Cuda, Metal
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
    pub parallelize: BTreeMap<String, LoopPar>,

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

    // When this option is enabled, the compiler will opt to use thread block clusters in place of
    // inter-block synchronization where applicable. This is only available on the CUDA backend.
    #[pyo3(get, set)]
    pub use_cuda_thread_block_clusters: bool,

    // When the use of thread block clusters is enabled (see the above flag), this option sets the
    // maximum number of thread blocks per cluster.
    #[pyo3(get)]
    pub max_thread_blocks_per_cluster: i64,

    #[pyo3(get, set)]
    pub use_cuda_graphs: bool,

    // Enable to make the compiler print intermediate ASTs to standard output.
    #[pyo3(get, set)]
    pub debug_print: bool,

    // Enable to have to compiler report the time spent in various passes.
    #[pyo3(get, set)]
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
            seq: false,
            verbose_backend_resolution: false,
            backend: CompileBackend::Auto,
            use_cuda_thread_block_clusters: false,
            max_thread_blocks_per_cluster: 8,
            use_cuda_graphs: false,
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

    fn __str__(&self) -> String {
        format!("{self:?}")
    }

    #[setter]
    fn set_max_thread_blocks_per_cluster(&mut self, n: i64) -> PyResult<()> {
        if n > 0 && (n & (n-1)) == 0 {
            self.max_thread_blocks_per_cluster = n;
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("The number of thread blocks per \
                                         cluster must be a power of two."))
        }
    }
}

// Constructs the default options object but containing the provided parallelization specification.
// This is useful when you want to parallelize while using the default options.
#[pyfunction]
pub fn par(p: BTreeMap<String, LoopPar>) -> PyResult<CompileOptions> {
    let mut opts = CompileOptions::default();
    opts.parallelize = p;
    Ok(opts)
}

// Constructs the default options but requesting sequential execution.
#[pyfunction]
pub fn seq() -> PyResult<CompileOptions> {
    let mut opts = CompileOptions::default();
    opts.seq = true;
    Ok(opts)
}
