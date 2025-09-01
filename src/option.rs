use crate::par::LoopPar;
use crate::utils::ast::ElemSize;

use std::collections::BTreeMap;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass(eq, eq_int, hash, frozen)]
#[derive(Clone, Debug, PartialEq, Hash)]
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

    // When enabled, the front-end compiler will print detailed information on why each disabled
    // backend is not considered to be available.
    #[pyo3(get, set)]
    pub verbose_backend_resolution: bool,

    // Enable to have the compiler report the time spent in various passes and print the AST after
    // each major transformation pass.
    #[pyo3(get, set)]
    pub debug_print: bool,

    // Enable to have the compiler write the generated source code for the target backend to a
    // file.
    #[pyo3(get, set)]
    pub write_output: bool,

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

    // Enables the use of CUDA graphs in the generated CUDA code, which record the CUDA API calls
    // on the first use to reduce the overhead of repeated kernel launches.
    #[pyo3(get, set)]
    pub use_cuda_graphs: bool,

    // By default, the compiler decides the type to use for all integer and floating-point scalars.
    // Setting the flags below overrides the decision made by the compiler.
    #[pyo3(get)]
    pub force_int_size: Option<ElemSize>,
    #[pyo3(get)]
    pub force_float_size: Option<ElemSize>,

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
            verbose_backend_resolution: false,
            backend: CompileBackend::Auto,
            use_cuda_thread_block_clusters: false,
            max_thread_blocks_per_cluster: 8,
            use_cuda_graphs: false,
            force_int_size: None,
            force_float_size: None,
            debug_print: false,
            write_output: false,
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

    #[setter]
    fn set_force_int_size(&mut self, sz: ElemSize) -> PyResult<()> {
        if sz.is_integer() {
            self.force_int_size = Some(sz);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Cannot use a non integer type for \
                                         integer scalars."))
        }
    }

    #[setter]
    fn set_force_float_size(&mut self, sz: ElemSize) -> PyResult<()> {
        if sz.is_floating_point() {
            self.force_float_size = Some(sz);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("Cannot use a non floating-point type \
                                         for floating-point scalars."))
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;

    #[test]
    fn set_blocks_per_cluster_power_of_two() {
        let mut opts = CompileOptions::default();
        assert!(opts.set_max_thread_blocks_per_cluster(16).is_ok());
    }

    #[test]
    fn set_blocks_per_cluster_non_power_of_two() {
        let mut opts = CompileOptions::default();
        assert_py_error_matches(
            opts.set_max_thread_blocks_per_cluster(12),
            "thread blocks per cluster must be a power of two"
        );
    }
}
