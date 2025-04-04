use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

pub const REDUCE_PAR_LABEL: &'static str = "_reduce";

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct LoopPar {
    pub nthreads: i64,
    pub reduction: bool
}

impl Default for LoopPar {
    fn default() -> Self {
        LoopPar {nthreads: 0, reduction: false}
    }
}

#[pymethods]
impl LoopPar {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:?}")
    }

    #[new]
    fn _loop_par_new() -> Self {
        LoopPar::default()
    }

    pub fn threads(&self, nthreads: i64) -> PyResult<Self> {
        if self.nthreads == 0 {
            Ok(LoopPar {nthreads, ..self.clone()})
        } else {
            Err(PyRuntimeError::new_err("The number of threads can only be set once"))
        }
    }

    pub fn reduce(&self) -> Self {
        LoopPar {reduction: true, ..self.clone()}
    }
}

impl LoopPar {
    pub fn is_parallel(&self) -> bool {
        self.nthreads > 0
    }

    pub fn try_merge(mut self, other: Option<&LoopPar>) -> Option<LoopPar> {
        if let Some(o) = other {
            self.nthreads = if self.nthreads == 0 {
                Some(o.nthreads)
            } else if o.nthreads == 0 {
                Some(self.nthreads)
            } else {
                None
            }?;
            self.reduction = self.reduction || o.reduction;
        };
        Some(self)
    }
}
