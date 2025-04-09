use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

pub const REDUCE_PAR_LABEL: &'static str = "_reduce";
pub const DEFAULT_TPB: i64 = 1024;

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct LoopPar {
    pub nthreads: i64,
    pub reduction: bool,
    pub tpb: i64
}

impl Default for LoopPar {
    fn default() -> Self {
        LoopPar {nthreads: 0, reduction: false, tpb: DEFAULT_TPB}
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

    pub fn tpb(&self, tpb: i64) -> PyResult<Self> {
        if self.tpb > 0 && self.tpb % 32 == 0 {
            Ok(LoopPar {tpb, ..self.clone()})
        } else {
            Err(PyRuntimeError::new_err("The number of threads per block must \
                                         be a positive integer divisible by 32."))
        }
    }
}

fn merge_values(l: i64, r: i64, default_value: i64) -> Option<i64> {
    if l == default_value || l == r {
        Some(r)
    } else if r == default_value {
        Some(l)
    } else {
        None
    }
}

impl LoopPar {
    pub fn is_parallel(&self) -> bool {
        self.nthreads > 0
    }

    pub fn try_merge(mut self, other: Option<&LoopPar>) -> Option<LoopPar> {
        if let Some(o) = other {
            self.nthreads = merge_values(self.nthreads, o.nthreads, 0)?;
            self.tpb = merge_values(self.tpb, o.tpb, DEFAULT_TPB)?;
            self.reduction = self.reduction || o.reduction;
        };
        Some(self)
    }
}
