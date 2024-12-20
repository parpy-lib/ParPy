use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub enum ParKind {
    CpuThreads(u64),
    GpuThreads(u64),
    GpuBlocks(u64)
}

#[pymethods]
impl ParKind {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ParSpec {
    pub kind: ParKind
}

#[pymethods]
impl ParSpec {
    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:?}")
    }

    #[new]
    fn new(kind: ParKind) -> Self {
        ParSpec {kind}
    }
}
