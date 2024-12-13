use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub enum ParKind {
    GpuBlocks(i64),
    GpuThreads(i64)
}
