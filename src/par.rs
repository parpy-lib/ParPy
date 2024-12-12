use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub enum ParKind {
    GpuGrid(),
    GpuThreads(isize)
}
