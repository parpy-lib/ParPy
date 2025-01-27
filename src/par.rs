use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub enum ParKind {
    GpuThreads(i64),
    GpuReduction {},
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
