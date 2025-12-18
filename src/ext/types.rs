use crate::utils::ast::ElemSize;
use crate::utils::name::Name;

use pyo3::prelude::*;

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeVar {
    pub id: Name
}

#[pymethods]
impl ShapeVar {
    #[new]
    fn new() -> ShapeVar {
        ShapeVar {id: Name::sym_str("")}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct TypeVar {
    pub id: Name
}

#[pymethods]
impl TypeVar {
    #[new]
    fn new() -> TypeVar {
        TypeVar {id: Name::sym_str("")}
    }
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Buffer(ElemSize, Vec<ShapeVar>),
    VarBuffer(TypeVar, Vec<ShapeVar>),
}
