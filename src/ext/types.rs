use crate::utils::ast::ElemSize;
use crate::utils::name::Name;

use pyo3::prelude::*;

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeVar {
    pub id: Name
}

#[pyclass(eq, frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Var(ShapeVar),
    Literal(i64),
}

#[pymethods]
impl Shape {
    #[staticmethod]
    fn make_var() -> Shape {
        Shape::Var(ShapeVar {id: Name::sym_str("")})
    }

    #[staticmethod]
    fn make_literal(n: i64) -> Shape {
        Shape::Literal(n)
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
    Buffer(ElemSize, Vec<Shape>),
    VarBuffer(TypeVar, Vec<Shape>),
}
