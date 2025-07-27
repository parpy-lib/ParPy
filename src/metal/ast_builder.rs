use crate::metal::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;

pub fn scalar(sz: ElemSize) -> Type {
    Type::Scalar {sz}
}

pub fn id(s: &str) -> Name {
    Name::new(s.to_string())
}

pub fn var(s: &str, ty: Type) -> Expr {
    Expr::Var {id: id(s), ty, i: Info::default()}
}

pub fn float(v: f64, sz: ElemSize) -> Expr {
    Expr::Float {v, ty: scalar(sz), i: Info::default()}
}
