use super::ast::*;
use crate::test::*;
use crate::utils::name::Name;

pub fn id(s: &str) -> Name {
    Name::new(s.to_string())
}

pub fn var(s: &str, ty: Option<Type>) -> Expr {
    let ty = ty.unwrap_or(Type::Void);
    Expr::Var {id: id(s), ty, i: i()}
}

pub fn int(v: i128) -> Expr {
    Expr::Int {v, ty: Type::Void, i: i()}
}

pub fn float(v: f64) -> Expr {
    Expr::Float {v, ty: Type::Void, i: i()}
}
