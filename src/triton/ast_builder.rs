use super::ast::*;
use crate::test::*;
use crate::utils::name::Name;

pub fn var(id: &str) -> Expr {
    Expr::Var {id: Name::sym_str(id), ty: Type::Void, i: i()}
}

pub fn int(v: i128) -> Expr {
    Expr::Int {v, ty: Type::Void, i: i()}
}
