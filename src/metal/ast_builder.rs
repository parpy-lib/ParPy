use crate::metal::ast::*;
use crate::utils::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;

pub fn scalar(sz: ElemSize) -> Type {
    Type::Scalar {sz}
}

pub fn pointer(ty: Type, mem: MemSpace) -> Type {
    Type::Pointer {ty: Box::new(ty), mem}
}

pub fn id(s: &str) -> Name {
    Name::new(s.to_string())
}

pub fn var(s: &str, ty: Type) -> Expr {
    Expr::Var {id: id(s), ty, i: Info::default()}
}

pub fn int(v: i128, sz: ElemSize) -> Expr {
    Expr::Int {v, ty: scalar(sz), i: Info::default()}
}

pub fn float(v: f64, sz: ElemSize) -> Expr {
    Expr::Float {v, ty: scalar(sz), i: Info::default()}
}

pub fn unop(op: UnOp, arg: Expr) -> Expr {
    let ty = arg.get_type().clone();
    Expr::UnOp {op, arg: Box::new(arg), ty, i: Info::default()}
}

pub fn binop(l: Expr, op: BinOp, r: Expr, ty: Type) -> Expr {
    Expr::BinOp {lhs: Box::new(l), op, rhs: Box::new(r), ty, i: Info::default()}
}

pub fn definition(ty: Type, id: Name, expr: Expr) -> Stmt {
    Stmt::Definition {ty, id, expr}
}
