use super::ast::*;
use crate::utils::info::Info;
use crate::utils::name::Name;

pub fn i() -> Info {
    Info::default()
}

pub fn id(s: &str) -> Name {
    Name::new(s.to_string())
}

pub fn scalar(sz: ElemSize) -> Type {
    Type::Scalar {sz}
}

pub fn bool_ty() -> Type {
    scalar(ElemSize::Bool)
}

pub fn pointer(ty: Type, mem: MemSpace) -> Type {
    Type::Pointer {ty: Box::new(ty), mem}
}

pub fn var(s: &str, ty: Type) -> Expr {
    Expr::Var {id: id(s), ty, i: Info::default()}
}

pub fn bool_expr(v: bool) -> Expr {
    Expr::Bool {v, ty: scalar(ElemSize::Bool), i: Info::default()}
}

pub fn int(v: i128, sz: Option<ElemSize>) -> Expr {
    let ty = scalar(sz.unwrap_or(ElemSize::I64));
    Expr::Int {v, ty, i: Info::default()}
}

pub fn float(v: f64, sz: Option<ElemSize>) -> Expr {
    let ty = scalar(sz.unwrap_or(ElemSize::F32));
    Expr::Float {v, ty, i: Info::default()}
}

pub fn unop(op: UnOp, arg: Expr, ty: Type) -> Expr {
    Expr::UnOp {op, arg: Box::new(arg), ty, i: Info::default()}
}

pub fn binop(l: Expr, op: BinOp, r: Expr, ty: Type) -> Expr {
    Expr::BinOp {lhs: Box::new(l), op, rhs: Box::new(r), ty, i: Info::default()}
}

pub fn thread_idx(dim: Dim) -> Expr {
    Expr::ThreadIdx {dim, ty: scalar(ElemSize::I64), i: Info::default()}
}

pub fn block_idx(dim: Dim) -> Expr {
    Expr::BlockIdx {dim, ty: scalar(ElemSize::I64), i: Info::default()}
}

pub fn array_access(target: Expr, idx: Expr, ty: Type) -> Expr {
    Expr::ArrayAccess {target: Box::new(target), idx: Box::new(idx), ty, i: Info::default()}
}

pub fn if_stmt(cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    Stmt::If {cond, thn, els, i: Info::default()}
}
