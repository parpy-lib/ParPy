use crate::py::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;

pub fn tyuk() -> Type {
    Type::Unknown
}

pub fn shape(v: Vec<i64>) -> Type {
    Type::Tensor {sz: ElemSize::I64, shape: v}
}

pub fn scalar(sz: ElemSize) -> Type {
    Type::Tensor {sz, shape: vec![]}
}

pub fn id(x: &str) -> Name {
    Name::new(x.to_string())
}

pub fn var(v: &str, ty: Type) -> Expr {
    Expr::Var {id: id(v), ty, i: Info::default()}
}

pub fn bool_expr(v: bool, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Bool {v, ty, i: Info::default()}
}

pub fn int(v: i64, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Int {v: v as i128, ty, i: Info::default()}
}

pub fn float(v: f64, o: Option<ElemSize>) -> Expr {
    let ty = o.map(scalar).unwrap_or(Type::Unknown);
    Expr::Float {v, ty, i: Info::default()}
}

pub fn unop(op: UnOp, arg: Expr) -> Expr {
    Expr::UnOp {op, arg: Box::new(arg), ty: Type::Unknown, i: Info::default()}
}

pub fn binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type) -> Expr {
    Expr::BinOp {
        lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: Info::default()
    }
}

pub fn subscript(target: Expr, idx: Expr) -> Expr {
    Expr::Subscript {
        target: Box::new(target), idx: Box::new(idx),
        ty: Type::Unknown, i: Info::default()
    }
}

pub fn slice(lo: Option<Expr>, hi: Option<Expr>) -> Expr {
    let wrap_box = |o: Option<Expr>| {
        if let Some(e) = o {
            Some(Box::new(e))
        } else {
            None
        }
    };
    let ty = Type::Unknown;
    Expr::Slice {lo: wrap_box(lo), hi: wrap_box(hi), ty, i: Info::default()}
}

pub fn tuple(elems: Vec<Expr>) -> Expr {
    Expr::Tuple {elems, ty: Type::Unknown, i: Info::default()}
}

pub fn call(f: &str, args: Vec<Expr>, ty: Type) -> Expr {
    Expr::Call {id: f.to_string(), args, ty, i: Info::default()}
}

pub fn assignment(lhs: Expr, rhs: Expr) -> Stmt {
    Stmt::Assign {dst: lhs, expr: rhs, labels: vec![], i: Info::default()}
}
