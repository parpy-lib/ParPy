use crate::ir::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;

pub fn id(x: &str) -> Name {
    Name::new(x.to_string())
}

pub fn scalar(sz: ElemSize) -> Type {
    Type::Tensor {sz, shape: vec![]}
}

pub fn shape(shape: Vec<i64>) -> Type {
    Type::Tensor {sz: ElemSize::I64, shape}
}

pub fn bool_ty() -> Type {
    scalar(ElemSize::Bool)
}

pub fn var(v: &str, ty: Type) -> Expr {
    let id = Name::new(v.to_string());
    Expr::Var {id, ty, i: Info::default()}
}

pub fn bool_expr(v: bool) -> Expr {
    Expr::Bool {v, ty: scalar(ElemSize::Bool), i: Info::default()}
}

pub fn int(v: i128, o: Option<ElemSize>) -> Expr {
    let ty = scalar(o.unwrap_or(ElemSize::I64));
    Expr::Int {v, ty, i: Info::default()}
}

pub fn float(v: f64, o: Option<ElemSize>) -> Expr {
    let ty = scalar(o.unwrap_or(ElemSize::F64));
    Expr::Float {v, ty, i: Info::default()}
}

pub fn unop(op: UnOp, arg: Expr) -> Expr {
    let ty = arg.get_type().clone();
    let i = arg.get_info();
    Expr::UnOp {op, arg: Box::new(arg), ty, i}
}

pub fn binop(lhs: Expr, op: BinOp, rhs: Expr, res_ty: Option<Type>) -> Expr {
    let ty = match res_ty {
        Some(ty) => ty,
        None => lhs.get_type().clone()
    };
    let i = lhs.get_info();
    Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
}

pub fn tensor_access(target: Expr, idx: Expr, ty: Type) -> Expr {
    Expr::TensorAccess {
        target: Box::new(target), idx: Box::new(idx), ty, i: Info::default()
    }
}

pub fn assign(lhs: Expr, rhs: Expr) -> Stmt {
    Stmt::Assign {dst: lhs, expr: rhs, i: Info::default()}
}

pub fn definition(ty: Type, id: Name, expr: Expr) -> Stmt {
    Stmt::Definition {ty, id, expr, i: Info::default()}
}

pub fn loop_par(n: i64) -> LoopPar {
    LoopPar::default().threads(n).unwrap()
}

pub fn for_loop_complete(
    var: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    par: LoopPar,
    body: Vec<Stmt>
) -> Stmt {
    Stmt::For {var, lo, hi, step, body, par, i: Info::default()}
}

pub fn for_loop(var: Name, n: i64, body: Vec<Stmt>) -> Stmt {
    let par = loop_par(n);
    for_loop_complete(var, int(0, None), int(10, None), 1, par, body)
}

fn if_cond_complete(cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    Stmt::If {cond, thn, els, i: Info::default()}
}

pub fn if_cond(thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
    if_cond_complete(bool_expr(true), thn, els)
}

fn while_loop_complete(cond: Expr, body: Vec<Stmt>) -> Stmt {
    Stmt::While {cond, body, i: Info::default()}
}

pub fn while_loop(body: Vec<Stmt>) -> Stmt {
    while_loop_complete(bool_expr(true), body)
}

pub fn sync_point(kind: SyncPointKind) -> Stmt {
    Stmt::SyncPoint {kind, i: Info::default()}
}

pub fn fun_def(body: Vec<Stmt>) -> FunDef {
    FunDef {
        id: Name::new("main".to_string()),
        params: vec![],
        body,
        res_ty: Type::Void,
        i: Info::default()
    }
}
