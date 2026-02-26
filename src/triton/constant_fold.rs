use super::ast::*;
use crate::utils::constant_fold::*;
use crate::utils::info::Info;
use crate::utils::smap::{SFlatten, SMapAccum};

impl CFExpr<Type> for Expr {
    fn mk_unop(op: UnOp, arg: Expr, ty: Type, i: Info) -> Expr {
        Expr::UnOp {op, arg: Box::new(arg), ty, i}
    }

    fn mk_binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type, i: Info) -> Expr {
        Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
    }

    fn bool_expr(v: bool, ty: Type, i: Info) -> Expr {
        Expr::Bool {v, ty, i}
    }

    fn int_expr(v: i128, ty: Type, i: Info) -> Expr {
        Expr::Int {v, ty, i}
    }

    fn float_expr(v: f64, ty: Type, i: Info) -> Expr {
        Expr::Float {v, ty, i}
    }

    fn get_bool_value(&self) -> Option<bool> {
        match self {
            Expr::Bool {v, ..} => Some(*v),
            _ => None
        }
    }

    fn get_int_value(&self) -> Option<i128> {
        match self {
            Expr::Int {v, ..} => Some(*v),
            _ => None
        }
    }

    fn get_float_value(&self) -> Option<f64> {
        match self {
            Expr::Float {v, ..} => Some(*v),
            _ => None
        }
    }
}

impl CFType for Type {
    fn is_bool(&self) -> bool {
        match self {
            Type::Tensor {sz: ElemSize::Bool, ..} => true,
            _ => false
        }
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Tensor {sz, ..} => sz.is_integer(),
            _ => false
        }
    }

    fn is_float(&self) -> bool {
        match self {
            Type::Tensor {sz, ..} => sz.is_floating_point(),
            _ => false
        }
    }
}

pub fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::UnOp {op, arg, ty, i} => {
            let arg = fold_expr(*arg);
            constant_fold_unop(op, arg, ty, i)
        },
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = fold_expr(*lhs);
            let rhs = fold_expr(*rhs);
            constant_fold_binop(lhs, op, rhs, ty, i)
        },
        _ => e.smap(fold_expr)
    }
}

fn fold_stmt(acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Pass {..} => acc,
        s => s.smap(fold_expr).sflatten(acc, fold_stmt)
    }
}

fn fold_top(t: Top) -> Top {
    match t {
        Top::Import {..} => t,
        Top::FunDef {triton_jit, id, params, body, i} => {
            let body = body.sflatten(vec![], fold_stmt);
            Top::FunDef {triton_jit, id, params, body, i}
        },
    }
}

pub fn apply(ast: Ast) -> Ast {
    Ast {tops: ast.tops.smap(fold_top)}
}
