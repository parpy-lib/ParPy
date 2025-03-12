use crate::ir::ast::BinOp;
use crate::ir::ast::UnOp;
use crate::utils::info::*;

pub enum LitKind {
    Bool, Int, Float
}

pub trait CFExpr<T> {
    fn mk_unop(op: UnOp, arg: Self, ty: T, i: Info) -> Self
        where Self: Sized;
    fn mk_binop(lhs: Self, op: BinOp, rhs: Self, ty: T, i: Info) -> Self
        where Self: Sized;
    fn bool_expr(v: bool, ty: T, i: Info) -> Self
        where Self: Sized;
    fn int_expr(v: i64, ty: T, i: Info) -> Self
        where Self: Sized;
    fn float_expr(v: f64, ty: T, i: Info) -> Self
        where Self: Sized;

    fn get_bool_value(&self) -> Option<bool>;
    fn get_int_value(&self) -> Option<i64>;
    fn get_float_value(&self) -> Option<f64>;

    fn literal_kind(&self) -> Option<LitKind> {
        if self.get_bool_value().is_some() {
            Some(LitKind::Bool)
        } else if self.get_int_value().is_some() {
            Some(LitKind::Int)
        } else if self.get_float_value().is_some() {
            Some(LitKind::Float)
        } else {
            None
        }
    }
}

pub trait CFType {
    fn is_bool(&self) -> bool;
    fn is_int(&self) -> bool;
    fn is_float(&self) -> bool;
}

fn apply_int_unop<T, E: CFExpr<T>>(
    op: UnOp,
    arg: E,
    ty: T,
    i: Info
) -> E {
    let v = arg.get_int_value().unwrap();
    let o = match op {
        UnOp::Sub => Some(-v),
        UnOp::Abs => Some(v.abs()),
        _ => None
    };
    match o {
        Some(v) => CFExpr::int_expr(v, ty, i),
        None => CFExpr::mk_unop(op, arg, ty, i)
    }
}

fn apply_float_unop<T, E: CFExpr<T>>(
    op: UnOp,
    arg: E,
    ty: T,
    i: Info
) -> E {
    let v = arg.get_float_value().unwrap();
    let o = match op {
        UnOp::Sub => Some(-v),
        UnOp::Exp => Some(v.exp()),
        UnOp::Log if v > 0.0 => Some(v.ln()),
        UnOp::Abs => Some(v.abs()),
        _ => None
    };
    match o {
        Some(v) => CFExpr::float_expr(v, ty, i),
        None => CFExpr::mk_unop(op, arg, ty, i)
    }
}

pub fn constant_fold_unop<T, E: CFExpr<T>>(
    op: UnOp,
    arg: E,
    ty: T,
    i: Info
) -> E {
    match arg.literal_kind() {
        Some(LitKind::Int) => apply_int_unop(op, arg, ty, i),
        Some(LitKind::Float) => apply_float_unop(op, arg, ty, i),
        _ => CFExpr::mk_unop(op, arg, ty, i)
    }
}

fn is_bool_neutral_elem<T, E: CFExpr<T>>(op: &BinOp, e: &E) -> bool {
    let v = e.get_bool_value().unwrap();
    match op {
        BinOp::And => v,
        BinOp::Or => !v,
        _ => false
    }
}

fn apply_bool_binop<T, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    let lv = lhs.get_bool_value().unwrap();
    let rv = rhs.get_bool_value().unwrap();
    match op {
        BinOp::And => CFExpr::bool_expr(lv && rv, ty, i),
        BinOp::Or => CFExpr::bool_expr(lv || rv, ty, i),
        _ => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}

fn is_int_neutral_elem<T, E: CFExpr<T>>(op: &BinOp, e: &E, is_rhs: bool) -> bool {
    let v = e.get_int_value().unwrap();
    match op {
        BinOp::Add => v == 0,
        BinOp::Sub if is_rhs => v == 0,
        BinOp::Mul => v == 1,
        BinOp::Div if is_rhs => v == 1,
        BinOp::Max => v == i64::MIN,
        BinOp::Min => v == i64::MAX,
        _ => false
    }
}

fn apply_int_int_binop<T, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    let lv = lhs.get_int_value().unwrap();
    let rv = rhs.get_int_value().unwrap();
    let o = match op {
        BinOp::Add => Some(lv + rv),
        BinOp::Sub => Some(lv - rv),
        BinOp::Mul => Some(lv * rv),
        BinOp::FloorDiv if rv != 0 => Some(lv / rv),
        BinOp::Rem if rv != 0 => Some(lv % rv),
        BinOp::BitAnd => Some(lv & rv),
        BinOp::Max => Some(i64::max(lv, rv)),
        BinOp::Min => Some(i64::min(lv, rv)),
        _ => None
    };
    match o {
        Some(v) => CFExpr::int_expr(v, ty, i),
        None => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}

fn apply_int_bool_binop<T, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    let lv = lhs.get_int_value().unwrap();
    let rv = rhs.get_int_value().unwrap();
    let o = match op {
        BinOp::Eq => Some(lv == rv),
        BinOp::Neq => Some(lv != rv),
        BinOp::Leq => Some(lv <= rv),
        BinOp::Geq => Some(lv >= rv),
        BinOp::Lt => Some(lv < rv),
        BinOp::Gt => Some(lv > rv),
        _ => None
    };
    match o {
        Some(v) => CFExpr::bool_expr(v, ty, i),
        None => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}

fn is_float_neutral_elem<T, E: CFExpr<T>>(op: &BinOp, e: &E, is_rhs: bool) -> bool {
    let v = e.get_float_value().unwrap();
    match op {
        BinOp::Add => v == 0.0,
        BinOp::Sub if is_rhs => v == 0.0,
        BinOp::Mul => v == 1.0,
        BinOp::Div if is_rhs => v == 1.0,
        _ => false
    }
}

fn apply_float_float_binop<T, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    let lv = lhs.get_float_value().unwrap();
    let rv = rhs.get_float_value().unwrap();
    let o = match op {
        BinOp::Add => Some(lv + rv),
        BinOp::Sub => Some(lv - rv),
        BinOp::Mul => Some(lv * rv),
        BinOp::Div => Some(lv / rv),
        BinOp::Max => Some(f64::max(lv, rv)),
        BinOp::Min => Some(f64::min(lv, rv)),
        _ => None
    };
    match o {
        Some(v) => CFExpr::float_expr(v, ty, i),
        None => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}

fn apply_float_bool_binop<T, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    let lv = lhs.get_float_value().unwrap();
    let rv = rhs.get_float_value().unwrap();
    let o = match op {
        BinOp::Eq => Some(lv == rv),
        BinOp::Neq => Some(lv != rv),
        BinOp::Leq => Some(lv <= rv),
        BinOp::Geq => Some(lv >= rv),
        BinOp::Lt => Some(lv < rv),
        BinOp::Gt => Some(lv > rv),
        _ => None
    };
    match o {
        Some(v) => CFExpr::bool_expr(v, ty, i),
        None => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}

pub fn constant_fold_binop<T: CFType, E: CFExpr<T>>(
    lhs: E,
    op: BinOp,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    match (lhs.literal_kind(), rhs.literal_kind()) {
        (Some(LitKind::Bool), Some(LitKind::Bool)) =>
            apply_bool_binop(lhs, op, rhs, ty, i),
        (None, Some(LitKind::Bool)) if is_bool_neutral_elem(&op, &rhs) => lhs,
        (Some(LitKind::Bool), None) if is_bool_neutral_elem(&op, &lhs) => rhs,
        (Some(LitKind::Int), Some(LitKind::Int)) => {
            if ty.is_int() {
                apply_int_int_binop(lhs, op, rhs, ty, i)
            } else if ty.is_bool() {
                apply_int_bool_binop(lhs, op, rhs, ty, i)
            } else {
                CFExpr::mk_binop(lhs, op, rhs, ty, i)
            }
        },
        (None, Some(LitKind::Int)) if is_int_neutral_elem(&op, &rhs, true) => lhs,
        (Some(LitKind::Int), None) if is_int_neutral_elem(&op, &lhs, false) => rhs,
        (Some(LitKind::Float), Some(LitKind::Float)) => {
            if ty.is_float() {
                apply_float_float_binop(lhs, op, rhs, ty, i)
            } else if ty.is_bool() {
                apply_float_bool_binop(lhs, op, rhs, ty, i)
            } else {
                CFExpr::mk_binop(lhs, op, rhs, ty, i)
            }
        },
        (None, Some(LitKind::Float)) if is_float_neutral_elem(&op, &rhs, true) => lhs,
        (Some(LitKind::Float), None) if is_float_neutral_elem(&op, &lhs, false) => rhs,
        _ => CFExpr::mk_binop(lhs, op, rhs, ty, i)
    }
}
