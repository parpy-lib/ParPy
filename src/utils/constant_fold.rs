use crate::ir::ast::BinOp;
use crate::ir::ast::UnOp;
use crate::utils::info::*;

#[derive(Debug, PartialEq)]
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
    fn int_expr(v: i128, ty: T, i: Info) -> Self
        where Self: Sized;
    fn float_expr(v: f64, ty: T, i: Info) -> Self
        where Self: Sized;

    fn get_bool_value(&self) -> Option<bool>;
    fn get_int_value(&self) -> Option<i128>;
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
        UnOp::BitNeg => Some(!v),
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
        UnOp::Cos => Some(v.cos()),
        UnOp::Sin => Some(v.sin()),
        UnOp::Sqrt => Some(v.sqrt()),
        UnOp::Tanh => Some(v.tanh()),
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
        BinOp::Max => v == i128::MIN,
        BinOp::Min => v == i128::MAX,
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
        BinOp::BitOr => Some(lv | rv),
        BinOp::BitXor => Some(lv ^ rv),
        BinOp::BitShl => Some(lv << rv),
        BinOp::BitShr => Some(lv >> rv),
        BinOp::Max => Some(i128::max(lv, rv)),
        BinOp::Min => Some(i128::min(lv, rv)),
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast::*;
    use crate::py::ast_builder::*;

    fn int_lit(v: i128) -> Expr {
        int(v, Some(ElemSize::I64))
    }

    fn float_lit(v: f64) -> Expr {
        float(v, Some(ElemSize::F32))
    }

    fn bool_lit(v: bool) -> Expr {
        bool_expr(v, Some(ElemSize::Bool))
    }

    fn apply_int_unop_h(op: UnOp, e: Expr, sz: ElemSize) -> Expr {
        apply_int_unop(op, e, scalar(sz), Info::default())
    }

    fn apply_float_unop_h(op: UnOp, e: Expr, sz: ElemSize) -> Expr {
        apply_float_unop(op, e, scalar(sz), Info::default())
    }

    fn apply_bool_binop_h(l: Expr, op: BinOp, r: Expr) -> Expr {
        apply_bool_binop(l, op, r, scalar(ElemSize::Bool), Info::default())
    }

    fn apply_int_int_binop_h(l: Expr, op: BinOp, r: Expr) -> Expr {
        apply_int_int_binop(l, op, r, scalar(ElemSize::I64), Info::default())
    }

    fn apply_int_bool_binop_h(l: Expr, op: BinOp, r: Expr) -> Expr {
        apply_int_bool_binop(l, op, r, scalar(ElemSize::Bool), Info::default())
    }

    fn apply_float_float_binop_h(l: Expr, op: BinOp, r: Expr) -> Expr {
        apply_float_float_binop(l, op, r, scalar(ElemSize::F32), Info::default())
    }

    fn apply_float_bool_binop_h(l: Expr, op: BinOp, r: Expr) -> Expr {
        apply_float_bool_binop(l, op, r, scalar(ElemSize::Bool), Info::default())
    }

    #[test]
    fn apply_int_unop_sub() {
        assert_eq!(
            apply_int_unop_h(UnOp::Sub, int_lit(1), ElemSize::I64),
            int_lit(-1)
        );
    }

    #[test]
    fn apply_int_unop_abs() {
        assert_eq!(
            apply_int_unop_h(UnOp::Abs, int_lit(-1), ElemSize::I64),
            int_lit(1)
        );
    }

    #[test]
    fn apply_int_unop_bitneg() {
        assert_eq!(
            apply_int_unop_h(UnOp::BitNeg, int_lit(1), ElemSize::I64),
            int_lit(-2)
        );
    }

    #[test]
    fn apply_float_unop_exp() {
        assert_eq!(
            apply_float_unop_h(UnOp::Exp, float_lit(1.0), ElemSize::F32),
            float_lit(std::f64::consts::E)
        );
    }

    #[test]
    fn apply_float_unop_cos() {
        assert_eq!(
            apply_float_unop_h(UnOp::Cos, float_lit(0.0), ElemSize::F32),
            float_lit(1.0)
        );
    }

    #[test]
    fn apply_float_unop_sin() {
        assert_eq!(
            apply_float_unop_h(UnOp::Sin, float_lit(0.0), ElemSize::F32),
            float_lit(0.0)
        );
    }

    #[test]
    fn apply_float_unop_abs() {
        assert_eq!(
            apply_float_unop_h(UnOp::Abs, float_lit(-1.5), ElemSize::F32),
            float_lit(1.5)
        );
    }

    #[test]
    fn apply_bool_and() {
        assert_eq!(
            apply_bool_binop_h(bool_lit(true), BinOp::And, bool_lit(false)),
            bool_lit(false)
        );
    }

    #[test]
    fn apply_bool_or() {
        assert_eq!(
            apply_bool_binop_h(bool_lit(true), BinOp::Or, bool_lit(false)),
            bool_lit(true)
        );
    }

    #[test]
    fn apply_bool_eq() {
        assert_eq!(
            apply_bool_binop_h(bool_lit(true), BinOp::Eq, bool_lit(false)),
            binop(bool_lit(true), BinOp::Eq, bool_lit(false), scalar(ElemSize::Bool))
        );
    }

    #[test]
    fn apply_int_int_add() {
        assert_eq!(
            apply_int_int_binop_h(int_lit(1), BinOp::Add, int_lit(2)),
            int_lit(3)
        );
    }

    #[test]
    fn apply_int_int_mul() {
        assert_eq!(
            apply_int_int_binop_h(int_lit(2), BinOp::Mul, int_lit(3)),
            int_lit(6)
        );
    }

    #[test]
    fn apply_int_int_div_zero() {
        // NOTE: Divisions by zero are intentionally not simplified, making sure the division takes
        // place at runtime.
        assert_eq!(
            apply_int_int_binop_h(int_lit(3), BinOp::FloorDiv, int_lit(0)),
            binop(int_lit(3), BinOp::FloorDiv, int_lit(0), scalar(ElemSize::I64))
        );
    }

    #[test]
    fn apply_int_int_float_div() {
        assert_eq!(
            apply_int_int_binop_h(int_lit(3), BinOp::Div, int_lit(2)),
            binop(int_lit(3), BinOp::Div, int_lit(2), scalar(ElemSize::I64))
        );
    }

    #[test]
    fn apply_int_int_max() {
        assert_eq!(
            apply_int_int_binop_h(int_lit(3), BinOp::Max, int_lit(4)),
            int_lit(4)
        );
    }

    #[test]
    fn apply_int_int_bitand() {
        assert_eq!(
            apply_int_int_binop_h(int_lit(3), BinOp::BitAnd, int_lit(4)),
            int_lit(0)
        );
    }

    #[test]
    fn apply_int_bool_leq() {
        assert_eq!(
            apply_int_bool_binop_h(int_lit(3), BinOp::Leq, int_lit(4)),
            bool_lit(true)
        );
    }

    #[test]
    fn apply_int_bool_neq() {
        assert_eq!(
            apply_int_bool_binop_h(int_lit(3), BinOp::Neq, int_lit(4)),
            bool_lit(true)
        );
    }

    #[test]
    fn apply_int_bool_invalid_op() {
        assert_eq!(
            apply_int_bool_binop_h(int_lit(3), BinOp::BitAnd, int_lit(4)),
            binop(int_lit(3), BinOp::BitAnd, int_lit(4), scalar(ElemSize::I64))
        );
    }

    #[test]
    fn apply_float_float_add() {
        assert_eq!(
            apply_float_float_binop_h(float_lit(1.5), BinOp::Add, float_lit(2.5)),
            float_lit(4.0)
        );
    }

    #[test]
    fn apply_float_float_div() {
        assert_eq!(
            apply_float_float_binop_h(float_lit(1.5), BinOp::Div, float_lit(2.5)),
            float_lit(0.6)
        );
    }

    #[test]
    fn apply_float_float_floor_div() {
        assert_eq!(
            apply_float_float_binop_h(float_lit(1.5), BinOp::FloorDiv, float_lit(2.5)),
            binop(float_lit(1.5), BinOp::FloorDiv, float_lit(2.5), scalar(ElemSize::F32))
        );
    }

    #[test]
    fn apply_float_bool_eq() {
        assert_eq!(
            apply_float_bool_binop_h(float_lit(1.5), BinOp::Eq, float_lit(2.5)),
            bool_lit(false)
        );
    }

    #[test]
    fn apply_float_bool_lt() {
        assert_eq!(
            apply_float_bool_binop_h(float_lit(1.5), BinOp::Lt, float_lit(2.5)),
            bool_lit(true)
        );
    }

    #[test]
    fn apply_float_bool_invalid_op() {
        assert_eq!(
            apply_float_bool_binop_h(float_lit(1.5), BinOp::Add, float_lit(2.5)),
            binop(float_lit(1.5), BinOp::Add, float_lit(2.5), Type::Unknown)
        );
    }
}
