use crate::utils::ast::*;
use crate::utils::constant_fold::CFExpr;
use crate::utils::info::Info;

fn make_multiply_chain<T: Clone, E: Clone + CFExpr<T>>(
    e: E,
    n: i64,
    ty: T,
    i: Info
) -> E {
    if n > 1 {
        let rhs = make_multiply_chain(e.clone(), n-1, ty.clone(), i.clone());
        E::mk_binop(e, BinOp::Mul, rhs, ty, i)
    } else {
        e
    }
}

fn try_simplify_power_operator<T: Clone, E: Clone + CFExpr<T>>(
    lhs: E,
    rhs: f64,
    ty: T,
    i: Info
) -> E {
    if rhs < 0.0 {
        // For a negative exponent, we rewrite the expression as one divided by the expression.
        E::mk_binop(
            E::float_expr(1.0, ty.clone(), i.clone()),
            BinOp::Div,
            try_simplify_power_operator(lhs, -rhs, ty.clone(), i.clone()),
            ty,
            i
        )
    } else if rhs.fract() == 0.5 {
        // We can replace a fraction of 0.5 with the use of the square root operator.
        E::mk_binop(
            try_simplify_power_operator(lhs.clone(), rhs.trunc(), ty.clone(), i.clone()),
            BinOp::Mul,
            E::mk_unop(UnOp::Sqrt, lhs, ty.clone(), i.clone()),
            ty,
            i
        )
    } else if rhs.trunc() == rhs && rhs <= 10.0 {
        // If the exponent is an integer value, we rewrite it to a series of multiplications.
        // However, to avoid excessive code explosion, we limit how many times we will do this.
        if rhs == 0.0 {
            E::float_expr(1.0, ty, i)
        } else {
            make_multiply_chain(lhs, rhs as i64, ty, i)
        }
    } else {
        E::mk_binop(lhs, BinOp::Pow, E::float_expr(rhs, ty.clone(), i.clone()), ty, i)
    }
}

/// Simplifies the binary power operator by rewriting it as a series of multiplications when the
/// exponent is known.
pub fn simplify_power_operator<T: Clone, E: Clone + CFExpr<T>>(
    lhs: E,
    rhs: E,
    ty: T,
    i: Info
) -> E {
    match rhs.get_float_value() {
        Some(v) => try_simplify_power_operator(lhs, v, ty, i),
        None => E::mk_binop(lhs, BinOp::Pow, rhs, ty, i)
    }
}
