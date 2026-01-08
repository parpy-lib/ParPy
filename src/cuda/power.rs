use super::ast::*;
use crate::utils::info::Info;
use crate::utils::smap::SMapAccum;

fn extract_float_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Float {v, ..} => Some(*v),
        _ => None
    }
}

fn make_multiply_chain(e: Expr, n: i64, ty: Type, i: Info) -> Expr {
    if n > 1 {
        let rhs = make_multiply_chain(e.clone(), n-1, ty.clone(), i.clone());
        Expr::BinOp {
            lhs: Box::new(e),
            op: BinOp::Mul,
            rhs: Box::new(rhs),
            ty,
            i
        }
    } else {
        e
    }
}

fn try_simplify_power_operator(lhs: Expr, rhs: f64, ty: Type, i: Info) -> Expr {
    // NOTE(larshum, 2026-01-08): We intentionally impose a limit on the number of times we will
    // unroll a power operation with an integer exponent into multiplications, to avoid excessive
    // code explosion.
    if rhs < 0.0 {
        Expr::BinOp {
            lhs: Box::new(Expr::Float {v: 1.0, ty: ty.clone(), i: i.clone()}),
            op: BinOp::Div,
            rhs: Box::new(try_simplify_power_operator(lhs, -rhs, ty.clone(), i.clone())),
            ty, i
        }
    } else if rhs.fract() == 0.5 {
        Expr::BinOp {
            lhs: Box::new(try_simplify_power_operator(
                 lhs.clone(),
                 rhs.trunc(),
                 ty.clone(),
                 i.clone()
            )),
            op: BinOp::Mul,
            rhs: Box::new(Expr::UnOp {
                op: UnOp::Sqrt,
                arg: Box::new(lhs),
                ty: ty.clone(),
                i: i.clone()
            }),
            ty, i
        }
    } else if rhs.trunc() == rhs && rhs <= 10.0 {
        if rhs == 0.0 {
            Expr::Float {v: 1.0, ty, i}
        } else {
            make_multiply_chain(lhs, rhs as i64, ty, i)
        }
    } else {
        Expr::BinOp {
            lhs: Box::new(lhs),
            op: BinOp::Pow,
            rhs: Box::new(Expr::Float {v: rhs, ty: ty.clone(), i: i.clone()}),
            ty, i
        }
    }
}

fn simplify_power_operator_expr(e: Expr) -> Expr {
    match e {
        Expr::BinOp {lhs, op: BinOp::Pow, rhs, ty, i} => {
            match extract_float_value(&rhs) {
                Some(v) => try_simplify_power_operator(*lhs, v, ty, i),
                None => Expr::BinOp {lhs, op: BinOp::Pow, rhs, ty, i}
            }
        },
        _ => e.smap(simplify_power_operator_expr)
    }
}

fn simplify_power_operator_stmt(s: Stmt) -> Stmt {
    s.smap(simplify_power_operator_stmt).smap(simplify_power_operator_expr)
}

fn simplify_power_operator_top(t: Top) -> Top {
    match t {
        Top::VarDef {ty, id, init} => {
            let init = match init {
                Some(e) => Some(simplify_power_operator_expr(e)),
                None => None
            };
            Top::VarDef {ty, id, init}
        },
        Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
            let body = body.smap(simplify_power_operator_stmt);
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

pub fn simplify_power_operator(ast: Ast) -> Ast {
    ast.into_iter()
        .map(simplify_power_operator_top)
        .collect::<Ast>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cuda::ast_builder::*;

    fn mk_pow(lhs: &str, rhs: f64, sz: ElemSize) -> Expr {
        binop(
            var(lhs, scalar(sz.clone())),
            BinOp::Pow,
            float(rhs, sz.clone()),
            scalar(sz)
        )
    }

    #[test]
    fn pow_zero() {
        let e = mk_pow("x", 0.0, ElemSize::F32);
        assert_eq!(simplify_power_operator_expr(e), float(1.0, ElemSize::F32));
    }

    #[test]
    fn pow_one() {
        let e = mk_pow("x", 1.0, ElemSize::F32);
        assert_eq!(simplify_power_operator_expr(e), var("x", scalar(ElemSize::F32)));
    }

    #[test]
    fn pow_square() {
        let e = mk_pow("x", 2.0, ElemSize::F32);
        let expected = binop(
            var("x", scalar(ElemSize::F32)),
            BinOp::Mul,
            var("x", scalar(ElemSize::F32)),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_sqrt() {
        let e = mk_pow("x", 0.5, ElemSize::F32);
        let expected = binop(
            float(1.0, ElemSize::F32),
            BinOp::Mul,
            unop(
                UnOp::Sqrt,
                var("x", scalar(ElemSize::F32)),
                scalar(ElemSize::F32)
            ),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_neg_inverse() {
        let e = mk_pow("x", -1.0, ElemSize::F32);
        let expected = binop(
            float(1.0, ElemSize::F32),
            BinOp::Div,
            var("x", scalar(ElemSize::F32)),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_neg_square() {
        let e = mk_pow("x", -2.0, ElemSize::F32);
        let expected = binop(
            float(1.0, ElemSize::F32),
            BinOp::Div,
            binop(
                var("x", scalar(ElemSize::F32)),
                BinOp::Mul,
                var("x", scalar(ElemSize::F32)),
                scalar(ElemSize::F32)
            ),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_large_negative_int_exponent() {
        let e = mk_pow("x", -120.0, ElemSize::F32);
        let expected = binop(
            float(1.0, ElemSize::F32),
            BinOp::Div,
            mk_pow("x", 120.0, ElemSize::F32),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_negative_non_int_exponent() {
        let e = mk_pow("x", -1.25, ElemSize::F32);
        let expected = binop(
            float(1.0, ElemSize::F32),
            BinOp::Div,
            mk_pow("x", 1.25, ElemSize::F32),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_mul_and_sqrt() {
        let e = mk_pow("x", 2.5, ElemSize::F32);
        let expected = binop(
            binop(
                var("x", scalar(ElemSize::F32)),
                BinOp::Mul,
                var("x", scalar(ElemSize::F32)),
                scalar(ElemSize::F32)
            ),
            BinOp::Mul,
            unop(
                UnOp::Sqrt,
                var("x", scalar(ElemSize::F32)),
                scalar(ElemSize::F32)
            ),
            scalar(ElemSize::F32)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_non_int_exponent() {
        let e = mk_pow("x", 2.75, ElemSize::F32);
        assert_eq!(simplify_power_operator_expr(e.clone()), e);
    }

    #[test]
    fn pow_f16() {
        let e = mk_pow("x", 1.5, ElemSize::F16);
        let expected = binop(
            var("x", scalar(ElemSize::F16)),
            BinOp::Mul,
            unop(UnOp::Sqrt, var("x", scalar(ElemSize::F16)), scalar(ElemSize::F16)),
            scalar(ElemSize::F16)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn pow_f64() {
        let e = mk_pow("x", 1.5, ElemSize::F64);
        let expected = binop(
            var("x", scalar(ElemSize::F64)),
            BinOp::Mul,
            unop(UnOp::Sqrt, var("x", scalar(ElemSize::F64)), scalar(ElemSize::F64)),
            scalar(ElemSize::F64)
        );
        assert_eq!(simplify_power_operator_expr(e), expected);
    }
}
