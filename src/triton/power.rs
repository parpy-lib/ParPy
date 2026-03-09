use super::ast::*;
use crate::utils::power;
use crate::utils::smap::SMapAccum;

fn try_unwrap_convert(e: &Expr) -> Option<(Expr, Type)> {
    match e {
        Expr::Convert {value, ty, ..} => Some((*value.clone(), ty.clone())),
        _ => None
    }
}

fn reconvert_rhs(e: Expr, conv_ty: &Type) -> Expr {
    match e {
        Expr::BinOp {lhs, op: BinOp::Pow, rhs, ty, i} => {
            let rhs = Box::new(Expr::Convert {
                value: rhs,
                ty: conv_ty.clone(),
                i: i.clone()
            });
            Expr::BinOp {lhs, op: BinOp::Pow, rhs, ty, i}
        },
        _ => e.smap(|e| reconvert_rhs(e, conv_ty))
    }
}

fn simplify_power_operator_expr(e: Expr) -> Expr {
    match e {
        Expr::BinOp {lhs, op: BinOp::Pow, rhs, ty, i} => {
            let lhs = simplify_power_operator_expr(*lhs);
            let rhs = simplify_power_operator_expr(*rhs);
            // If the right-hand side is a literal value wrapped in a convert, we unwrap it and
            // re-wrap the result if the power operator was not eliminated.
            if let Some((value, conv_ty)) = try_unwrap_convert(&rhs) {
                reconvert_rhs(power::simplify_power_operator(lhs, value, ty, i), &conv_ty)
            } else {
                power::simplify_power_operator(lhs, rhs, ty, i)
            }
        },
        _ => e.smap(simplify_power_operator_expr)
    }
}

fn simplify_power_operator_stmt(s: Stmt) -> Stmt {
    s.smap(simplify_power_operator_stmt)
        .smap(simplify_power_operator_expr)
}

fn simplify_power_operator_top(t: Top) -> Top {
    match t {
        Top::FunDef {id, params, body, i} => {
            let body = body.smap(simplify_power_operator_stmt);
            Top::FunDef {id, params, body, i}
        },
        Top::KernelFunDef {decorators, id, params, body, i} => {
            let body = body.smap(simplify_power_operator_stmt);
            Top::KernelFunDef {decorators, id, params, body, i}
        },
        _ => t
    }
}

pub fn simplify_power_operator(ast: Ast) -> Ast {
    Ast {tops: ast.tops.smap(simplify_power_operator_top)}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::triton::ast_builder::*;

    #[test]
    fn simplify_pow_int() {
        let e = Expr::BinOp {
            lhs: Box::new(var("x", None)),
            op: BinOp::Pow,
            rhs: Box::new(float(2.0)),
            ty: Type::Void,
            i: i()
        };
        let expected = Expr::BinOp {
            lhs: Box::new(var("x", None)),
            op: BinOp::Mul,
            rhs: Box::new(var("x", None)),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(simplify_power_operator_expr(e), expected);
    }

    #[test]
    fn simplify_pow_cast_exponent() {
        let e = Expr::BinOp {
            lhs: Box::new(var("x", None)),
            op: BinOp::Pow,
            rhs: Box::new(Expr::Convert {
                value: Box::new(float(2.0)),
                ty: Type::Tensor {sz: ElemSize::F64, shape: Shape::Num(1)},
                i: i()
            }),
            ty: Type::Void,
            i: i()
        };
        let expected = Expr::BinOp {
            lhs: Box::new(var("x", None)),
            op: BinOp::Mul,
            rhs: Box::new(var("x", None)),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(simplify_power_operator_expr(e), expected);
    }
}
