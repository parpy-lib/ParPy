use super::ast::*;
use crate::utils::constant_fold::*;
use crate::utils::info::Info;
use crate::utils::smap::SMapAccum;

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
            Type::Tensor {sz, shape} => shape.is_empty() && *sz == ElemSize::Bool,
            _ => false
        }
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Tensor {sz, shape} => shape.is_empty() && sz.is_signed_integer(),
            _ => false
        }
    }

    fn is_float(&self) -> bool {
        match self {
            Type::Tensor {sz, shape} => shape.is_empty() && sz.is_floating_point(),
            _ => false
        }
    }
}

fn fold_expr(e: Expr) -> Expr {
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
        Expr::Convert {e, ty} => {
            let e = fold_expr(*e);
            match e {
                Expr::Float {v, i, ..} if v.is_infinite() => {
                    Expr::Float {v, ty, i}
                },
                _ => {
                    Expr::Convert {e: Box::new(e), ty}
                }
            }
        },
        Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::IfExpr {..} | Expr::StructFieldAccess {..} |
        Expr::TensorAccess {..} | Expr::Call {..} => e.smap(fold_expr)
    }
}

fn fold_stmt(s: Stmt) -> Stmt {
    s.smap(fold_stmt).smap(fold_expr)
}

fn fold_fun(fun: FunDef) -> FunDef {
    let body = fun.body.smap(fold_stmt);
    FunDef {body, ..fun}
}

pub fn fold(ast: Ast) -> Ast {
    let defs = ast.defs.smap(fold_fun);
    Ast {defs, ..ast}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder::*;

    fn cf(e: &Expr) -> Expr {
        fold_expr(e.clone())
    }

    #[test]
    fn add() {
        let e = binop(int(2), BinOp::Add, int(3), None);
        assert_eq!(cf(&e), int(5));
    }

    #[test]
    fn nested_sub_mul() {
        let e = binop(binop(int(4), BinOp::Sub, int(1), None), BinOp::Mul, int(2), None);
        assert_eq!(cf(&e), int(6));
    }

    #[test]
    fn float_sub() {
        let e = binop(float(2.5), BinOp::Sub, float(1.5), None);
        assert_eq!(cf(&e), float(1.0));
    }

    #[test]
    fn unary_sub() {
        let e = unop(UnOp::Sub, float(2.5));
        assert_eq!(cf(&e), float(-2.5));
    }

    #[test]
    fn exp_sub() {
        let e = unop(UnOp::Exp, binop(float(2.5), BinOp::Sub, float(2.5), None));
        assert_eq!(cf(&e), float(1.0));
    }

    #[test]
    fn int_equality() {
        let e = binop(int(2), BinOp::Eq, int(3), Some(bool_ty()));
        assert_eq!(cf(&e), bool_expr(false));
    }

    #[test]
    fn float_lt() {
        let e = binop(float(1.5), BinOp::Lt, float(2.5), Some(bool_ty()));
        assert_eq!(cf(&e), bool_expr(true));
    }

    #[test]
    fn div_by_zero_untouched() {
        let e = binop(int(3), BinOp::Div, int(0), None);
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn log_zero_untouched() {
        let e = unop(UnOp::Log, float(0.0));
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn invalid_types_untouched() {
        let e = binop(int(2), BinOp::Add, float(3.0), None);
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn convert_float32_inf() {
        let float32 = Type::Tensor {sz: ElemSize::F32, shape: vec![]};
        let e = Expr::Convert {
            e: Box::new(float(f64::INFINITY)),
            ty: float32.clone()
        };
        assert_eq!(cf(&e), float_with_ty(f64::INFINITY, Some(float32)));
    }

    #[test]
    fn convert_float32_neginf() {
        let float32 = Type::Tensor {sz: ElemSize::F32, shape: vec![]};
        let e = Expr::Convert {
            e: Box::new(float(-f64::INFINITY)),
            ty: float32.clone()
        };
        assert_eq!(cf(&e), float_with_ty(-f64::INFINITY, Some(float32)));
    }

    #[test]
    fn complicated_constant_fold_not_supported() {
        // Could be simplified to 2x + 2
        let x = var("x");
        let e = binop(binop(x.clone(), BinOp::Add, int(1), None), BinOp::Mul, int(2), None);
        assert_eq!(cf(&e), e);
    }

    fn test_identity(op: BinOp, id: Expr, rhs: bool) {
        let x = var("x");
        let e = if rhs {
            binop(x.clone(), op, id, None)
        } else {
            binop(id, op, x.clone(), None)
        };
        assert_eq!(cf(&e), x);
    }

    #[test]
    fn add_identity() {
        test_identity(BinOp::Add, float(0.0), true);
        test_identity(BinOp::Add, float(0.0), false);
        test_identity(BinOp::Add, int(0), true);
        test_identity(BinOp::Add, int(0), false);
    }

    #[test]
    fn sub_identity_rhs() {
        test_identity(BinOp::Sub, int(0), true);
    }

    #[test]
    #[should_panic]
    fn sub_identity_lhs() {
        test_identity(BinOp::Sub, int(0), false);
    }

    #[test]
    fn mul_identity() {
        test_identity(BinOp::Mul, float(1.0), true);
        test_identity(BinOp::Mul, float(1.0), false);
        test_identity(BinOp::Mul, int(1), true);
        test_identity(BinOp::Mul, int(1), false);
    }

    #[test]
    fn div_identity() {
        test_identity(BinOp::Div, float(1.0), true);
        test_identity(BinOp::Div, int(1), true);
    }
}
