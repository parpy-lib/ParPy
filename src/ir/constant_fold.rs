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

fn fold_def(def: FunDef) -> FunDef {
    let body = def.body.smap(fold_stmt);
    FunDef {body, ..def}
}

fn fold_top(t: Top) -> Top {
    match t {
        Top::FunDef {v} => Top::FunDef {v: fold_def(v)},
        Top::StructDef {..} | Top::ExtDecl {..} => t,
    }
}

pub fn fold(ast: Ast) -> Ast {
    let tops = ast.tops.smap(fold_top);
    let main = fold_def(ast.main);
    Ast {tops, main}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ast_builder::*;

    fn cf(e: &Expr) -> Expr {
        fold_expr(e.clone())
    }

    #[test]
    fn add() {
        let e = binop(int(2, None), BinOp::Add, int(3, None), None);
        assert_eq!(cf(&e), int(5, None));
    }

    #[test]
    fn nested_sub_mul() {
        let e = binop(
            binop(int(4, None), BinOp::Sub, int(1, None), None),
            BinOp::Mul, int(2, None), None
        );
        assert_eq!(cf(&e), int(6, None));
    }

    #[test]
    fn float_sub() {
        let e = binop(float(2.5, None), BinOp::Sub, float(1.5, None), None);
        assert_eq!(cf(&e), float(1.0, None));
    }

    #[test]
    fn unary_sub() {
        let e = unop(UnOp::Sub, float(2.5, None));
        assert_eq!(cf(&e), float(-2.5, None));
    }

    #[test]
    fn exp_sub() {
        let e = unop(UnOp::Exp, binop(float(2.5, None), BinOp::Sub, float(2.5, None), None));
        assert_eq!(cf(&e), float(1.0, None));
    }

    #[test]
    fn int_equality() {
        let e = binop(int(2, None), BinOp::Eq, int(3, None), Some(bool_ty()));
        assert_eq!(cf(&e), bool_expr(false));
    }

    #[test]
    fn float_lt() {
        let e = binop(float(1.5, None), BinOp::Lt, float(2.5, None), Some(bool_ty()));
        assert_eq!(cf(&e), bool_expr(true));
    }

    #[test]
    fn div_by_zero_untouched() {
        let e = binop(int(3, None), BinOp::Div, int(0, None), None);
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn log_zero_untouched() {
        let e = unop(UnOp::Log, float(0.0, None));
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn invalid_types_untouched() {
        let e = binop(int(2, None), BinOp::Add, float(3.0, None), None);
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn convert_float32_inf() {
        let e = Expr::Convert {
            e: Box::new(float(f64::INFINITY, Some(ElemSize::F32))),
            ty: scalar(ElemSize::F32)
        };
        assert_eq!(cf(&e), float(f64::INFINITY, Some(ElemSize::F32)));
    }

    #[test]
    fn convert_float32_neginf() {
        let e = Expr::Convert {
            e: Box::new(float(-f64::INFINITY, Some(ElemSize::F32))),
            ty: scalar(ElemSize::F32)
        };
        assert_eq!(cf(&e), float(-f64::INFINITY, Some(ElemSize::F32)));
    }

    #[test]
    fn complicated_constant_fold_not_supported() {
        // Could be simplified to 2x + 2
        let x = var("x", scalar(ElemSize::I64));
        let e = binop(
            binop(x.clone(), BinOp::Add, int(1, None), None),
            BinOp::Mul, int(2, None), None
        );
        assert_eq!(cf(&e), e);
    }

    fn test_identity(op: BinOp, id: Expr, rhs: bool) {
        let x = var("x", id.get_type().clone());
        let e = if rhs {
            binop(x.clone(), op, id, None)
        } else {
            binop(id, op, x.clone(), None)
        };
        assert_eq!(cf(&e), x);
    }

    #[test]
    fn add_identity() {
        test_identity(BinOp::Add, float(0.0, None), true);
        test_identity(BinOp::Add, float(0.0, None), false);
        test_identity(BinOp::Add, int(0, None), true);
        test_identity(BinOp::Add, int(0, None), false);
    }

    #[test]
    fn sub_identity_rhs() {
        test_identity(BinOp::Sub, int(0, None), true);
    }

    #[test]
    #[should_panic]
    fn sub_identity_lhs() {
        test_identity(BinOp::Sub, int(0, None), false);
    }

    #[test]
    fn mul_identity() {
        test_identity(BinOp::Mul, float(1.0, None), true);
        test_identity(BinOp::Mul, float(1.0, None), false);
        test_identity(BinOp::Mul, int(1, None), true);
        test_identity(BinOp::Mul, int(1, None), false);
    }

    #[test]
    fn div_identity() {
        test_identity(BinOp::Div, float(1.0, None), true);
        test_identity(BinOp::Div, int(1, None), true);
    }
}
