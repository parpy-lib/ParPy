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
            Type::Tensor {sz, shape} => shape.is_empty() && sz == &ElemSize::Bool,
            _ => false
        }
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Tensor {sz, shape} => shape.is_empty() && sz.is_integer(),
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
        _ => e.smap(fold_expr)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;

    fn bool_lit(v: bool) -> Expr {
        bool_expr(v, Some(ElemSize::Bool))
    }

    fn int_lit(v: i64) -> Expr {
        int(v, Some(ElemSize::I64))
    }

    fn uint_lit(v: i64) -> Expr {
        int(v, Some(ElemSize::U32))
    }

    fn float_lit(v: f64) -> Expr {
        float(v, Some(ElemSize::F64))
    }

    fn vec_ty(sz: ElemSize) -> Type {
        Type::Tensor {sz, shape: vec![10]}
    }

    #[test]
    fn get_bool_value() {
        assert_eq!(bool_lit(true).get_bool_value(), Some(true));
    }

    #[test]
    fn get_bool_value_from_int() {
        assert_eq!(int_lit(2).get_bool_value(), None);
    }

    #[test]
    fn get_int_value() {
        assert_eq!(int_lit(2).get_int_value(), Some(2));
    }

    #[test]
    fn get_int_value_from_float() {
        assert_eq!(float_lit(2.5).get_int_value(), None);
    }

    #[test]
    fn get_float_value() {
        assert_eq!(float_lit(2.5).get_float_value(), Some(2.5));
    }

    #[test]
    fn get_float_value_from_int() {
        assert_eq!(int_lit(2).get_float_value(), None);
    }

    #[test]
    fn literal_kind_bool() {
        assert_eq!(bool_lit(true).literal_kind(), Some(LitKind::Bool));
    }

    #[test]
    fn literal_kind_int() {
        assert_eq!(int_lit(2).literal_kind(), Some(LitKind::Int));
    }

    #[test]
    fn literal_kind_float() {
        assert_eq!(float_lit(2.5).literal_kind(), Some(LitKind::Float));
    }

    #[test]
    fn literal_kind_var() {
        assert_eq!(var("x", Type::Unknown).literal_kind(), None);
    }

    #[test]
    fn is_bool_type() {
        assert!(scalar(ElemSize::Bool).is_bool());
    }

    #[test]
    fn is_bool_type_int_fails() {
        assert!(!scalar(ElemSize::I64).is_bool());
    }

    #[test]
    fn is_int_type_signed_int() {
        assert!(scalar(ElemSize::I64).is_int());
    }

    #[test]
    fn is_int_type_unsigned_int() {
        assert!(scalar(ElemSize::U32).is_int());
    }

    #[test]
    fn is_float_type() {
        assert!(scalar(ElemSize::F32).is_float());
    }

    #[test]
    fn is_bool_type_vector_fails() {
        assert!(!vec_ty(ElemSize::Bool).is_bool());
    }

    #[test]
    fn is_int_type_vector_fails() {
        assert!(!vec_ty(ElemSize::I32).is_int());
    }

    #[test]
    fn is_float_type_vector_fails() {
        assert!(!vec_ty(ElemSize::F32).is_float());
    }

    #[test]
    fn test_fold_unary_sub() {
        let e = unop(UnOp::Sub, int_lit(1));
        assert_eq!(fold_expr(e), int_lit(-1));
    }

    #[test]
    fn test_fold_add() {
        let e = binop(int_lit(5), BinOp::Add, int_lit(2), scalar(ElemSize::I64));
        assert_eq!(fold_expr(e), int_lit(7));
    }

    #[test]
    fn test_fold_nested_int_add() {
        let e = binop(
            int_lit(2),
            BinOp::Add,
            binop(int_lit(3), BinOp::Add, int_lit(4), scalar(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        assert_eq!(fold_expr(e), int_lit(9));
    }

    #[test]
    fn test_fold_mul() {
        let e = binop(int_lit(5), BinOp::Mul, int_lit(2), scalar(ElemSize::I64));
        assert_eq!(fold_expr(e), int_lit(10));
    }

    #[test]
    fn test_fold_int_div() {
        let e = binop(int_lit(5), BinOp::FloorDiv, int_lit(2), scalar(ElemSize::I64));
        assert_eq!(fold_expr(e), int_lit(2));
    }

    #[test]
    fn test_fold_unsigned_int_div() {
        let e = binop(uint_lit(5), BinOp::FloorDiv, uint_lit(2), scalar(ElemSize::U32));
        assert_eq!(fold_expr(e), uint_lit(2));
    }

    #[test]
    fn test_fold_float_div() {
        let e = binop(float_lit(5.0), BinOp::Div, float_lit(2.0), scalar(ElemSize::F32));
        assert_eq!(fold_expr(e), float_lit(2.5));
    }

    #[test]
    fn test_fold_int_bool_op() {
        let e = binop(int_lit(2), BinOp::Geq, int_lit(2), scalar(ElemSize::Bool));
        assert_eq!(fold_expr(e), bool_lit(true));
    }

    #[test]
    fn test_fold_float_bool_op() {
        let e = binop(float_lit(2.5), BinOp::Lt, float_lit(2.0), scalar(ElemSize::Bool));
        assert_eq!(fold_expr(e), bool_lit(false));
    }

    #[test]
    fn test_fold_nested_float_ops() {
        let e = binop(
            float_lit(1.5),
            BinOp::Mul,
            binop(float_lit(2.5), BinOp::Add, float_lit(3.5), scalar(ElemSize::F32)),
            scalar(ElemSize::F32)
        );
        assert_eq!(fold_expr(e), float_lit(9.0));
    }

    #[test]
    fn test_fold_in_convert() {
        let e = Expr::Convert {
            e: Box::new(binop(int_lit(1), BinOp::Add, int_lit(2), scalar(ElemSize::I64))),
            ty: scalar(ElemSize::I16)
        };
        let expected = Expr::Convert {
            e: Box::new(int_lit(3)),
            ty: scalar(ElemSize::I16)
        };
        assert_eq!(fold_expr(e), expected);
    }

    #[test]
    fn test_fold_convert_inf_float() {
        let e = Expr::Convert {
            e: Box::new(float_lit(f64::INFINITY)),
            ty: scalar(ElemSize::F16)
        };
        assert_eq!(fold_expr(e), float(f64::INFINITY, Some(ElemSize::F16)));
    }
}
