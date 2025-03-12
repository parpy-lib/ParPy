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

    fn int_expr(v: i64, ty: Type, i: Info) -> Expr {
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

    fn get_int_value(&self) -> Option<i64> {
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
        *self == Type::Boolean
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Scalar {sz} if sz.is_signed_integer() => true,
            _ => false
        }
    }

    fn is_float(&self) -> bool {
        match self {
            Type::Scalar {sz} if sz.is_floating_point() => true,
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
        Expr::Ternary {..} | Expr::StructFieldAccess {..} | Expr::ArrayAccess {..} |
        Expr::Struct {..} | Expr::ShflXorSync {..} | Expr::ThreadIdx {..} |
        Expr::BlockIdx {..} => e.smap(fold_expr)
    }
}

fn fold_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::If {cond, thn, els} => {
            match fold_expr(cond) {
                Expr::Bool {v, ..} => {
                    let body = if v {
                        thn.smap(fold_stmt)
                    } else {
                        els.smap(fold_stmt)
                    };
                    Stmt::Scope {body}
                },
                // Integer values are allowed in conditions to be consistent with how it works in
                // Python and in CUDA C++.
                Expr::Int {v, ..} => {
                    let body = if v == 0 {
                        els.smap(fold_stmt)
                    } else {
                        thn.smap(fold_stmt)
                    };
                    Stmt::Scope {body}
                },
                cond => {
                    let thn = thn.smap(fold_stmt);
                    let els = els.smap(fold_stmt);
                    Stmt::If {cond, thn, els}
                }
            }
        },
        Stmt::For {body, ..} if body.is_empty() => {
            Stmt::Scope {body}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::AllocShared {..} |
        Stmt::For {..} | Stmt::While {..} | Stmt::Syncthreads {} |
        Stmt::Dim3Definition {..} | Stmt::KernelLaunch {..} |
        Stmt::MallocAsync {..} | Stmt::FreeAsync {..} | Stmt::Scope {..} => {
            s.smap(fold_stmt).smap(fold_expr)
        }
    }
}

fn fold_top(top: Top) -> Top {
    top.smap(fold_stmt)
}

pub fn fold(ast: Ast) -> Ast {
    ast.into_iter().map(fold_top).collect::<Ast>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::info::*;
    use crate::utils::name::*;

    fn var(id: &str) -> Expr {
        Expr::Var {id: Name::sym_str(id), ty: Type::Boolean, i: Info::default()}
    }

    fn bool_expr(v: bool) -> Expr {
        Expr::Bool {v, ty: Type::Boolean, i: Info::default()}
    }

    fn int_with_ty(v: i64, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(Type::Scalar {sz: ElemSize::I64});
        Expr::Int {v, ty, i: Info::default()}
    }

    fn int(v: i64) -> Expr {
        int_with_ty(v, None)
    }

    fn float_with_ty(v: f64, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(Type::Scalar {sz: ElemSize::F64});
        Expr::Float {v, ty, i: Info::default()}
    }

    fn float(v: f64) -> Expr {
        float_with_ty(v, None)
    }

    fn unop(op: UnOp, arg: Expr) -> Expr {
        let ty = arg.get_type().clone();
        let i = arg.get_info();
        Expr::mk_unop(op, arg, ty, i)
    }

    fn binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Option<Type>) -> Expr {
        let ty = match ty {
            Some(ty) => ty,
            None => lhs.get_type().clone()
        };
        let i = lhs.get_info();
        Expr::mk_binop(lhs, op, rhs, ty, i)
    }

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
        let e = binop(int(2), BinOp::Eq, int(3), Some(Type::Boolean));
        assert_eq!(cf(&e), bool_expr(false));
    }

    #[test]
    fn float_lt() {
        let e = binop(float(1.5), BinOp::Lt, float(2.5), Some(Type::Boolean));
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
        let float32 = Type::Scalar {sz: ElemSize::F32};
        let e = Expr::Convert {
            e: Box::new(float(f64::INFINITY)),
            ty: float32.clone()
        };
        assert_eq!(cf(&e), float_with_ty(f64::INFINITY, Some(float32)));
    }

    #[test]
    fn convert_float32_neginf() {
        let float32 = Type::Scalar {sz: ElemSize::F32};
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

    fn if_stmt(cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        Stmt::If {cond, thn, els}
    }

    fn assign(dst: Expr, expr: Expr) -> Stmt {
        Stmt::Assign {dst, expr}
    }

    fn cfs(stmt: &Stmt) -> Stmt {
        fold_stmt(stmt.clone())
    }

    #[test]
    fn eliminate_if_false() {
        let thn = vec![assign(var("x"), int(1))];
        let els = vec![assign(var("x"), int(2))];
        let s = if_stmt(
            binop(int(1), BinOp::Eq, int(2), Some(Type::Boolean)),
            thn.clone(), els.clone()
        );
        assert_eq!(cfs(&s), Stmt::Scope {body: els});
    }
}
