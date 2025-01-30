use super::ast::*;
use crate::utils::info::Info;
use crate::utils::smap::SMapAccum;

fn constant_fold_unop(
    op: UnOp,
    arg: Expr,
    ty: Type,
    i: Info
) -> Expr {
    let reconstruct = |op, arg, ty, i| {
        Expr::UnOp {op, arg: Box::new(arg), ty, i}
    };
    let arg = fold_expr(arg);
    match op {
        UnOp::Sub => match arg {
            Expr::Int {v, ..} => Expr::Int {v: -v, ty, i},
            Expr::Float {v, ..} => Expr::Float {v: -v, ty, i},
            _ => reconstruct(op, arg, ty, i)
        },
        UnOp::Exp => match arg {
            Expr::Float {v, ..} => Expr::Float {v: v.exp(), ty, i},
            _ => reconstruct(op, arg, ty, i)
        },
        UnOp::Log => match arg {
            Expr::Float {v, ..} if v > 0.0 => Expr::Float {v: v.ln(), ty, i},
            _ => reconstruct(op, arg, ty, i)
        }
    }
}

fn apply_bool_op(
    op: BinOp,
    lhs: bool,
    rhs: bool,
    ty: Type,
    i: Info
) -> Expr {
    let bool_expr = |v| Expr::Bool {v, ty: Type::Boolean, i: i.clone()};
    let reconstruct = |lhs, op, rhs| {
        let lhs = Box::new(bool_expr(lhs));
        let rhs = Box::new(bool_expr(rhs));
        Expr::BinOp {lhs, op, rhs, ty: ty.clone(), i: i.clone()}
    };
    match op {
        BinOp::BoolAnd => bool_expr(lhs && rhs),
        _ => reconstruct(lhs, op, rhs)
    }
}

fn is_bool_neutral_elem(op: &BinOp, v: bool) -> bool {
    match op {
        BinOp::BoolAnd => v == true,
        _ => false
    }
}

fn apply_int_op(
    op: BinOp,
    lhs: i64,
    rhs: i64,
    ty: Type,
    i: Info
) -> Expr {
    let int_expr = |v| Expr::Int {v, ty: ty.clone(), i: i.clone()};
    let reconstruct = |lhs, op, rhs| {
        let lhs = Box::new(int_expr(lhs));
        let rhs = Box::new(int_expr(rhs));
        Expr::BinOp {lhs, op, rhs, ty: ty.clone(), i: i.clone()}
    };
    let bool_expr = |v| Expr::Bool {v, ty: Type::Boolean, i: i.clone()};
    match op {
        BinOp::Add => int_expr(lhs + rhs),
        BinOp::Sub => int_expr(lhs - rhs),
        BinOp::Mul => int_expr(lhs * rhs),
        BinOp::Div if rhs != 0 => int_expr(lhs / rhs),
        BinOp::Rem if rhs != 0 => int_expr(lhs % rhs),
        BinOp::BitAnd => int_expr(lhs & rhs),
        BinOp::Eq => bool_expr(lhs == rhs),
        BinOp::Neq => bool_expr(lhs != rhs),
        BinOp::Lt => bool_expr(lhs < rhs),
        BinOp::Gt => bool_expr(lhs > rhs),
        BinOp::Max => int_expr(i64::max(lhs, rhs)),
        BinOp::Min => int_expr(i64::min(lhs, rhs)),
        _ => reconstruct(lhs, op, rhs)
    }
}

fn is_int_neutral_elem(op: &BinOp, v: i64, rhs: bool) -> bool {
    match op {
        BinOp::Add => v == 0,
        BinOp::Sub if rhs => v == 0,
        BinOp::Mul => v == 1,
        BinOp::Div if rhs => v == 1,
        BinOp::BitAnd => v == 0,
        // NOTE: We can simplify 'x != 0' as 'x' in C++
        BinOp::Neq if rhs => v == 0,
        BinOp::Max => v == i64::MIN,
        BinOp::Min => v == i64::MAX,
        _ => false
    }
}

fn apply_float_op(
    op: BinOp,
    lhs: f64,
    rhs: f64,
    ty: Type,
    i: Info
) -> Expr {
    let float_expr = |v| Expr::Float {v, ty: ty.clone(), i: i.clone()};
    let reconstruct = |lhs, op, rhs| {
        let lhs = Box::new(float_expr(lhs));
        let rhs = Box::new(float_expr(rhs));
        Expr::BinOp {lhs, op, rhs, ty: ty.clone(), i: i.clone()}
    };
    let bool_expr = |v| Expr::Bool {v, ty: Type::Boolean, i: i.clone()};
    match op {
        BinOp::Add => float_expr(lhs + rhs),
        BinOp::Sub => float_expr(lhs - rhs),
        BinOp::Mul => float_expr(lhs * rhs),
        BinOp::Div => float_expr(lhs / rhs),
        BinOp::Eq => bool_expr(lhs == rhs),
        BinOp::Neq => bool_expr(lhs != rhs),
        BinOp::Lt => bool_expr(lhs < rhs),
        BinOp::Gt => bool_expr(lhs > rhs),
        BinOp::Max => float_expr(f64::max(lhs, rhs)),
        BinOp::Min => float_expr(f64::min(lhs, rhs)),
        _ => reconstruct(lhs, op, rhs)
    }
}

fn is_float_neutral_elem(op: &BinOp, v: f64, rhs: bool) -> bool {
    match op {
        BinOp::Add => v == 0.0,
        BinOp::Sub if rhs => v == 0.0,
        BinOp::Mul => v == 1.0,
        BinOp::Div if rhs => v == 1.0,
        _ => false
    }
}

fn constant_fold_binop(
    lhs: Expr,
    op: BinOp,
    rhs: Expr,
    ty: Type,
    i: Info
) -> Expr {
    let reconstruct = |lhs, op, rhs, ty, i| {
        Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
    };
    let lhs = fold_expr(lhs);
    let rhs = fold_expr(rhs);
    match (lhs, rhs) {
        (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) =>
            apply_bool_op(op, lv, rv, ty, i),
        (lhs, Expr::Bool {v: rv, ..}) if is_bool_neutral_elem(&op, rv) => lhs,
        (Expr::Bool {v: lv, ..}, rhs) if is_bool_neutral_elem(&op, lv) => rhs,
        (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) =>
            apply_int_op(op, lv, rv, ty, i),
        (lhs, Expr::Int {v: rv, ..}) if is_int_neutral_elem(&op, rv, true) => lhs,
        (Expr::Int {v: lv, ..}, rhs) if is_int_neutral_elem(&op, lv, false) => rhs,
        (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) =>
            apply_float_op(op, lv, rv, ty, i),
        (lhs, Expr::Float {v: rv, ..}) if is_float_neutral_elem(&op, rv, true) => lhs,
        (Expr::Float {v: lv, ..}, rhs) if is_float_neutral_elem(&op, lv, false) => rhs,
        (lhs, rhs) => reconstruct(lhs, op, rhs, ty, i)
    }
}

fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::UnOp {op, arg, ty, i} => constant_fold_unop(op, *arg, ty, i),
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            constant_fold_binop(*lhs, op, *rhs, ty, i)
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
        Expr::StructFieldAccess {..} | Expr::ArrayAccess {..} |
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
                cond => {
                    let thn = thn.smap(fold_stmt);
                    let els = els.smap(fold_stmt);
                    Stmt::If {cond, thn, els}
                },
            }
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::AllocShared {..} |
        Stmt::For {..} | Stmt::Syncthreads {} | Stmt::Dim3Definition {..} |
        Stmt::KernelLaunch {..} | Stmt::Scope {..} => {
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
        Expr::UnOp {op, arg: Box::new(arg), ty, i}
    }

    fn binop(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        let ty = lhs.get_type().clone();
        let i = lhs.get_info();
        Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
    }

    fn cf(e: &Expr) -> Expr {
        fold_expr(e.clone())
    }

    #[test]
    fn add() {
        let e = binop(int(2), BinOp::Add, int(3));
        assert_eq!(cf(&e), int(5));
    }

    #[test]
    fn nested_sub_mul() {
        let e = binop(binop(int(4), BinOp::Sub, int(1)), BinOp::Mul, int(2));
        assert_eq!(cf(&e), int(6));
    }

    #[test]
    fn float_sub() {
        let e = binop(float(2.5), BinOp::Sub, float(1.5));
        assert_eq!(cf(&e), float(1.0));
    }

    #[test]
    fn unary_sub() {
        let e = unop(UnOp::Sub, float(2.5));
        assert_eq!(cf(&e), float(-2.5));
    }

    #[test]
    fn exp_sub() {
        let e = unop(UnOp::Exp, binop(float(2.5), BinOp::Sub, float(2.5)));
        assert_eq!(cf(&e), float(1.0));
    }

    #[test]
    fn int_equality() {
        let e = binop(int(2), BinOp::Eq, int(3));
        assert_eq!(cf(&e), bool_expr(false));
    }

    #[test]
    fn float_lt() {
        let e = binop(float(1.5), BinOp::Lt, float(2.5));
        assert_eq!(cf(&e), bool_expr(true));
    }

    #[test]
    fn div_by_zero_untouched() {
        let e = binop(int(3), BinOp::Div, int(0));
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn log_zero_untouched() {
        let e = unop(UnOp::Log, float(0.0));
        assert_eq!(cf(&e), e);
    }

    #[test]
    fn invalid_types_untouched() {
        let e = binop(int(2), BinOp::Add, float(3.0));
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
        let e = binop(binop(x.clone(), BinOp::Add, int(1)), BinOp::Mul, int(2));
        assert_eq!(cf(&e), e);
    }

    fn test_identity(op: BinOp, id: Expr, rhs: bool) {
        let x = var("x");
        let e = if rhs {
            binop(x.clone(), op, id)
        } else {
            binop(id, op, x.clone())
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
        let s = if_stmt(binop(int(1), BinOp::Eq, int(2)), thn.clone(), els.clone());
        assert_eq!(cfs(&s), Stmt::Scope {body: els});
    }
}
