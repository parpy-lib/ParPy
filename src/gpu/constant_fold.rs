use super::ast::*;
use crate::utils::constant_fold::*;
use crate::utils::info::Info;
use crate::utils::smap::*;

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
        *self == Type::Scalar {sz: ElemSize::Bool}
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Scalar {..} => true,
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
                Expr::Float {v, i, ..} if v.is_infinite() => Expr::Float {v, ty, i},
                _ => Expr::Convert {e: Box::new(e), ty}
            }
        },
        Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::Assign {..} | Expr::IfExpr {..} | Expr::ArrayAccess {..} |
        Expr::Call {..} | Expr::PyCallback {..} | Expr::ThreadIdx {..} |
        Expr::BlockIdx {..} => {
            e.smap(fold_expr)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum LitBoolValue { True, False, Unknown }

fn literal_bool_value(cond: &Expr) -> LitBoolValue {
    match cond {
        Expr::Bool {v, ..} if *v => LitBoolValue::True,
        Expr::Bool {v, ..} if !*v => LitBoolValue::False,
        Expr::Int {v, ..} if *v != 0 => LitBoolValue::True,
        Expr::Int {v, ..} if *v == 0 => LitBoolValue::False,
        _ => LitBoolValue::Unknown
    }
}

fn fold_stmt_acc(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
            let init = fold_expr(init);
            let cond = fold_expr(cond);
            let incr = fold_expr(incr);
            let body = fold_stmts(body);
            acc.push(Stmt::For {var_ty, var, init, cond, incr, body, unroll, i});
            acc
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = fold_expr(cond);
            match literal_bool_value(&cond) {
                LitBoolValue::True => thn.sfold_owned(acc, fold_stmt_acc),
                LitBoolValue::False => els.sfold_owned(acc, fold_stmt_acc),
                LitBoolValue::Unknown => {
                    let thn = fold_stmts(thn);
                    let els = fold_stmts(els);
                    acc.push(Stmt::If {cond, thn, els, i});
                    acc
                }
            }
        },
        Stmt::While {cond, body, i} => {
            let cond = fold_expr(cond);
            let body = fold_stmts(body);
            acc.push(Stmt::While {cond, body, i});
            acc
        },
        Stmt::Scope {body, ..} => body.sfold_owned(acc, fold_stmt_acc),
        Stmt::Definition {..} | Stmt::Return {..} | Stmt::Expr {..} |
        Stmt::ParallelReduction {..} | Stmt::Synchronize {..} | Stmt::WarpReduce {..} |
        Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} | Stmt::AllocDevice {..} |
        Stmt::AllocShared {..} | Stmt::FreeDevice {..} | Stmt::CopyMemory {..} => {
            acc.push(s.smap(fold_expr));
            acc
        }
    }
}

fn fold_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.sfold_owned(vec![], fold_stmt_acc)
}

fn fold_top(top: Top) -> Top {
    match top {
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = fold_stmts(body);
            Top::KernelFunDef {attrs, id, params, body, i}
        },
        Top::FunDef {ret_ty, id, params, body, target, i} => {
            let body = fold_stmts(body);
            Top::FunDef {ret_ty, id, params, body, target, i}
        },
        Top::ExtDecl {..} => top
    }
}

pub fn fold(ast: Ast) -> Ast {
    ast.into_iter().map(fold_top).collect::<Ast>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;
    use crate::test::*;
    use crate::utils::ast::*;

    fn cf(e: Expr) -> Expr {
        fold_expr(e)
    }

    fn uop(op: UnOp, arg: Expr) -> Expr {
        let ty = arg.get_type().clone();
        unop(op, arg, ty)
    }

    fn bop(l: Expr, op: BinOp, r: Expr) -> Expr {
        let ty = l.get_type().clone();
        binop(l, op, r, ty)
    }

    #[test]
    fn neg_unop() {
        assert_eq!(cf(uop(UnOp::Sub, int(-5, None))), int(5, None));
    }

    #[test]
    fn add_binop() {
        let e = bop(int(2, None), BinOp::Add, int(3, None));
        assert_eq!(cf(e), int(5, None));
    }

    #[test]
    fn nested_int_binary_ops() {
        let e = bop(
            bop(int(2, None), BinOp::Mul, int(3, None)),
            BinOp::Add,
            bop(int(7, None), BinOp::FloorDiv, int(2, None))
        );
        assert_eq!(cf(e), int(9, None));
    }

    #[test]
    fn float_sub() {
        let e = bop(float(2.5, None), BinOp::Sub, float(1.5, None));
        assert_eq!(cf(e), float(1.0, None));
    }

    #[test]
    fn convert_inf_float() {
        let e = Expr::Convert {
            e: Box::new(float(f64::INFINITY, None)),
            ty: scalar(ElemSize::F16)
        };
        assert_eq!(cf(e), Expr::Float {v: f64::INFINITY, ty: scalar(ElemSize::F16), i: i()});
    }

    #[test]
    fn convert_finite_float() {
        let e = Expr::Convert {
            e: Box::new(float(2.5, None)),
            ty: scalar(ElemSize::F16)
        };
        assert_eq!(cf(e.clone()), e);
    }

    fn _loop() -> (Expr, Expr, Expr) {
        let init = int(0, None);
        let cond = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Lt,
            int(10, Some(ElemSize::I64)),
            scalar(ElemSize::Bool)
        );
        let incr = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Add,
            int(10, Some(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        (init, cond, incr)
    }

    #[test]
    fn lit_bool_bool_true() {
        assert_eq!(literal_bool_value(&bool_expr(true)), LitBoolValue::True);
    }

    #[test]
    fn lit_bool_bool_false() {
        assert_eq!(literal_bool_value(&bool_expr(false)), LitBoolValue::False);
    }

    #[test]
    fn lit_bool_int_zero() {
        assert_eq!(literal_bool_value(&int(0, None)), LitBoolValue::False);
    }

    #[test]
    fn lit_bool_int_non_zero() {
        assert_eq!(literal_bool_value(&int(4, None)), LitBoolValue::True);
    }

    #[test]
    fn lit_bool_var_unknown() {
        let e = var("x", scalar(ElemSize::Bool));
        assert_eq!(literal_bool_value(&e), LitBoolValue::Unknown);
    }

    #[test]
    fn fold_if_cond_stmt() {
        let s = if_stmt(bool_expr(true), vec![], vec![]);
        assert_eq!(fold_stmt_acc(vec![], s), vec![]);
    }
}
