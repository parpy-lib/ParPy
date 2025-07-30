use super::ast::*;
use crate::utils::constant_fold::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
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
                Expr::Float {v, i, ..} if v.is_infinite() => Expr::Float {v, ty, i},
                _ => Expr::Convert {e: Box::new(e), ty}
            }
        },
        Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::IfExpr {..} | Expr::StructFieldAccess {..} |
        Expr::ArrayAccess {..} | Expr::Call {..} | Expr::Struct {..} |
        Expr::ThreadIdx {..} | Expr::BlockIdx {..} => e.smap(fold_expr)
    }
}

fn replace_dim_indices_with_zero(e: Expr) -> Expr {
    match e {
        Expr::ThreadIdx {ty, i, ..} => Expr::Int {v: 0, ty, i},
        Expr::BlockIdx {ty, i, ..} => Expr::Int {v: 0, ty, i},
        _ => e.smap(replace_dim_indices_with_zero)
    }
}

fn is_zero_value(init: &Expr) -> bool {
    let init = fold_expr(replace_dim_indices_with_zero(init.clone()));
    match init {
        Expr::Int {v, ..} if v == 0 => true,
        _ => false
    }
}

fn cond_upper_bound(var: &Name, cond: &Expr) -> Option<i128> {
    match cond {
        Expr::BinOp {lhs, op: BinOp::Lt, rhs, ..} => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Var {id, ..}, Expr::Int {v, ..}) if id == var => Some(*v),
                _ => None
            }
        },
        _ => None
    }
}

fn incr_rhs(var: &Name, incr: &Expr) -> Option<i128> {
    match incr {
        Expr::BinOp {lhs, op: BinOp::Add, rhs, ..} => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Var {id, ..}, Expr::Int {v, ..}) if id == var => Some(*v),
                _ => None
            }
        },
        _ => None
    }
}

// If every iteration of a for-loop runs on a distinct thread (with a unique thread and block
// index) and each thread runs the for-loop exactly one iteration, we can eliminate the for-loop to
// improve readability of the generated code.
fn loop_runs_once(var: &Name, init: &Expr, cond: &Expr, incr: &Expr) -> bool {
    if is_zero_value(init) {
        match (cond_upper_bound(var, cond), incr_rhs(var, incr)) {
            (Some(l), Some(r)) if l == r => true,
            _ => false
        }
    } else {
        false
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
        Stmt::For {var_ty, var, init, cond, incr, body, i} => {
            let init = fold_expr(init);
            let cond = fold_expr(cond);
            let incr = fold_expr(incr);
            if loop_runs_once(&var, &init, &cond, &incr) {
                acc.push(Stmt::Definition {ty: var_ty, id: var, expr: init, i});
                body.sfold_owned(acc, fold_stmt_acc)
            } else {
                let body = fold_stmts(body);
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i});
                acc
            }
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
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
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
        Top::KernelFunDef {attrs, id, params, body} => {
            let body = fold_stmts(body);
            Top::KernelFunDef {attrs, id, params, body}
        },
        Top::FunDef {ret_ty, id, params, body, target} => {
            let body = fold_stmts(body);
            Top::FunDef {ret_ty, id, params, body, target}
        },
        Top::StructDef {..} => top
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

    #[test]
    fn replace_thread_index_with_zero() {
        let e = Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()};
        assert_eq!(replace_dim_indices_with_zero(e), int(0, Some(ElemSize::I64)));
    }

    #[test]
    fn replace_block_index_with_zero() {
        let e = Expr::BlockIdx {dim: Dim::X, ty: scalar(ElemSize::I16), i: i()};
        assert_eq!(replace_dim_indices_with_zero(e), int(0, Some(ElemSize::I16)));
    }

    #[test]
    fn is_zero_value_zero_expr() {
        assert!(is_zero_value(&int(0, None)));
    }

    #[test]
    fn is_zero_value_thread_index() {
        let e = Expr::ThreadIdx {dim: Dim::Y, ty: scalar(ElemSize::I64), i: i()};
        assert!(is_zero_value(&e));
    }

    #[test]
    fn is_zero_value_non_zero_int() {
        assert!(!is_zero_value(&int(1, None)));
    }

    #[test]
    fn cond_upper_bound_binop_lt() {
        let cond = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Lt,
            int(10, Some(ElemSize::I64)),
            scalar(ElemSize::Bool)
        );
        assert_eq!(cond_upper_bound(&id("x"), &cond), Some(10));
    }

    #[test]
    fn cond_upper_bound_wrong_binop_operands() {
        let cond = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Gt,
            var("y", scalar(ElemSize::I64)),
            scalar(ElemSize::Bool)
        );
        assert_eq!(cond_upper_bound(&id("x"), &cond), None);
    }

    #[test]
    fn cond_upper_bound_wrong_form() {
        let cond = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Gt,
            int(0, Some(ElemSize::I64)),
            scalar(ElemSize::Bool)
        );
        assert_eq!(cond_upper_bound(&id("x"), &cond), None);
    }

    #[test]
    fn incr_rhs_binop_add() {
        let incr = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Add,
            int(1, Some(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        assert_eq!(incr_rhs(&id("x"), &incr), Some(1));
    }

    #[test]
    fn incr_rhs_wrong_operands() {
        let incr = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Add,
            var("y", scalar(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        assert_eq!(incr_rhs(&id("x"), &incr), None);
    }

    #[test]
    fn incr_rhs_invalid_form() {
        let incr = binop(
            var("x", scalar(ElemSize::I64)),
            BinOp::Sub,
            int(1, Some(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        assert_eq!(incr_rhs(&id("x"), &incr), None);
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
    fn loop_runs_once_true() {
        let (init, cond, incr) = _loop();
        assert!(loop_runs_once(&id("x"), &init, &cond, &incr));
    }

    #[test]
    fn loop_runs_once_non_matching_cond_and_incr() {
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
            int(1, Some(ElemSize::I64)),
            scalar(ElemSize::I64)
        );
        assert!(!loop_runs_once(&id("x"), &init, &cond, &incr));
    }

    #[test]
    fn loop_runs_once_non_zero_init() {
        let init = int(1, None);
        assert!(!loop_runs_once(&id("x"), &init, &init, &init));
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
    fn fold_for_loop_stmt_running_once() {
        let (init, cond, incr) = _loop();
        let s = Stmt::For {
            var_ty: scalar(ElemSize::I64), var: id("x"),
            init: init.clone(), cond, incr,
            body: vec![], i: i()
        };
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::I64), id: id("x"), expr: init, i: i()
            }
        ];
        assert_eq!(fold_stmt_acc(vec![], s), expected);
    }

    #[test]
    fn fold_if_cond_stmt() {
        let s = if_stmt(bool_expr(true), vec![], vec![]);
        assert_eq!(fold_stmt_acc(vec![], s), vec![]);
    }
}
