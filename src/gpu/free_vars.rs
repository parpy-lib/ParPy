use super::ast::*;
use crate::py::ast as py_ast;
use crate::utils::ast::ExprType;
use crate::utils::free_vars::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

impl FreeVars<py_ast::Type> for Expr {
    fn fv(&self, env: FVEnv<py_ast::Type>) -> FVEnv<py_ast::Type> {
        match self {
            Expr::Var {id, ..} => use_variable(env, &id, &py_ast::Type::Unknown),
            Expr::PyCallback {args, ..} => {
                args.sfold(env, |env, e: &py_ast::Expr| e.fv(env))
            },
            Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::Assign {..} |
            Expr::IfExpr {..} | Expr::ArrayAccess {..} | Expr::Call {..} |
            Expr::Convert {..} | Expr::ThreadIdx {..} | Expr::BlockIdx {..} => {
                self.sfold(env, |env, e| e.fv(env))
            }
        }
    }
}

impl FreeVars<py_ast::Type> for Stmt {
    fn fv(&self, env: FVEnv<py_ast::Type>) -> FVEnv<py_ast::Type> {
        match self {
            Stmt::Definition {id, expr, ..} => {
                let env = expr.fv(env);
                bind_variable(env, &id, &py_ast::Type::Unknown)
            },
            Stmt::For {var, init, cond, incr, body, ..} |
            Stmt::ParallelReduction {var, init, cond, incr, body, ..} => {
                let env = init.fv(env);
                let env = bind_variable(env, &var, &py_ast::Type::Unknown);
                let env = cond.fv(env);
                let env = incr.fv(env);
                body.sfold(env, |env, s| s.fv(env))
            },
            Stmt::AllocShared {id, ..} => {
                bind_variable(env, &id, &py_ast::Type::Unknown)
            },
            Stmt::If {..} | Stmt::While {..} | Stmt::Return {..} |
            Stmt::Scope {..} | Stmt::Expr {..} | Stmt::Synchronize {..} |
            Stmt::WarpReduce {..} | Stmt::ClusterReduce {..} |
            Stmt::KernelLaunch {..} | Stmt::AllocDevice {..} |
            Stmt::FreeDevice {..} | Stmt::CopyMemory {..} => {
                let env = self.sfold(env, |env, s: &Stmt| s.fv(env));
                self.sfold(env, |env, e: &Expr| e.fv(env))
            }
        }
    }
}

impl FreeVars<Type> for Expr {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Expr::Var {id, ty, ..} => use_variable(env, &id, &ty),
            Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::Assign {..} |
            Expr::IfExpr {..} | Expr::ArrayAccess {..} | Expr::Call {..} |
            Expr::PyCallback {..} | Expr::Convert {..} | Expr::ThreadIdx {..} |
            Expr::BlockIdx {..} => {
                self.sfold(env, |env, e| e.fv(env))
            }
        }
    }
}

impl FreeVars<Type> for Stmt {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Stmt::Definition {id, expr, ..} => {
                let env = expr.fv(env);
                bind_variable(env, &id, expr.get_type())
            },
            Stmt::For {var, init, cond, incr, body, ..} |
            Stmt::ParallelReduction {var, init, cond, incr, body, ..} => {
                let env = init.fv(env);
                let env = bind_variable(env, &var, init.get_type());
                let env = cond.fv(env);
                let env = incr.fv(env);
                body.sfold(env, |env, s| s.fv(env))
            },
            Stmt::AllocShared {elem_ty, id, ..} => {
                bind_variable(env, &id, &elem_ty)
            },
            Stmt::If {..} | Stmt::While {..} | Stmt::Return {..} | Stmt::Scope {..} |
            Stmt::Expr {..} | Stmt::Synchronize {..} | Stmt::WarpReduce {..} |
            Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} |
            Stmt::AllocDevice {..} | Stmt::FreeDevice {..} |
            Stmt::CopyMemory {..} => {
                let env = self.sfold(env, |env, s: &Stmt| s.fv(env));
                self.sfold(env, |env, e: &Expr| e.fv(env))
            }
        }
    }
}

pub fn free_variables(s: &Vec<Stmt>) -> BTreeMap<Name, Type> {
    let env = s.sfold(FVEnv::default(), |env, s| s.fv(env));
    env.free
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;
    use crate::test::*;

    use std::fmt::Debug;

    fn mk_fv_env<T>(bound: Vec<(Name, T)>, free: Vec<(Name, T)>) -> FVEnv<T> {
        FVEnv {
            bound: bound.into_iter().collect::<BTreeMap<Name, T>>(),
            free: free.into_iter().collect::<BTreeMap<Name, T>>()
        }
    }

    fn assert_eq_fv_env<T: Debug + PartialEq>(
        env: FVEnv<T>,
        bound: Vec<(Name, T)>,
        free: Vec<(Name, T)>
    ) {
        let env_bound = env.bound.into_iter().collect::<Vec<(Name, T)>>();
        assert_eq!(env_bound, bound);
        let env_free = env.free.into_iter().collect::<Vec<(Name, T)>>();
        assert_eq!(env_free, free);
    }

    #[test]
    fn free_vars_unbound_var() {
        let ty = Type::Scalar {sz: ElemSize::I32};
        let e = Expr::Var {id: id("x"), ty: ty.clone(), i: i()};
        let env = e.fv(FVEnv::<Type>::default());
        assert_eq_fv_env(env, vec![], vec![(id("x"), ty)]);
    }

    #[test]
    fn free_vars_bound_var() {
        let ty = Type::Scalar {sz: ElemSize::I32};
        let e = Expr::Var {id: id("x"), ty: ty.clone(), i: i()};
        let env = mk_fv_env(vec![(id("x"), ty.clone())], vec![]);
        assert_eq_fv_env(e.fv(env), vec![(id("x"), ty)], vec![]);
    }

    #[test]
    fn free_vars_addition() {
        let ty = Type::Scalar {sz: ElemSize::I32};
        let e = Expr::BinOp {
            lhs: Box::new(Expr::Var {id: id("x"), ty: ty.clone(), i: i()}),
            op: BinOp::Add,
            rhs: Box::new(Expr::Var {id: id("y"), ty: ty.clone(), i: i()}),
            ty: ty.clone(),
            i: i()
        };
        let env = mk_fv_env(vec![], vec![]);
        assert_eq_fv_env(e.fv(env), vec![], vec![(id("x"), ty.clone()), (id("y"), ty)]);
    }

    #[test]
    fn free_vars_bound_in_def() {
        let ty = Type::Scalar {sz: ElemSize::I32};
        let s = Stmt::Definition {
            ty: ty.clone(),
            id: id("x"),
            expr: Expr::Int {v: 1, ty: ty.clone(), i: i()},
            i: i()
        };
        let env = mk_fv_env(vec![], vec![]);
        assert_eq_fv_env(s.fv(env), vec![(id("x"), ty)], vec![]);
    }
}
