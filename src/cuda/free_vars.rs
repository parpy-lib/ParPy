use super::ast::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub struct FVEnv<T> {
    pub bound: BTreeMap<Name, T>,
    pub free: BTreeMap<Name, T>
}

impl<T> Default for FVEnv<T> {
    fn default() -> Self {
        FVEnv {bound: BTreeMap::new(), free: BTreeMap::new()}
    }
}

pub trait FreeVariables<T> {
    fn fv(&self, env: FVEnv<T>) -> FVEnv<T>;

    fn free_variables(&self) -> BTreeMap<Name, T> {
        let env = self.fv(FVEnv::default());
        env.free
    }
}

impl FreeVariables<Type> for Expr {
    fn fv(&self, mut env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Expr::Var {id, ty, ..} => {
                if !env.bound.contains_key(&id) {
                    env.free.insert(id.clone(), ty.clone());
                };
                env
            },
            Expr::Int {..} | Expr::Float {..} => env,
            Expr::UnOp {arg, ..} => arg.fv(env),
            Expr::BinOp {lhs, rhs, ..} => {
                let env = lhs.fv(env);
                rhs.fv(env)
            },
            Expr::StructFieldAccess {target, ..} => target.fv(env),
            Expr::ArrayAccess {target, idx, ..} => {
                let env = target.fv(env);
                idx.fv(env)
            },
            Expr::Struct {fields, ..} => {
                fields.iter().fold(env, |env, (_, e)| e.fv(env))
            },
            Expr::Convert {e, ..} => {
                e.fv(env)
            },
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => env,
            Expr::Inf {..} => env,
            Expr::Exp {arg, ..} | Expr::Log {arg, ..} => arg.fv(env),
            Expr::Max {lhs, rhs, ..} | Expr::Min {lhs, rhs, ..} => {
                let env = lhs.fv(env);
                rhs.fv(env)
            }
        }
    }
}

impl FreeVariables<Type> for Stmt {
    fn fv(&self, mut env: FVEnv<Type>) -> FVEnv<Type> {
        match self {
            Stmt::Definition {id, expr, ..} => {
                let mut env = expr.fv(env);
                env.bound.insert(id.clone(), expr.get_type().clone());
                env
            },
            Stmt::Assign {dst, expr, ..} => {
                let env = dst.fv(env);
                expr.fv(env)
            },
            Stmt::For {var, init, cond, body, ..} => {
                env.bound.insert(var.clone(), init.get_type().clone());
                let env = init.fv(env);
                let env = cond.fv(env);
                body.fv(env)
            },
            Stmt::If {cond, thn, els, ..} => {
                let env = cond.fv(env);
                let env = thn.fv(env);
                els.fv(env)
            },
            Stmt::Syncthreads {..} => env,
            Stmt::KernelLaunch {args, ..} => {
                args.fv(env)
            },
        }
    }
}

impl<U: FreeVariables<Type>> FreeVariables<Type> for Vec<U> {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        self.iter()
            .fold(env, |env, t| t.fv(env))
    }
}
