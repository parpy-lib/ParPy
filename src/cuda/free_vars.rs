use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

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
            Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::StructFieldAccess {..} |
            Expr::ArrayAccess {..} | Expr::Struct {..} | Expr::Convert {..} |
            Expr::ShflXorSync {..} | Expr::ThreadIdx {..} |
            Expr::BlockIdx {..} => {
                self.sfold(|env, e| e.fv(env), env)
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
            Stmt::AllocShared {ty, id, ..} => {
                env.bound.insert(id.clone(), ty.clone());
                env
            },
            Stmt::For {var, init, cond, body, ..} => {
                env.bound.insert(var.clone(), init.get_type().clone());
                let env = init.fv(env);
                let env = cond.fv(env);
                body.fv(env)
            },
            Stmt::Dim3Definition {id, ..} => {
                env.bound.insert(id.clone(), Type::Void);
                env
            },
            Stmt::Assign {..} | Stmt::If {..} | Stmt::Syncthreads {..} |
            Stmt::KernelLaunch {..} | Stmt::Scope {..} => {
                self.sfold(|env, s| s.fv(env), env)
            }
        }
    }
}

impl<U: FreeVariables<Type>> FreeVariables<Type> for Vec<U> {
    fn fv(&self, env: FVEnv<Type>) -> FVEnv<Type> {
        self.iter()
            .fold(env, |env, t| t.fv(env))
    }
}
