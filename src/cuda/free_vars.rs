use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

pub struct FVEnv {
    pub bound: BTreeMap<Name, Type>,
    pub free: BTreeMap<Name, Type>
}

impl Default for FVEnv {
    fn default() -> Self {
        FVEnv {bound: BTreeMap::new(), free: BTreeMap::new()}
    }
}

fn fv_expr(mut env: FVEnv, e: &Expr) -> FVEnv {
    match e {
        Expr::Var {id, ty, ..} => {
            if !env.bound.contains_key(&id) {
                env.free.insert(id.clone(), ty.clone());
            };
            env
        },
        Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::UnOp {..} | Expr::BinOp {..} | Expr::Ternary {..} |
        Expr::StructFieldAccess {..} | Expr::ArrayAccess {..} | Expr::Struct {..} |
        Expr::Convert {..} | Expr::ShflXorSync {..} | Expr::ThreadIdx {..} |
        Expr::BlockIdx {..} => e.sfold(env, fv_expr),
    }
}

fn fv_stmt(mut env: FVEnv, s: &Stmt) -> FVEnv {
    match s {
        Stmt::Definition {id, expr, ..} => {
            let mut env = fv_expr(env, expr);
            env.bound.insert(id.clone(), expr.get_type().clone());
            env
        },
        Stmt::AllocShared {ty, id, ..} => {
            env.bound.insert(id.clone(), ty.clone());
            env
        },
        Stmt::For {var, init, cond, incr, body, ..} => {
            env.bound.insert(var.clone(), init.get_type().clone());
            let env = fv_expr(env, init);
            let env = fv_expr(env, cond);
            let env = fv_expr(env, incr);
            body.iter().fold(env, fv_stmt)
        },
        Stmt::Dim3Definition {id, ..} => {
            env.bound.insert(id.clone(), Type::Void);
            env
        },
        Stmt::Assign {..} | Stmt::If {..} | Stmt::While {..} |
        Stmt::Syncthreads {..} | Stmt::KernelLaunch {..} |
        Stmt::MallocAsync {..} | Stmt::FreeAsync {..} | Stmt::Scope {..} => {
            let env = s.sfold(env, fv_expr);
            s.sfold(env, fv_stmt)
        }
    }
}

pub fn free_variables(s: &Vec<Stmt>) -> BTreeMap<Name, Type> {
    let env = s.iter().fold(FVEnv::default(), fv_stmt);
    env.free
}
