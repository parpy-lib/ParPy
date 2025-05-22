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
        _ => todo!()
    }
}

fn fv_stmt(mut env: FVEnv, s: &Stmt) -> FVEnv {
    match s {
        Stmt::Definition {id, expr, ..} => {
            let mut env = fv_expr(env, expr);
            env.bound.insert(id.clone(), expr.get_type().clone());
            env
        },
        _ => todo!()
    }
}

pub fn free_variables(s: &Vec<Stmt>) -> BTreeMap<Name, Type> {
    let env = s.iter().fold(FVEnv::default(), fv_stmt);
    env.free
}
