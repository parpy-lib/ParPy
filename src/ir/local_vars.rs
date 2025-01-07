use crate::info::Info;
use super::ast::*;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
struct LocalVar {
    ty: Type,
    id: String,
    i: Info
}

impl PartialEq for LocalVar {
    fn eq(&self, v: &Self) -> bool {
        self.id == v.id
    }
}

impl Eq for LocalVar {}

impl Hash for LocalVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

fn collect_local_variable_declarations_stmt(
    mut vars: HashSet<LocalVar>,
    s: &Stmt
) -> HashSet<LocalVar> {
    match s {
        Stmt::Declaration {id, i, ..} => {
            let v = LocalVar {ty: Type::Unknown, id: id.clone(), i: i.clone()};
            vars.remove(&v);
            vars
        },
        Stmt::Assignment {dst: Expr::Var {id, ..}, e, i, ..} => {
            let v = LocalVar {
                ty: e.get_type().clone(),
                id: id.clone(),
                i: i.clone()
            };
            vars.insert(v);
            vars
        },
        Stmt::Assignment {..} => vars,
        Stmt::For {body, ..} => {
            let local_vars = collect_local_variable_declarations_stmts(body);
            for v in local_vars {
                vars.insert(v);
            }
            vars
        },
    }
}

fn collect_local_variable_declarations_stmts(body: &Vec<Stmt>) -> HashSet<LocalVar> {
    body.iter()
        .fold(HashSet::new(), |vars, stmt| collect_local_variable_declarations_stmt(vars, stmt))
}

fn insert_local_variable_declarations_body(
    body: Vec<Stmt>,
    local_vars: HashSet<LocalVar>
) -> Vec<Stmt> {
    local_vars.into_iter()
        .map(|LocalVar {ty, id, i}| Stmt::Declaration {
            ty: ty.clone(), id: id.clone(), i: i.clone()
        })
        .chain(body.into_iter())
        .collect()
}

fn insert_local_variable_declarations_top(top: Top) -> Top {
    match top {
        Top::KernelDef {id, params, body, nblocks, nthreads, i} => {
            let local_vars = collect_local_variable_declarations_stmts(&body);
            let body = insert_local_variable_declarations_body(body, local_vars);
            Top::KernelDef {id, params, body, nblocks, nthreads, i}
        },
    }
}

pub fn insert_local_variable_declarations(ast: Ast) -> Ast {
    ast.into_iter()
        .map(insert_local_variable_declarations_top)
        .collect::<Ast>()
}
