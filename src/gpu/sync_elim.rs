use super::ast::*;
use crate::utils::smap::*;

fn remove_redundant_synchronization_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Synchronize {scope: ref s1, ..} => {
            match acc.last() {
                Some(Stmt::Synchronize {scope: s2, ..}) if s1 == s2 => (),
                _ => acc.push(s),
            };
            acc
        },
        _ => s.sflatten(acc, remove_redundant_synchronization_stmt)
    }
}

fn remove_redundant_synchronization_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.sflatten(vec![], remove_redundant_synchronization_stmt)
}

fn remove_redundant_synchronization_top(t: Top) -> Top {
    match t {
        Top::KernelFunDef {attrs, id, params, body} => {
            let mut body = remove_redundant_synchronization_stmts(body);
            match body.last() {
                Some(Stmt::Synchronize {scope: SyncScope::Block, ..}) => {
                    body.pop();
                },
                _ => ()
            };
            Top::KernelFunDef {attrs, id, params, body}
        },
        _ => t
    }
}

pub fn remove_redundant_synchronization(ast: Ast) -> Ast {
    ast.into_iter().map(remove_redundant_synchronization_top).collect::<Ast>()
}
