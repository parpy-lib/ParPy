use crate::gpu::ast::*;
use crate::utils::err::*;
use crate::utils::smap::{SFlatten, SMapAccum};

fn remove_sync_stmt(mut stmts: Vec<Stmt>, s: Stmt) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::Synchronize {scope: SyncScope::Block, i} => {
            if let Some(Stmt::ParallelReduction {..}) = stmts.last() {
                Ok(stmts)
            } else {
                stmts.push(Stmt::Synchronize {scope: SyncScope::Block, i});
                Ok(stmts)
            }
        },
        _ => s.sflatten_result(stmts, remove_sync_stmt)
    }
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = body.sflatten_result(vec![], remove_sync_stmt)?;
            Ok(Top::KernelFunDef {attrs, id, params, body, i})
        },
        _ => Ok(t)
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(apply_top)
}
