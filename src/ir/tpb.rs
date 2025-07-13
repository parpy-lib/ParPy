use super::ast::*;
use crate::par;
use crate::prickle_compile_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::smap::SFold;
use crate::utils::smap::SMapAccum;

fn merge_tpb(ltpb: i64, rtpb: i64, i: &Info) -> CompileResult<i64> {
    if ltpb == par::DEFAULT_TPB {
        Ok(rtpb)
    } else if rtpb == par::DEFAULT_TPB {
        Ok(ltpb)
    } else if ltpb == rtpb {
        Ok(ltpb)
    } else {
        prickle_compile_error!(i, "Found inconsistent threads per block {0} and \
                                 {1} among labels within parallel for-loop.",
                                 ltpb, rtpb)
    }
}

fn find_threads_per_block_stmt(acc: i64, s: &Stmt) -> CompileResult<i64> {
    match s {
        Stmt::For {par, i, ..} => merge_tpb(acc, par.tpb, &i),
        _ => s.sfold_result(Ok(acc), find_threads_per_block_stmt)
    }
}

fn find_threads_per_block(acc: i64, stmts: &Vec<Stmt>) -> CompileResult<i64> {
    stmts.sfold_result(Ok(acc), find_threads_per_block_stmt)
}

fn propagate_threads_per_block_stmt(tpb: i64, s: Stmt) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, mut par, i} => {
            let body = propagate_threads_per_block(tpb, body);
            par.tpb = tpb;
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        _ => s.smap(|s| propagate_threads_per_block_stmt(tpb, s))
    }
}

fn propagate_threads_per_block(tpb: i64, stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.smap(|s| propagate_threads_per_block_stmt(tpb, s))
}

fn propagate_configuration_stmt(s: Stmt) -> CompileResult<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            let tpb = find_threads_per_block(par.tpb, &body)?;
            let body = propagate_threads_per_block(tpb, body);
            Ok(Stmt::For {var, lo, hi, step, body, par, i})
        },
        _ => s.smap_result(propagate_configuration_stmt)
    }
}

fn propagate_configuration_stmts(stmts: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.smap_result(propagate_configuration_stmt)
}

fn propagate_configuration_fun_def(fun: FunDef) -> CompileResult<FunDef> {
    let body = propagate_configuration_stmts(fun.body)?;
    Ok(FunDef {body, ..fun})
}

/// Ensure all parallel for-loops within the same parallel loop nest agree on the number of threads
/// to use per block.
pub fn propagate_configuration(ast: Ast) -> CompileResult<Ast> {
    let defs = ast.defs.smap_result(propagate_configuration_fun_def)?;
    Ok(Ast {defs, ..ast})
}
