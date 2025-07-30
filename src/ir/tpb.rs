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

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ast_builder::*;
    use crate::test::*;

    #[test]
    fn merge_tpb_default_default() {
        assert_eq!(merge_tpb(par::DEFAULT_TPB, par::DEFAULT_TPB, &i()), Ok(par::DEFAULT_TPB));
    }

    #[test]
    fn merge_tpb_default_non_default() {
        assert_eq!(merge_tpb(par::DEFAULT_TPB, 128, &i()), Ok(128));
    }

    #[test]
    fn merge_tpb_non_default_default() {
        assert_eq!(merge_tpb(128, par::DEFAULT_TPB, &i()), Ok(128));
    }

    #[test]
    fn merge_tbp_inconsistent() {
        assert_error_matches(merge_tpb(128, 256, &i()), r"inconsistent threads per block");
    }

    #[test]
    fn test_no_propagation_for_sequential_loops() {
        let p = LoopPar::default().tpb(512).unwrap();
        let body = vec![
            for_loop_complete(id("x"), int(0, None), int(10, None), 1, p, vec![
                for_loop(id("y"), 10, vec![])
            ])
        ];
        match propagate_configuration_stmts(body).unwrap().get(0).unwrap() {
            Stmt::For {body, par, ..} => {
                assert_eq!(par.tpb, 512);
                match body.get(0).unwrap() {
                    Stmt::For {par, ..} => assert_eq!(par.tpb, par::DEFAULT_TPB),
                    _ => assert!(false)
                };
            },
            _ => assert!(false)
        }
    }

    #[test]
    fn test_propagation_nested_for_loops() {
        let p = LoopPar::default().threads(10).unwrap().tpb(512).unwrap();
        let body = vec![
            for_loop_complete(id("x"), int(0, None), int(10, None), 1, p, vec![
                for_loop(id("y"), 10, vec![])
            ])
        ];
        match propagate_configuration_stmts(body).unwrap().get(0).unwrap() {
            Stmt::For {body, par, ..} => {
                assert_eq!(par.tpb, 512);
                match body.get(0).unwrap() {
                    Stmt::For {par, ..} => assert_eq!(par.tpb, 512),
                    _ => assert!(false)
                };
            },
            _ => assert!(false)
        }
    }
}
