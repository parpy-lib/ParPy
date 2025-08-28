use super::ast::*;
use crate::par;
use crate::parpy_compile_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

type TpbMapping = BTreeMap<Name, i64>;

fn merge_tpb(ltpb: i64, rtpb: i64, i: &Info) -> CompileResult<i64> {
    if ltpb == par::DEFAULT_TPB {
        Ok(rtpb)
    } else if rtpb == par::DEFAULT_TPB {
        Ok(ltpb)
    } else if ltpb == rtpb {
        Ok(ltpb)
    } else {
        parpy_compile_error!(i, "Found inconsistent threads per block {0} and \
                                 {1} among labels within parallel statements.",
                                 ltpb, rtpb)
    }
}

fn collect_tpb_expr(acc: i64, e: &Expr) -> CompileResult<i64> {
    match e {
        Expr::Call {args, par, i, ..} => {
            let acc = merge_tpb(acc, par.tpb, &i);
            args.sfold_result(acc, collect_tpb_expr)
        },
        _ => e.sfold_result(Ok(acc), collect_tpb_expr)
    }
}

fn collect_tpb_stmt_par(acc: i64, s: &Stmt) -> CompileResult<i64> {
    match s {
        Stmt::For {body, par, i, ..} => {
            let tpb = merge_tpb(acc, par.tpb, &i);
            body.sfold_result(tpb, collect_tpb_stmt_par)
        },
        _ => {
            let acc = s.sfold_result(Ok(acc), collect_tpb_stmt_par);
            s.sfold_result(acc, collect_tpb_expr)
        }
    }
}

fn collect_tpb_stmt(mut acc: TpbMapping, s: &Stmt) -> CompileResult<TpbMapping> {
    match s {
        Stmt::For {var, body, par, ..} if par.is_parallel() => {
            let tpb = body.sfold_result(Ok(par.tpb), collect_tpb_stmt_par)?;
            acc.insert(var.clone(), tpb);
        },
        _ => ()
    };
    Ok(acc)
}

fn collect_tpb_fun_def(fun: &FunDef) -> CompileResult<TpbMapping> {
    fun.body.sfold_result(Ok(BTreeMap::new()), collect_tpb_stmt)
}

fn propagate_tpb_expr(tpb: i64, e: Expr) -> Expr {
    match e {
        Expr::Call {id, args, par, ty, i} => {
            let par = par.tpb(tpb).unwrap();
            let args = args.smap(|e| propagate_tpb_expr(tpb, e));
            Expr::Call {id, args, par, ty, i}
        },
        _ => e.smap(|e| propagate_tpb_expr(tpb, e))
    }
}

fn propagate_tpb_stmt_par(tpb: i64, s: Stmt) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            let body = body.smap(|s| propagate_tpb_stmt_par(tpb, s));
            let par = par.tpb(tpb).unwrap();
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        _ => {
            let s = s.smap(|s| propagate_tpb_stmt_par(tpb, s));
            s.smap(|e| propagate_tpb_expr(tpb, e))
        }
    }
}

fn propagate_tpb_stmt(mapping: &TpbMapping, s: Stmt) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            if let Some(tpb) = mapping.get(&var) {
                let body = body.smap(|s| propagate_tpb_stmt_par(*tpb, s));
                let par = par.tpb(*tpb).unwrap();
                Stmt::For {var, lo, hi, step, body, par, i}
            } else {
                let body = body.smap(|s| propagate_tpb_stmt(&mapping, s));
                Stmt::For {var, lo, hi, step, body, par, i}
            }
        },
        _ => s.smap(|s| propagate_tpb_stmt(&mapping, s))
    }
}

fn propagate_tpb_fun_def(mapping: TpbMapping, fun: FunDef) -> FunDef {
    let body = fun.body.smap(|s| propagate_tpb_stmt(&mapping, s));
    FunDef {body, ..fun}
}

/// Ensure all parallel for-loops within the same parallel loop nest agree on the number of threads
/// to use per block.
pub fn propagate_configuration(ast: Ast) -> CompileResult<Ast> {
    let mapping = collect_tpb_fun_def(&ast.main)?;
    let main = propagate_tpb_fun_def(mapping, ast.main);
    Ok(Ast {main, ..ast})
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

    fn to_map<K: std::cmp::Ord, V>(v: Vec<(K, V)>) -> BTreeMap<K, V> {
        v.into_iter().collect::<_>()
    }

    fn nested_loops1(tpb: i64) -> Stmt {
        assert!(par::DEFAULT_TPB != tpb);
        let p = LoopPar::default().threads(10).unwrap().tpb(tpb).unwrap();
        for_loop_complete(id("x"), int(0, None), int(10, None), 1, p, vec![
            for_loop(id("y"), 10, vec![])
        ])
    }

    fn nested_loops2(tpb: i64) -> Stmt {
        assert!(par::DEFAULT_TPB != tpb);
        let p = LoopPar::default().threads(10).unwrap().tpb(tpb).unwrap();
        for_loop(id("x"), 10, vec![
            for_loop_complete(id("y"), int(0, None), int(10, None), 1, p, vec![])
        ])
    }

    #[test]
    fn collect_nested_tpb_outer() {
        let s = nested_loops1(128);
        let mapping = collect_tpb_stmt(BTreeMap::new(), &s);
        assert_eq!(mapping, Ok(to_map(vec![(id("x"), 128)])));
    }

    #[test]
    fn collect_nested_tpb_inner() {
        let s = nested_loops2(128);
        let mapping = collect_tpb_stmt(BTreeMap::new(), &s);
        assert_eq!(mapping, Ok(to_map(vec![(id("x"), 128)])));
    }

    #[test]
    fn propagation_nested_tpb_inward() {
        let tpb = 256;
        let s = nested_loops1(tpb);
        let mapping = to_map(vec![(id("x"), tpb)]);
        let p = LoopPar::default().threads(10).unwrap().tpb(tpb).unwrap();
        let expected = for_loop_complete(id("x"), int(0, None), int(10, None), 1, p.clone(), vec![
            for_loop_complete(id("y"), int(0, None), int(10, None), 1, p, vec![])
        ]);
        assert_eq!(propagate_tpb_stmt(&mapping, s), expected);
    }

    #[test]
    fn propagation_nested_tpb_outward() {
        let tpb = 512;
        let s = nested_loops2(tpb);
        let mapping = to_map(vec![(id("x"), tpb)]);
        let p = LoopPar::default().threads(10).unwrap().tpb(tpb).unwrap();
        let expected = for_loop_complete(id("x"), int(0, None), int(10, None), 1, p.clone(), vec![
            for_loop_complete(id("y"), int(0, None), int(10, None), 1, p, vec![])
        ]);
        assert_eq!(propagate_tpb_stmt(&mapping, s), expected);
    }
}
