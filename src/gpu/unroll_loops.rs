use super::ast::*;
use crate::option::CompileOptions;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::{SFlatten, SMapAccum};
use crate::utils::substitute::SubVars;

use std::collections::BTreeMap;

#[derive(Clone)]
struct LoopInfo {
    var_ty: Type,
    var: Name,
    init: Expr,
    cond: Expr,
    incr: Expr,
    body: Vec<Stmt>,
    i: Info
}

fn extract_integer(e: &Expr) -> Option<i128> {
    match e {
        Expr::Int {v, ..} => Some(*v),
        _ => None
    }
}

fn apply_op(op: &BinOp, l: i128, r: i128) -> Option<i128> {
    match op {
        BinOp::Add => Some(l + r),
        BinOp::Mul => Some(l * r),
        _ => None
    }
}

fn lower_bound_of_first_thread(init: &Expr) -> Option<i128> {
    match init {
        Expr::ThreadIdx {..} | Expr::BlockIdx {..} => Some(0),
        Expr::BinOp {lhs, op, rhs, ..} => {
            lower_bound_of_first_thread(&lhs)
                .zip(lower_bound_of_first_thread(&rhs))
                .and_then(|(l, r)| apply_op(op, l, r))
        },
        Expr::Convert {e, ..} => lower_bound_of_first_thread(e),
        _ => extract_integer(init)
    }
}

fn upper_bound_value(var: &Name, cond: &Expr) -> Option<i128> {
    match cond {
        Expr::BinOp {lhs, op: BinOp::Lt, rhs, ..} => {
            match lhs.as_ref() {
                Expr::Var {id, ..} if id.eq(var) => extract_integer(rhs),
                _ => None
            }
        },
        _ => None
    }
}

fn extract_step_size(var: &Name, incr: &Expr) -> Option<i128> {
    match incr {
        Expr::BinOp {lhs, op: BinOp::Add, rhs, ..} => {
            match lhs.as_ref() {
                Expr::Var {id, ..} if id.eq(var) => extract_integer(rhs),
                _ => None
            }
        },
        _ => None
    }
}

fn unroll_stmt(
    li: &LoopInfo,
    hi: i128,
    num_threads: i128,
    stmt: Stmt
) -> Vec<Stmt> {
    let init_plus_offset = |ofs| {
        if ofs == 0 {
            li.init.clone()
        } else {
            Expr::BinOp {
                lhs: Box::new(li.init.clone()),
                op: BinOp::Add,
                rhs: Box::new(Expr::Int {v: ofs, ty: li.var_ty.clone(), i: li.i.clone()}),
                ty: li.var_ty.clone(),
                i: li.i.clone()
            }
        }
    };
    let mk_sub_env = |ofs| {
        let sub_expr = init_plus_offset(ofs);
        let mut sub_env = BTreeMap::new();
        sub_env.insert(li.var.clone(), sub_expr);
        sub_env
    };
    let mut acc = vec![];
    let mut ofs = 0;
    while ofs + num_threads <= hi {
        let sub_env = mk_sub_env(ofs);
        acc.push(stmt.clone().sub_vars(&sub_env));
        ofs += num_threads;
    }
    // Insert a conditionally executed statement, that we only run for threads whose index is below
    // a certain upper-bound. If the statement is an assignment, we wrap it in an if-expression to
    // help later analyses as it remains in the same scope.
    if ofs < hi {
        let sub_env = mk_sub_env(ofs);
        let max_thread_idx = hi - ofs;
        match stmt {
            Stmt::Expr {e: e @ Expr::Assign {..}, ..} => {
                let els_expr = Expr::Convert {
                    e: Box::new(Expr::Int {
                        v: 0, ty: Type::Scalar {sz: ElemSize::I32}, i: li.i.clone()
                    }),
                    ty: Type::Void
                };
                let ternary_stmt = Stmt::Expr {
                    e: Expr::IfExpr {
                        cond: Box::new(Expr::BinOp {
                            lhs: Box::new(li.init.clone()),
                            op: BinOp::Lt,
                            rhs: Box::new(Expr::Int {
                                v: max_thread_idx, ty: li.var_ty.clone(), i: li.i.clone()
                            }),
                            ty: li.var_ty.clone(),
                            i: li.i.clone()
                        }),
                        thn: Box::new(Expr::Convert {
                            e: Box::new(e.sub_vars(&sub_env)),
                            ty: Type::Void
                        }),
                        els: Box::new(els_expr),
                        ty: li.var_ty.clone(),
                        i: li.i.clone()
                    },
                    i: li.i.clone()
                };
                acc.push(ternary_stmt);
            },
            _ => {
                let cond_stmt = Stmt::If {
                    cond: Expr::BinOp {
                        lhs: Box::new(li.init.clone()),
                        op: BinOp::Lt,
                        rhs: Box::new(Expr::Int {
                            v: max_thread_idx, ty: li.var_ty.clone(), i: li.i.clone()
                        }),
                        ty: li.var_ty.clone(),
                        i: li.i.clone()
                    },
                    thn: vec![stmt.sub_vars(&sub_env)],
                    els: vec![],
                    i: li.i.clone()
                };
                acc.push(cond_stmt);
            }
        }
    }
    acc
}

fn unroll_body(
    li: &LoopInfo,
    hi: i128,
    num_threads: i128,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    body.into_iter()
        .map(|s| unroll_stmt(&li, hi, num_threads, s))
        .flatten()
        .collect::<Vec<Stmt>>()
}

fn is_leaf_stmt(s: &Stmt) -> bool {
    match s {
        Stmt::For {..} |
        Stmt::If {..} |
        Stmt::While {..} |
        Stmt::Scope {..} |
        Stmt::ParallelReduction {..} => {
            false
        },
        Stmt::Definition {..} |
        Stmt::Return {..} |
        Stmt::Expr {..} |
        Stmt::Synchronize {..} |
        Stmt::WarpReduce {..} |
        Stmt::ClusterReduce {..} |
        Stmt::KernelLaunch {..} |
        Stmt::AllocDevice {..} |
        Stmt::AllocShared {..} |
        Stmt::FreeDevice {..} |
        Stmt::CopyMemory {..} => {
            true
        }
    }
}

fn is_singleton_stmt_body(body: &Vec<Stmt>) -> bool {
    body.len() == 1 && body.iter().all(is_leaf_stmt)
}

fn should_unroll_loop(
    li: &LoopInfo,
    opts: &CompileOptions,
    lo: i128,
    hi: i128,
    step_size: i128
) -> bool {
    let niters = ((hi - lo) + step_size - 1) / step_size;
    // NOTE(larshum, 2025-11-28): We manually perform loop unrolling in three cases:
    // 1. The loop runs exactly one iteration for all threads. In this case, we simply replace the
    //    for-loop by a definition corresponding to the initial value of the for-loop.
    // 2. The loop contains exactly one statement, and it contains no more iterations than the
    //    globally specified limit (using the 'max_unroll_count' attribute). Note that this single
    //    statement must be a leaf node (as is the case for the reduction operators).
    //
    // If the user explicitly asks for unrolling in other cases, we inform the underlying compiler
    // about this instead of doing the unrolling ourselves.
    hi-lo == step_size ||
    (is_singleton_stmt_body(&li.body) && niters <= opts.max_unroll_count as i128)
}

fn try_unroll_loop(
    mut li: LoopInfo,
    opts: &CompileOptions
) -> Option<Vec<Stmt>> {
    let lo = lower_bound_of_first_thread(&li.init)?;
    let hi = upper_bound_value(&li.var, &li.cond)?;
    let step_size = extract_step_size(&li.var, &li.incr)?;
    if should_unroll_loop(&li, &opts, lo, hi, step_size) {
        let body = li.body.drain(..).collect::<Vec<Stmt>>();
        Some(unroll_body(&li, hi, step_size, body))
    } else {
        None
    }
}

fn unroll_loops_stmt(mut acc: Vec<Stmt>, s: Stmt, opts: &CompileOptions) -> Vec<Stmt> {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
            let li = LoopInfo {
                var_ty: var_ty.clone(),
                var: var.clone(),
                init: init.clone(),
                cond: cond.clone(),
                incr: incr.clone(),
                body: body.clone(),
                i: i.clone()
            };
            match try_unroll_loop(li, &opts) {
                Some(unrolled_body) => {
                    let mut body = unrolled_body
                        .sflatten(vec![], |acc, s| unroll_loops_stmt(acc, s, &opts));
                    acc.append(&mut body)
                },
                None => {
                    let body = body.sflatten(vec![], |acc, s| {
                        unroll_loops_stmt(acc, s, &opts)
                    });
                    acc.push(Stmt::For {
                        var_ty, var, init, cond, incr, body, unroll, i
                    });
                }
            };
            acc
        },
        _ => s.sflatten(acc, |acc, s| unroll_loops_stmt(acc, s, &opts))
    }
}

fn unroll_loops_top(t: Top, opts: &CompileOptions) -> Top {
    match t {
        Top::ExtDecl {..} | Top::FunDef {..} => t,
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = body.sflatten(vec![], |acc, s| unroll_loops_stmt(acc, s, &opts));
            Top::KernelFunDef {attrs, id, params, body, i}
        }
    }
}

pub fn apply(ast: Ast, opts: &CompileOptions) -> Ast {
    ast.smap(|t| unroll_loops_top(t, &opts))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;
    use crate::test::*;
    use crate::utils::pprint::*;

    fn eq_stmts(lhs: Vec<Stmt>, rhs: Vec<Stmt>) {
        let (_, l) = pprint_iter(lhs.iter(), PrettyPrintEnv::default(), "\n");
        let (_, r) = pprint_iter(rhs.iter(), PrettyPrintEnv::default(), "\n");
        assert_eq!(
            lhs,
            rhs,
            "LHS: {l}\nRHS: {r}"
        );
    }

    #[test]
    fn singleton_for_loop_is_unrolled() {
        let x = var("x", scalar(ElemSize::I64));
        let y = var("y", scalar(ElemSize::I64));
        let assign_to_y = |rhs| {
            assign(y.clone(), binop(y.clone(), BinOp::Add, rhs, scalar(ElemSize::I64)))
        };
        let s = Stmt::For {
            var_ty: scalar(ElemSize::I64),
            var: id("x"),
            init: int(0, Some(ElemSize::I64)),
            cond: binop(x.clone(), BinOp::Lt, int(10, Some(ElemSize::I64)), scalar(ElemSize::Bool)),
            incr: binop(x.clone(), BinOp::Add, int(1, Some(ElemSize::I64)), scalar(ElemSize::I64)),
            body: vec![assign_to_y(x.clone())],
            unroll: false,
            i: i()
        };
        let opts = CompileOptions::default();
        let expected = (0..10).into_iter()
            .map(|n| {
                let rhs = if n == 0 {
                    int(0, Some(ElemSize::I64))
                } else {
                    binop(
                        int(0, Some(ElemSize::I64)),
                        BinOp::Add,
                        int(n, Some(ElemSize::I64)),
                        scalar(ElemSize::I64))
                };
                assign_to_y(rhs)
            })
            .collect::<Vec<Stmt>>();
        eq_stmts(unroll_loops_stmt(vec![], s, &opts), expected);
    }

    #[test]
    fn singleton_unroll_threads() {
        let x = var("x", scalar(ElemSize::I64));
        let y = var("y", scalar(ElemSize::I64));
        let assign_to_y = |rhs| {
            assign(y.clone(), binop(y.clone(), BinOp::Add, rhs, scalar(ElemSize::I64)))
        };
        let s = Stmt::For {
            var_ty: scalar(ElemSize::I64),
            var: id("x"),
            init: Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()},
            cond: binop(x.clone(), BinOp::Lt, int(64, Some(ElemSize::I64)), scalar(ElemSize::Bool)),
            incr: binop(x.clone(), BinOp::Add, int(32, Some(ElemSize::I64)), scalar(ElemSize::I64)),
            body: vec![assign_to_y(x.clone())],
            unroll: false,
            i: i()
        };
        let opts = CompileOptions::default();
        let expected = (0..2).into_iter()
            .map(|n| {
                let rhs = if n == 0 {
                    Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()}
                } else {
                    binop(
                        Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()},
                        BinOp::Add,
                        int(n*32, Some(ElemSize::I64)),
                        scalar(ElemSize::I64)
                    )
                };
                assign_to_y(rhs)
            })
            .collect::<Vec<Stmt>>();
        eq_stmts(unroll_loops_stmt(vec![], s, &opts), expected);
    }

    #[test]
    fn singleton_partial_unroll() {
        let x = var("x", scalar(ElemSize::I64));
        let y = var("y", scalar(ElemSize::I64));
        let assign_to_y = |rhs| {
            assign(y.clone(), binop(y.clone(), BinOp::Add, rhs, scalar(ElemSize::I64)))
        };
        let s = Stmt::For {
            var_ty: scalar(ElemSize::I64),
            var: id("x"),
            init: Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()},
            cond: binop(x.clone(), BinOp::Lt, int(48, Some(ElemSize::I64)), scalar(ElemSize::Bool)),
            incr: binop(x.clone(), BinOp::Add, int(32, Some(ElemSize::I64)), scalar(ElemSize::I64)),
            body: vec![assign_to_y(x.clone())],
            unroll: false,
            i: i()
        };
        let opts = CompileOptions::default();
        let tidx = Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I64), i: i()};
        let expected = vec![
            assign_to_y(tidx.clone()),
            Stmt::Expr {
                e: Expr::IfExpr {
                    cond: Box::new(binop(
                        tidx.clone(),
                        BinOp::Lt,
                        int(48-32, Some(ElemSize::I64)),
                        scalar(ElemSize::Bool)
                    )),
                    thn: Box::new(convert(
                        Type::Void,
                        Expr::Assign {
                            lhs: Box::new(y.clone()),
                            rhs: Box::new(binop(
                                y.clone(),
                                BinOp::Add,
                                binop(
                                    tidx.clone(),
                                    BinOp::Add,
                                    int(32, Some(ElemSize::I64)),
                                    scalar(ElemSize::I64)),
                                scalar(ElemSize::I64))),
                            ty: scalar(ElemSize::I64),
                            i: i()
                        }
                    )),
                    els: Box::new(convert(Type::Void, int(0, Some(ElemSize::I64)))),
                    ty: scalar(ElemSize::I64),
                    i: i()
                },
                i: i()
            }
        ];
        eq_stmts(unroll_loops_stmt(vec![], s, &opts), expected);
    }

    #[test]
    fn for_loop_containing_non_empty_if_is_not_unrolled() {
        let x = id("x");
        let body = if_stmt(
            bool_expr(true),
            vec![assign(var("y", scalar(ElemSize::I64)), int(1, Some(ElemSize::I64)))],
            vec![assign(var("y", scalar(ElemSize::I64)), int(2, Some(ElemSize::I64)))]
        );
        let s = Stmt::For {
            var_ty: scalar(ElemSize::I64),
            var: x.clone(),
            init: int(0, Some(ElemSize::I64)),
            cond: binop(var("x", scalar(ElemSize::I64)), BinOp::Lt, int(10, Some(ElemSize::I64)), scalar(ElemSize::Bool)),
            incr: binop(var("x", scalar(ElemSize::I64)), BinOp::Add, int(1, Some(ElemSize::I64)), scalar(ElemSize::I64)),
            body: vec![body],
            unroll: false,
            i: i()
        };
        let opts = CompileOptions::default();
        eq_stmts(unroll_loops_stmt(vec![], s.clone(), &opts), vec![s]);
    }
}
