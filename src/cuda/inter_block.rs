use super::par;
use super::par_tree;
use crate::parir_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::reduce;
use crate::utils::smap::{SFold, SMapAccum};

use std::collections::{BTreeMap, BTreeSet};

fn insert_synchronization_points_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let is_par = par.is_parallel();
            let body = insert_synchronization_points(body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i: i.clone()});
            if is_par {
                acc.push(Stmt::SyncPoint {block_local: false, i});
            }
        },
        Stmt::While {cond, body, i} => {
            let body = insert_synchronization_points(body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = insert_synchronization_points(thn);
            let els = insert_synchronization_points(els);
            acc.push(Stmt::If {cond, thn, els, i});
        },
    }
    acc
}

fn insert_synchronization_points(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.into_iter()
        .fold(vec![], |acc, s| {
            insert_synchronization_points_stmt(acc, s)
        })
}

fn classify_synchronization_points_par_stmt(
    node: &par_tree::ParNode,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Alloc {..} |
        Stmt::Free {..} => {
            acc.push(s);
        },
        Stmt::SyncPoint {block_local, i} => {
            let prev_stmt_is_block_local_for = match acc.last() {
                Some(Stmt::For {par, ..}) => {
                    par.nthreads > 0 && par.nthreads <= par::DEFAULT_TPB
                },
                _ => false
            };
            // When the synchronization statement is found in the innermost level of parallelism,
            // and the preceding for-loop is a parallel for-loop with at most the default number of
            // threads per block (par::DEFAULT_TPB), we consider this synchronization point to be
            // block-local. In this case, we can use a CUDA intrinsic instead of splitting up the
            // kernel.
            let s = if node.innermost_parallelism() && prev_stmt_is_block_local_for {
                Stmt::SyncPoint {block_local: true, i}
            } else {
                Stmt::SyncPoint {block_local, i}
            };
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let node = node.children.get(&var).unwrap_or(node);
            let body = classify_synchronization_points_par_stmts(node, body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = classify_synchronization_points_par_stmts(node, body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = classify_synchronization_points_par_stmts(node, thn);
            let els = classify_synchronization_points_par_stmts(node, els);
            acc.push(Stmt::If {cond, thn, els, i});
        }
    };
    acc
}

fn classify_synchronization_points_par_stmts(
    par: &par_tree::ParNode,
    stmts: Vec<Stmt>
) -> Vec<Stmt> {
    stmts.into_iter()
        .fold(vec![], |acc, s| classify_synchronization_points_par_stmt(par, acc, s))
}

fn classify_synchronization_points_stmt(
    t: &par_tree::ParTree,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = if let Some(node) = t.roots.get(&var) {
                classify_synchronization_points_par_stmts(&node, body)
            } else {
                body.smap(|s| classify_synchronization_points_stmt(t, s))
            };
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Alloc {..} | Stmt::Free {..} => {
            s.smap(|s| classify_synchronization_points_stmt(t, s))
        }
    }
}

fn classify_synchronization_points(
    par: &par_tree::ParTree,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    body.smap(|s| classify_synchronization_points_stmt(par, s))
}

fn extract_neutral_element(
    op: &BinOp, sz: &ElemSize, i: &Info
) -> CompileResult<Expr> {
    match reduce::neutral_element(op, sz, i) {
        Some(ne) => Ok(ne),
        None => parir_compile_error!(i, "Reduction operation {0} has unknown \
                                         neutral element.",
                                         op.pprint_default())
    }
}

fn inner_multi_block_reduce_loop(
    loop_idx: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    nblocks: i64,
    block_idx: Name,
    lhs: Expr,
    op: BinOp,
    rhs: Expr,
    i: &Info
) -> Stmt {
    let i64_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
    let block = Expr::Var {id: block_idx, ty: i64_ty.clone(), i: i.clone()};
    let lo_expr = Expr::BinOp {
        lhs: Box::new(lo),
        op: BinOp::Add,
        rhs: Box::new(Expr::BinOp {
            lhs: Box::new(block),
            op: BinOp::Mul,
            rhs: Box::new(Expr::Int {
                v: step * par::DEFAULT_TPB,
                ty: i64_ty.clone(), i: i.clone()
            }),
            ty: i64_ty.clone(),
            i: i.clone()
        }),
        ty: i64_ty.clone(),
        i: i.clone()
    };
    let assign = Stmt::Assign {
        dst: lhs.clone(),
        expr: Expr::BinOp {
            lhs: Box::new(lhs.clone()), op, rhs: Box::new(rhs),
            ty: lhs.get_type().clone(), i: i.clone()
        },
        i: i.clone()
    };
    Stmt::For {
        var: loop_idx, lo: lo_expr, hi, step: step * nblocks,
        body: vec![assign],
        par: LoopParallelism {nthreads: par::DEFAULT_TPB, reduction: true},
        i: i.clone()
    }
}

fn outer_multi_block_reduce_loop(
    inner_loop: Stmt,
    temp_id: Name,
    ne: Expr,
    var: Name,
    nblocks: usize,
    i: &Info
) -> Stmt {
    let i64_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
    let init_stmt = Stmt::Definition {
        ty: ne.get_type().clone(),
        id: temp_id,
        expr: ne,
        i: i.clone()
    };
    Stmt::For {
        var,
        lo: Expr::Int {v: 0, ty: i64_ty.clone(), i: i.clone()},
        hi: Expr::Int {v: nblocks as i64, ty: i64_ty.clone(), i: i.clone()},
        step: 1,
        body: vec![init_stmt, inner_loop],
        par: LoopParallelism {nthreads: nblocks as i64, reduction: false},
        i: i.clone()
    }
}

fn single_block_reduce_loop(
    var: Name,
    nblocks: usize,
    lhs: Expr,
    op: BinOp,
    rhs: Expr,
    i: &Info
) -> Stmt {
    let i64_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
    let assign = Stmt::Assign {
        dst: lhs.clone(),
        expr: Expr::BinOp {
            lhs: Box::new(lhs.clone()),
            op,
            rhs: Box::new(rhs),
            ty: lhs.get_type().clone(),
            i: i.clone()
        },
        i: i.clone()
    };
    // Use the number of blocks, or at most 1024 threads, to ensure the loop is mapped to a single
    // block.
    let par = LoopParallelism {
        nthreads: i64::min(nblocks as i64, 1024),
        reduction: true
    };
    Stmt::For {
        var,
        lo: Expr::Int {v: 0, ty: i64_ty.clone(), i: i.clone()},
        hi: Expr::Int {v: nblocks as i64, ty: i64_ty.clone(), i: i.clone()},
        step: 1, body: vec![assign], par, i: i.clone()
    }
}

fn split_inter_block_parallel_reductions_stmt(s: Stmt) -> CompileResult<Stmt> {
    let is_inter_block_reduction = |par: &LoopParallelism| {
        par.reduction && par.nthreads > par::DEFAULT_TPB
    };
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if is_inter_block_reduction(&par) => {
            if par.nthreads % par::DEFAULT_TPB == 0 {
                let nblocks = (par.nthreads / par::DEFAULT_TPB) as usize;
                let (lhs, op, rhs, sz, i) = reduce::extract_reduction_operands(body, &i)?;
                let ty = Type::Tensor {sz: sz.clone(), shape: vec![]};
                let ne = extract_neutral_element(&op, &sz, &i)?;
                let id = Name::sym_str("block_idx");
                let temp_id = Name::sym_str("tmp");
                // NOTE: We use a variable here, which is technically incorrect. However, by doing
                // so, we can reuse the code for allocating temporary memory for local variables
                // used across multiple kernels. It will always apply to the generated code for a
                // multi-block reduction because we insert a inter-block synchronization point
                // between the two loops.
                let temp_access = Expr::Var {
                    id: temp_id.clone(), ty: ty.clone(), i: i.clone()
                };
                let l1 = inner_multi_block_reduce_loop(
                    var, lo, hi, step, nblocks as i64,
                    id.clone(), temp_access.clone(), op.clone(), rhs, &i
                );
                let l2 = outer_multi_block_reduce_loop(
                    l1, temp_id, ne, id.clone(), nblocks, &i
                );
                let l3 = single_block_reduce_loop(
                    id, nblocks, lhs, op, temp_access, &i
                );
                let thn = vec![
                    l2,
                    Stmt::SyncPoint {block_local: false, i: i.clone()},
                    l3
                ];
                let bool_ty = Type::Tensor {sz: ElemSize::Bool, shape: vec![]};
                let true_expr = Expr::Bool {v: true, ty: bool_ty, i: i.clone()};
                Ok(Stmt::If {cond: true_expr, thn, els: vec![], i: i.clone()})
            } else {
                parir_compile_error!(i, "Multi-block reductions must use a \
                                         multiple of {0} threads.",
                                         par::DEFAULT_TPB)
            }
        },
        _ => s.smap_result(split_inter_block_parallel_reductions_stmt)
    }
}

fn flatten_true_conds(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = body.into_iter().fold(vec![], flatten_true_conds);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = body.into_iter().fold(vec![], flatten_true_conds);
            acc.push(Stmt::While {cond, body, i})
        },
        Stmt::If {cond: Expr::Bool {v: true, ..}, thn, ..} => {
            let mut thn = thn.into_iter().fold(vec![], flatten_true_conds);
            acc.append(&mut thn);
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = thn.into_iter().fold(vec![], flatten_true_conds);
            let els = els.into_iter().fold(vec![], flatten_true_conds);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s)
        }
    };
    acc
}

fn split_inter_block_parallel_reductions(
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    let body = body.smap_result(split_inter_block_parallel_reductions_stmt)?;
    Ok(body.into_iter().fold(vec![], flatten_true_conds))
}

fn is_inter_block_sync_point(s: &Stmt) -> bool {
    match s {
        Stmt::SyncPoint {block_local: false, ..} => true,
        _ => false
    }
}

fn contains_inter_block_sync_point(acc: bool, s: &Stmt) -> bool {
    s.sfold(acc || is_inter_block_sync_point(s), contains_inter_block_sync_point)
}

// Determines if the statement is a sequential for-loop containing an inter-block synchronization
// point.
fn is_seq_loop_with_inter_block_sync_point(s: &Stmt) -> bool {
    match s {
        Stmt::For {body, par, ..} if !par.is_parallel() => {
            body.sfold(false, contains_inter_block_sync_point)
        },
        _ => false
    }
}

fn hoist_chunk(
    var: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    par: LoopParallelism,
    i: Info,
    chunk: &[Stmt]
) -> CompileResult<Vec<Stmt>> {
    // As we perform an inclusive split, each part of the split will always contain at
    // least one element, so it is safe to unwrap this.
    let last_stmt = chunk.last().unwrap();

    if is_seq_loop_with_inter_block_sync_point(last_stmt) {
        // If the last statement of the chunk is a sequential loop with an inter-block
        // synchronization point, we extract the pre-statements and then process the
        // sequential loop afterward.
        let pre_stmts = hoist_inner_seq_loops_par_stmts(chunk[..chunk.len()-1].to_vec())?;
        let pre_stmt = Stmt::For {
            var: var.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            step,
            body: pre_stmts,
            par: par.clone(),
            i: i.clone()
        };
        let seq_loop_stmt = match last_stmt.clone() {
            Stmt::For {var: seq_var, lo: seq_lo, hi: seq_hi, step: seq_step,
                       body: seq_body, par: seq_par, i: seq_i} => {
                // Split up the body of the sequential for-loop such that each inter-block
                // synchronization point is at the end of a chunk. We place each chunk inside the
                // outer parallel for-loop.
                let inner_stmts = seq_body.split_inclusive(is_inter_block_sync_point)
                    .map(|chunk| {
                        let s = Stmt::For {
                            var: var.clone(),
                            lo: lo.clone(),
                            hi: hi.clone(),
                            step,
                            body: chunk.to_vec(),
                            par: par.clone(),
                            i: i.clone()
                        };
                        // If a parallel for-loop ends with a synchronization point, we include
                        // this after it as well so that it is properly split up.
                        match chunk.last() {
                            Some(sync_point @ Stmt::SyncPoint {..}) => {
                                vec![s, sync_point.clone()]
                            },
                            _ => vec![s]
                        }
                    })
                    .map(hoist_inner_seq_loops_par_stmts)
                    .collect::<CompileResult<Vec<Vec<Stmt>>>>()?
                    .concat();
                // Reconstruct the sequential for-loop outside of the parallel for-loops.
                Ok(Stmt::For {
                    var: seq_var, lo: seq_lo, hi: seq_hi, step: seq_step,
                    body: inner_stmts, par: seq_par, i: seq_i
                })
            },
            _ => parir_compile_error!(&i, "Internal error when hoisting \
                                           sequential loop")
        }?;
        Ok(vec![pre_stmt, seq_loop_stmt])
    } else {
        // Otherwise, if the chunk does not contain any applicable sequential loops, we recurse
        // into the body to produce the resulting body of the parallel for-loop.
        let body = hoist_inner_seq_loops_par_stmts(chunk.to_vec())?;

        // If the body contains an applicable loop after recursing down, we run the outer
        // transformation again to hoist it outside of this loop as well.
        if body.iter().any(is_seq_loop_with_inter_block_sync_point) {
            hoist_seq_loops(var, lo, hi, step, body, par, i)
        } else {
            Ok(vec![Stmt::For { var, lo, hi, step, body, par, i }])
        }
    }
}

fn hoist_seq_loops(
    var: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    body: Vec<Stmt>,
    par: LoopParallelism,
    i: Info
) -> CompileResult<Vec<Stmt>> {
    Ok(body.split_inclusive(is_seq_loop_with_inter_block_sync_point)
        .map(|chunk| {
            hoist_chunk(var.clone(), lo.clone(), hi.clone(), step, par.clone(),
                        i.clone(), chunk)
        })
        .collect::<CompileResult<Vec<Vec<Stmt>>>>()?
        .concat())
}

fn hoist_inner_seq_loops_par_stmt(
    acc: CompileResult<Vec<Stmt>>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    let mut acc = acc?;
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            acc.append(&mut hoist_seq_loops(var, lo, hi, step, body, par, i)?);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = hoist_inner_seq_loops_par_stmts(body)?;
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = hoist_inner_seq_loops_par_stmts(body)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = hoist_inner_seq_loops_par_stmts(thn)?;
            let els = hoist_inner_seq_loops_par_stmts(els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        }
    };
    Ok(acc)
}

fn hoist_inner_seq_loops_par_stmts(stmts: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), hoist_inner_seq_loops_par_stmt)
}

fn hoist_inner_sequential_loops_stmt(
    t: &par_tree::ParTree,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s);
        },
        Stmt::For {ref var, ..} if t.roots.contains_key(&var) => {
            let mut stmts = hoist_inner_seq_loops_par_stmt(Ok(vec![]), s)?;
            acc.append(&mut stmts);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = hoist_inner_sequential_loops(t, body)?;
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = hoist_inner_sequential_loops(t, body)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = hoist_inner_sequential_loops(t, thn)?;
            let els = hoist_inner_sequential_loops(t, els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        },
    }
    Ok(acc)
}

fn hoist_inner_sequential_loops(
    t: &par_tree::ParTree,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    body.into_iter()
        .fold(Ok(vec![]), |acc, s| hoist_inner_sequential_loops_stmt(t, acc?, s))
}

fn split_inter_block_synchronization_kernel(
    mut acc: Vec<Stmt>, s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = split_inter_block_synchronization_kernel_stmts(body);
            let mut bodies = body.split_inclusive(is_inter_block_sync_point)
                .map(|chunk| {
                    Stmt::For {
                        var: var.clone(), lo: lo.clone(), hi: hi.clone(), step,
                        body: chunk.to_vec(), par: par.clone(), i: i.clone()}
                })
                .collect::<Vec<Stmt>>();
            acc.append(&mut bodies);
        },
        Stmt::While {cond, body, i} => {
            let body = split_inter_block_synchronization_kernel_stmts(body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = split_inter_block_synchronization_kernel_stmts(thn);
            let els = split_inter_block_synchronization_kernel_stmts(els);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s);
        }
    };
    acc
}

fn split_inter_block_synchronization_kernel_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.clone()
        .into_iter()
        .fold(vec![], split_inter_block_synchronization_kernel)
}

fn split_inter_block_synchronization_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::For {ref par, ..} if par.is_parallel() => {
            let stmts = split_inter_block_synchronization_kernel(vec![], s.clone());
            let i = s.get_info();
            let true_expr = Expr::Bool {v: true, ty: Type::Tensor {sz: ElemSize::Bool, shape: vec![]}, i: i.clone()};
            Stmt::If {cond: true_expr, thn: stmts, els: vec![], i}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::For {..} | Stmt::While {..} | Stmt::If {..} | Stmt::Alloc {..} |
        Stmt::Free {..} => s.smap(split_inter_block_synchronization_stmt)
    }
}

fn split_inter_block_synchronizations(body: Vec<Stmt>) -> Vec<Stmt> {
    body.into_iter()
        .map(split_inter_block_synchronization_stmt)
        .fold(vec![], flatten_true_conds)
}

#[derive(Clone, Debug)]
struct SyncPointEnv {
    in_parallel: bool,
    parallel_loop_body: bool
}

impl SyncPointEnv {
    fn new() -> Self {
        SyncPointEnv {in_parallel: false, parallel_loop_body: false}
    }

    fn enter_loop(&self, is_parallel: bool) -> Self {
        SyncPointEnv {
            in_parallel: self.in_parallel || is_parallel,
            parallel_loop_body: is_parallel
        }
    }
}

fn eliminate_unnecessary_synchronization_points_stmt(
    env: SyncPointEnv,
    mut acc: Vec<Stmt>,
    s: Stmt,
) -> Vec<Stmt> {
    match s {
        Stmt::SyncPoint {block_local: false, ..} if !env.in_parallel => (),
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let env = env.enter_loop(par.is_parallel());
            let body = eliminate_unnecessary_synchronization_points_stmts(env, body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let env = env.enter_loop(false);
            let body = eliminate_unnecessary_synchronization_points_stmts(env, body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = eliminate_unnecessary_synchronization_points_stmts(env.clone(), thn);
            let els = eliminate_unnecessary_synchronization_points_stmts(env, els);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::Free {..} => {
            acc.push(s);
        },
    }
    acc
}

fn eliminate_unnecessary_synchronization_points_stmts(
    env: SyncPointEnv,
    mut stmts: Vec<Stmt>,
) -> Vec<Stmt> {
    if env.parallel_loop_body {
        if let Some(Stmt::SyncPoint {..}) = stmts.last() {
            stmts.pop();
        }
    }
    stmts.into_iter()
        .fold(vec![], |acc, s| {
            eliminate_unnecessary_synchronization_points_stmt(env.clone(), acc, s)
        })
}

fn eliminate_unnecessary_synchronization_points(body: Vec<Stmt>) -> Vec<Stmt> {
    let env = SyncPointEnv::new();
    eliminate_unnecessary_synchronization_points_stmts(env, body)
}

fn sub_var_expr(from_id: &Name, to_id: &Name, e: Expr) -> Expr {
    match e {
        Expr::Var {id, ty, i} if id == *from_id => {
            Expr::Var {id: to_id.clone(), ty, i}
        },
        _ => e.smap(|e| sub_var_expr(from_id, to_id, e))
    }
}

fn sub_var_stmt(from_id: &Name, to_id: &Name, s: Stmt) -> Stmt {
    s.smap(|s| sub_var_stmt(from_id, to_id, s))
        .smap(|e| sub_var_expr(from_id, to_id, e))
}

fn resymbolize_duplicated_loops_stmt(
    mut vars: BTreeSet<Name>,
    s: Stmt
) -> (BTreeSet<Name>, Stmt) {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let (var, body) = if vars.contains(&var) {
                let new_id = var.clone().with_new_sym();
                let body = body.smap(|s| sub_var_stmt(&var, &new_id, s));
                (new_id, body)
            } else {
                vars.insert(var.clone());
                (var, body)
            };
            let (vars, body) = body.smap_accum_l(vars, resymbolize_duplicated_loops_stmt);
            (vars, Stmt::For {var, lo, hi, step, body, par, i})
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::If {..} | Stmt::While {..} | Stmt::Alloc {..} | Stmt::Free {..} => {
            s.smap_accum_l(vars, resymbolize_duplicated_loops_stmt)
        }
    }
}

fn resymbolize_duplicated_loops(body: Vec<Stmt>) -> Vec<Stmt> {
    let (_, body) = body.smap_accum_l(BTreeSet::new(), |acc, s| {
        resymbolize_duplicated_loops_stmt(acc, s)
    });
    body
}

#[derive(Clone, Debug, PartialEq)]
struct DefUse {
    pub defs: BTreeSet<Name>,
    pub uses: BTreeSet<Name>
}

impl Default for DefUse {
    fn default() -> Self {
        DefUse {defs: BTreeSet::new(), uses: BTreeSet::new()}
    }
}

fn collect_uses_kernel_expr(
    mut acc: DefUse,
    e: &Expr
) -> DefUse {
    match e {
        Expr::Var {id, ..} => {
            acc.uses.insert(id.clone());
            acc
        },
        _ => e.sfold(acc, collect_uses_kernel_expr)
    }
}

fn collect_def_use_kernel_stmt(
    mut acc: DefUse,
    s: &Stmt
) -> DefUse {
    match s {
        Stmt::Definition {id, expr, ..} => {
            acc.defs.insert(id.clone());
            collect_uses_kernel_expr(acc, expr)
        },
        Stmt::For {var, body, ..} => {
            acc.defs.insert(var.clone());
            body.sfold(acc, collect_def_use_kernel_stmt)
        },
        _ => {
            let acc = s.sfold(acc, collect_def_use_kernel_stmt);
            s.sfold(acc, collect_uses_kernel_expr)
        }
    }
}


fn collect_def_use_stmt(
    acc: (Vec<DefUse>, BTreeSet<Name>),
    s: &Stmt
) -> (Vec<DefUse>, BTreeSet<Name>) {
    match s {
        Stmt::Definition {id, ..} => {
            let (def_uses, mut bound_vars) = acc;
            bound_vars.insert(id.clone());
            (def_uses, bound_vars)
        },
        Stmt::For {var, body, par, ..} => {
            let (mut def_uses, mut bound_vars) = acc;
            if par.is_parallel() {
                let mut du = DefUse::default();
                du.defs.insert(var.clone());
                def_uses.push(body.sfold(du, collect_def_use_kernel_stmt));
                (def_uses, bound_vars)
            } else {
                bound_vars.insert(var.clone());
                body.sfold((def_uses, bound_vars), collect_def_use_stmt)
            }
        },
        _ => s.sfold(acc, collect_def_use_stmt)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TempData {
    id: Name,
    ty: Type,
    expr: Expr,
    size: i64
}

struct TempDataEnv {
    temp_vars: BTreeSet<Name>,
    data: BTreeMap<Name, TempData>,
    par_structure: Vec<(Name, i64)>
}

impl TempDataEnv {
    pub fn new(temp_vars: BTreeSet<Name>) -> Self {
        TempDataEnv {temp_vars, data: BTreeMap::new(), par_structure: vec![]}
    }
}

fn collect_variable_temp_data(
    mut env: TempDataEnv,
    s: &Stmt
) -> CompileResult<TempDataEnv> {
    match s {
        Stmt::Definition {ty, id, i, ..} if env.temp_vars.contains(&id) => {
            let idx_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
            let init_expr = Expr::Int {v: 0, ty: idx_ty.clone(), i: i.clone()};
            let (idx_expr, size) = env.par_structure.iter()
                .rev()
                .fold((init_expr, 1), |acc, (id, dim)| {
                    let (idx_expr, acc_size) = acc;
                    let e = Expr::BinOp {
                        lhs: Box::new(idx_expr),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::BinOp {
                            lhs: Box::new(Expr::Var {
                                id: id.clone(), ty: idx_ty.clone(), i: i.clone()
                            }),
                            op: BinOp::Mul,
                            rhs: Box::new(Expr::Int {
                                v: acc_size, ty: idx_ty.clone(), i: i.clone()
                            }),
                            ty: idx_ty.clone(),
                            i: i.clone()
                        }),
                        ty: idx_ty.clone(),
                        i: i.clone()
                    };
                    (e, acc_size * dim)
                });
            let (ty, sz) = match ty {
                Type::Tensor {sz, ..} => {
                    let ptr_ty = Type::Pointer {
                        ty: Box::new(ty.clone()), count: size as usize
                    };
                    (ptr_ty, sz.clone())
                },
                _ => parir_compile_error!(i, "Cannot allocate temporary data \
                                              for non-tensor variable {id}.")?
            };
            let new_id = id.clone().with_new_sym();
            let expr = Expr::TensorAccess {
                target: Box::new(Expr::Var {
                    id: new_id.clone(), ty: ty.clone(), i: i.clone()
                }),
                idx: Box::new(idx_expr),
                ty: Type::Tensor {sz, shape: vec![]},
                i: i.clone()
            };
            let temp_data = TempData {
                id: new_id, ty, expr, size
            };
            env.data.insert(id.clone(), temp_data);
            Ok(env)
        },
        Stmt::For {var, body, par, ..} if par.is_parallel() => {
            env.par_structure.push((var.clone(), par.nthreads));
            let mut env = body.sfold_result(Ok(env), collect_variable_temp_data)?;
            env.par_structure.pop();
            Ok(env)
        },
        _ => s.sfold_result(Ok(env), collect_variable_temp_data)
    }
}

fn replace_variables_with_temporary_data_expr(
    env: &TempDataEnv,
    e: Expr
) -> Expr {
    match e {
        Expr::Var {id, ..} if env.data.contains_key(&id) => {
            env.data.get(&id).unwrap().expr.clone()
        },
        _ => e.smap(|e| replace_variables_with_temporary_data_expr(env, e))
    }
}

fn replace_variables_with_temporary_data(
    env: &TempDataEnv,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::Definition {id, expr, i, ..} if env.data.contains_key(&id) => {
            let entry = env.data.get(&id).unwrap();
            let expr = replace_variables_with_temporary_data_expr(env, expr);
            Stmt::Assign {dst: entry.expr.clone(), expr, i}
        },
        _ => {
            s.smap(|s| replace_variables_with_temporary_data(env, s))
                .smap(|e| replace_variables_with_temporary_data_expr(env, e))
        }
    }
}

fn generate_alloc_stmt(id: &Name, ty: Type, sz: i64) -> Stmt {
    let elem_ty = match ty {
        Type::Pointer {ty, ..} => *ty,
        _ => panic!("Cannot allocate temporary data for non-pointer type")
    };
    Stmt::Alloc {id: id.clone(), elem_ty, sz: sz as usize, i: Info::default()}
}

fn generate_dealloc_stmt(id: &Name) -> Stmt {
    Stmt::Free {id: id.clone(), i: Info::default()}
}

/// Allocate temporary data for local variables that end up being defined and used in separate
/// kernels after hoisting sequential loops.
fn allocate_temporary_data(
    params: &Vec<Param>,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    // Collect the names of all parameters - we use these as a starting point for keeping track of
    // the variables that are bound outside of kernel code. Such variables do not need to be stored
    // in temporary data.
    let param_ids = params.iter()
        .map(|Param {id, ..}| id.clone())
        .collect::<BTreeSet<Name>>();

    // Collect the def/use sets for each kernel.
    let acc = (vec![], param_ids);
    let (def_use_sets, bound) = body.sfold(acc, collect_def_use_stmt);

    // Find the variables that are used in a kernel in which they are not defined, excluding
    // variables that are bound outside kernel code (e.g., iteration variables of sequential
    // loops). These are the variables for which we need to allocate temporary data.
    let temp_vars = def_use_sets.into_iter()
        .map(|DefUse {defs, uses}| {
            uses.difference(&defs)
                .map(|n| n.clone())
                .collect::<BTreeSet<Name>>()
        })
        .flatten()
        .map(|n| n.clone())
        .collect::<BTreeSet<Name>>()
        .difference(&bound)
        .map(|n| n.clone())
        .collect::<BTreeSet<Name>>();

    // Collect information needed to allocate, deallocate, and use the temporary data associated
    // with each of the variables identified in the previous step.
    let env = body.sfold_result(Ok(TempDataEnv::new(temp_vars)), collect_variable_temp_data)?;

    // Update all uses of a variable, including its definition, to appropriately access the
    // temporary data allocated for it.
    let body = body.smap(|s| replace_variables_with_temporary_data(&env, s));

    // Generate the allocating and deallocating statements and insert them in the beginning and the
    // end of the body, respectively.
    let alloc = env.data.iter()
        .map(|(_, TempData {id: new_id, ty, size, ..})| {
            generate_alloc_stmt(new_id, ty.clone(), size.clone())
        });
    let dealloc = env.data.iter()
        .map(|(_, TempData {id: new_id, ..})| generate_dealloc_stmt(new_id));
    Ok(alloc.chain(body.into_iter())
        .chain(dealloc)
        .collect::<Vec<Stmt>>())
}

/// Restructure the code such that inter-block synchronizations are eliminated from the code. In
/// particular, we identify the synchronization points within parallel code and classify them based
/// on whether they require inter-block synchronization. Parallel reductions requiring inter-block
/// synchronization are transformed into two parts; one that runs in parallel across multiple
/// blocks, and a second part that runs within a block.
///
/// When we have inter-block synchronization points inside a sequential loop, where the iteration
/// order matters, the code is transformed by hoisting the sequential loop outside of the parallel
/// code. The code is updated such that statements execute in a correct order with respect to the
/// original program.
///
/// As part of the transformation, assignments to local variables are replaced with a declaration
/// when this is required to maintain correctness after moving the code. Further, temporary data is
/// allocated for representing local variables that are defined and used in separate kernels after
/// transforming the AST. These are performed to restore the AST to a valid state.
pub fn restructure_inter_block_synchronization(ast: Ast) -> CompileResult<Ast> {
    let Ast {fun: FunDef {id, params, body, i}, structs} = ast;
    // Insert a synchronization point at the end of each parallel for-loop, and determine for each
    // of them whether they require inter-block synchronization.
    let body = insert_synchronization_points(body);
    let par = par_tree::build_tree(&body);
    let body = classify_synchronization_points(&par, body);

    // Split up inter-block parallel reductions in two parts. In the first part, the full reduction
    // is split across the blocks, such that each block writes to its own temporary memory
    // location. Then, in the second part, we reduce these values to a single value, stored in the
    // original location.
    let body = split_inter_block_parallel_reductions(body)?;

    // Hoist sequential loops inside parallel code containing inter-block synchronization points
    // such that they occur outside of the parallel code, and restructure the code to ensure the
    // execution order remains valid.
    let body = hoist_inner_sequential_loops(&par, body)?;

    // Split parallel code on any remaining inter-block synchronization points, to ensure these are
    // only found at the end of a parallel for-loop.
    let body = split_inter_block_synchronizations(body);

    // Remove synchronization points that are unnecessary. A synchronization point is deemed
    // unnecessary either if it occurs outside of parallel code or if it occurs at the end of a
    // parallel for-loop.
    let body = eliminate_unnecessary_synchronization_points(body);

    // After hoisting, the code may be restructured such that the definition and the use(s) of a
    // local variable end up in separate parallel kernels. First, we promote assignments to
    // definitions, to properly handle repeated assignments to a local variable ending up in
    // separate kernels. Second, we allocate temporary data for storing local variables, when
    // these are defined and used in separate kernels.
    let body = allocate_temporary_data(&params, body)?;

    // After the above transformations, we may end up with repeated use of a parallel for-loop. To
    // make sure later transformations work as expected, we need to re-symbolize loop variables.
    let body = resymbolize_duplicated_loops(body);

    Ok(Ast {fun: FunDef {id, params, body, i}, structs})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder::*;

    fn for_(id: &str, n: i64, body: Vec<Stmt>) -> Stmt {
        let var = Name::new(id.to_string());
        for_loop(var, n, body)
    }

    fn stmts_str(stmts: &Vec<Stmt>) -> String {
        let (_, s) = pprint_iter(stmts.iter(), PrettyPrintEnv::new(), "\n");
        s
    }

    fn print_stmts(lhs: &Vec<Stmt>, rhs: &Vec<Stmt>) {
        let separator = str::repeat("=", 10);
        println!("{0}\n{1}\n{2}", stmts_str(&lhs), separator, stmts_str(&rhs));
    }

    fn assert_sync(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let body = insert_synchronization_points(body);
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    fn assert_classify(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let par = par_tree::build_tree(&body);
        let body = classify_synchronization_points(&par, body);
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    fn assert_hoist(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let par = par_tree::build_tree(&body);
        let body = hoist_inner_sequential_loops(&par, body).unwrap();
        let body = eliminate_unnecessary_synchronization_points(body);
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    #[test]
    fn empty_sync_points() {
        assert_sync(vec![], vec![]);
    }

    #[test]
    fn single_par_loop_sync_points() {
        let s = vec![for_("x", 10, vec![])];
        let expected = vec![
            for_("x", 10, vec![]),
            sync_point(false)
        ];
        assert_sync(s, expected);
    }

    #[test]
    fn subsequent_par_loops_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 32, vec![]),
                for_("z", 32, vec![])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 32, vec![]),
                sync_point(false),
                for_("z", 32, vec![]),
                sync_point(false)
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_loops_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(false),
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_loops_classify() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(false),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(true),
                ])
            ])
        ];
        assert_classify(body, expected);
    }

    #[test]
    fn par_seq_par_loops_local_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(var("q"), int(3))]),
                    sync_point(true),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(var("q"), int(3))]),
                    sync_point(true)
                ])
            ])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_seq_par_loops_inter_block_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(4)),
                for_("y", 0, vec![
                    assign(var("b"), int(0)),
                    for_("z", 2048, vec![assign(var("c"), int(3))]),
                    sync_point(false),
                    assign(var("d"), int(1))
                ]),
                assign(var("e"), int(2))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(4))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(var("b"), int(0)),
                    for_("z", 2048, vec![assign(var("c"), int(3))])
                ]),
                for_("x", 10, vec![assign(var("d"), int(1))])
            ]),
            for_("x", 10, vec![assign(var("e"), int(2))])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_in_cond_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                if_cond(
                    vec![],
                    vec![for_("y", 10, vec![])]
                )
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                if_cond(
                    vec![],
                    vec![
                        for_("y", 10, vec![]),
                        sync_point(false)
                    ]
                )
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_in_while_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(false)
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_in_while_classify() {
        let body = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(true)
                ])
            ])
        ];
        assert_classify(body, expected)
    }

    #[test]
    fn par_seq_par_seq_par_sync_points() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![])
                        ])
                    ])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(false)
                        ])
                    ]),
                    sync_point(false)
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_seq_par_classify() {
        let identified_sync_points = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(false)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(true)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        assert_classify(identified_sync_points, expected);
    }

    #[test]
    fn par_seq_par_seq_par_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(1)),
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(var("b"), int(2))]),
                            sync_point(true)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(1))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(var("b"), int(2))]),
                            sync_point(true)
                        ])
                    ])
                ])
            ])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_seq_par_seq_par_double_inter_block_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(1)),
                for_("y", 0, vec![
                    assign(var("b"), int(2)),
                    for_("z", 10, vec![
                        assign(var("c"), int(3)),
                        for_("w", 0, vec![
                            assign(var("d"), int(4)),
                            for_("v", 2048, vec![assign(var("e"), int(5))]),
                            sync_point(false),
                            assign(var("f"), int(6))
                        ]),
                        assign(var("g"), int(7))
                    ]),
                    sync_point(false),
                    assign(var("h"), int(8))
                ]),
                assign(var("i"), int(9))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(1))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(var("b"), int(2)),
                    for_("z", 10, vec![assign(var("c"), int(3))])
                ]),
                for_("w", 0, vec![
                    for_("x", 10, vec![
                        for_("z", 10, vec![
                            assign(var("d"), int(4)),
                            for_("v", 2048, vec![assign(var("e"), int(5))]),
                        ])
                    ]),
                    for_("x", 10, vec![
                        for_("z", 10, vec![assign(var("f"), int(6))]),
                    ]),
                ]),
                for_("x", 10, vec![
                    for_("z", 10, vec![assign(var("g"), int(7))]),
                ]),
                for_("x", 10, vec![assign(var("h"), int(8))]),
            ]),
            for_("x", 10, vec![assign(var("i"), int(9))])
        ];
        assert_hoist(body, expected)
    }
}
