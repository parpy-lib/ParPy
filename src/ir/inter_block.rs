use super::ast::*;
use super::par_tree;
use super::target_constraints::TargetConstraint;
use crate::option;
use crate::par;
use crate::parpy_compile_error;
use crate::option::CompileOptions;
use crate::utils::ast::ScalarSizes;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::reduce;
use crate::utils::smap::{SFlatten, SFold, SMapAccum};
use crate::utils::substitute::SubVars;

use std::collections::{BTreeMap, BTreeSet};

fn collect_host_functions(
    mapping: &BTreeMap<Name, TargetConstraint>
) -> BTreeSet<Name> {
    mapping.iter()
        .filter(|(_, cls)| match cls {
            TargetConstraint::HostOnly => true,
            _ => false
        })
        .map(|(id, _)| id.clone())
        .collect::<BTreeSet<Name>>()
}

fn insert_synchronization_points_stmt(
    mut acc: Vec<Stmt>,
    s: Stmt,
    host_functions: &BTreeSet<Name>
) -> Vec<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let is_par = par.is_parallel();
            let body = insert_synchronization_points(body, &host_functions);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i: i.clone()});
            if is_par {
                acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i});
            }
            acc
        },
        Stmt::Expr {
            e: Expr::Call {id, args, par, ty, i: ei}, i
        } if host_functions.contains(&id) => {
            acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i: i.clone()});
            acc.push(Stmt::Expr {e: Expr::Call {id, args, par, ty, i: ei}, i: i.clone()});
            acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i});
            acc
        },
        Stmt::Expr {e: Expr::PyCallback {id, args, ty, i: ei}, i} => {
            acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i: i.clone()});
            acc.push(Stmt::Expr {e: Expr::PyCallback {id, args, ty, i: ei}, i: i.clone()});
            acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i});
            acc
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::If {..} | Stmt::While {..} | Stmt::Expr {..} | Stmt::Return {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.sflatten(acc, |acc, s| {
                insert_synchronization_points_stmt(acc, s, &host_functions)
            })
        },
    }
}

fn insert_synchronization_points(
    stmts: Vec<Stmt>,
    host_functions: &BTreeSet<Name>
) -> Vec<Stmt> {
    stmts.sflatten(vec![], |acc, s| {
        insert_synchronization_points_stmt(acc, s, &host_functions)
    })
}

fn determine_cuda_cluster_size(opts: &option::CompileOptions) -> Option<i64> {
    match opts.backend {
        option::CompileBackend::Cuda if opts.use_cuda_thread_block_clusters =>
            Some(opts.max_thread_blocks_per_cluster),
        _ => None
    }
}

pub fn classify_parallelism(
    opts: &option::CompileOptions,
    par: &LoopPar
) -> SyncPointKind {
    let cluster_size = determine_cuda_cluster_size(opts).unwrap_or(0);
    if par.nthreads > 0 && par.nthreads <= par.tpb {
        SyncPointKind::BlockLocal
    } else if cluster_size > 0 && par.nthreads > 0 && par.nthreads <= cluster_size * par.tpb {
        SyncPointKind::BlockCluster
    } else {
        SyncPointKind::InterBlock
    }
}

fn classify_synchronization_points_par_stmt(
    node: &par_tree::ParNode,
    opts: &option::CompileOptions,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::If {..} |
        Stmt::While {..} | Stmt::Expr {..} | Stmt::Return {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.sflatten(acc, |acc, s| {
                classify_synchronization_points_par_stmt(node, opts, acc, s)
            })
        },
        Stmt::SyncPoint {i, ..} => {
            let prev_stmt_sync_kind = match acc.last() {
                Some(Stmt::For {par, ..}) => classify_parallelism(opts, par),
                _ => SyncPointKind::InterBlock
            };

            // When the synchronization statement is found in the innermost level of parallelism,
            // its kind depends on the preceding for-loop.
            let s = if node.innermost_parallelism() {
                Stmt::SyncPoint {kind: prev_stmt_sync_kind, i}
            } else {
                Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i}
            };
            acc.push(s);
            acc
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let node = node.children.get(&var).unwrap_or(node);
            let body = classify_synchronization_points_par_stmts(node, opts, body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
            acc
        },
    }
}

fn classify_synchronization_points_par_stmts(
    par: &par_tree::ParNode,
    opts: &option::CompileOptions,
    stmts: Vec<Stmt>
) -> Vec<Stmt> {
    stmts.sflatten(vec![], |acc, s| {
        classify_synchronization_points_par_stmt(par, opts, acc, s)
    })
}

fn classify_synchronization_points_stmt(
    t: &par_tree::ParTree,
    opts: &option::CompileOptions,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = if let Some(node) = t.roots.get(&var) {
                classify_synchronization_points_par_stmts(&node, opts, body)
            } else {
                body.smap(|s| classify_synchronization_points_stmt(t, opts, s))
            };
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
        Stmt::SyncPoint {..} | Stmt::While {..} | Stmt::Expr {..} | Stmt::If {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.smap(|s| classify_synchronization_points_stmt(t, opts, s))
        }
    }
}

fn classify_synchronization_points(
    par: &par_tree::ParTree,
    opts: &option::CompileOptions,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    body.smap(|s| classify_synchronization_points_stmt(par, opts, s))
}

fn extract_neutral_element(
    op: &BinOp, sz: &ElemSize, i: &Info
) -> CompileResult<Expr> {
    match reduce::neutral_element(op, sz, i) {
        Some(ne) => Ok(ne),
        None => parpy_compile_error!(i, "Reduction operation {0} has unknown \
                                           neutral element.",
                                           op.pprint_default())
    }
}

fn inner_multi_block_reduce_loop(
    loop_idx: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    nblocks: i128,
    tpb: i64,
    block_idx: Name,
    lhs: Expr,
    op: BinOp,
    rhs: Expr,
    i: &Info
) -> Stmt {
    let int_ty = lo.get_type().clone();
    let int_lit = |v| Expr::Int {v, ty: int_ty.clone(), i: i.clone()};
    let block = Expr::Var {id: block_idx, ty: int_ty.clone(), i: i.clone()};
    let block_ofs = Expr::BinOp {
        lhs: Box::new(Expr::BinOp {
            lhs: Box::new(Expr::BinOp {
                lhs: Box::new(hi.clone()),
                op: BinOp::Sub,
                rhs: Box::new(lo.clone()),
                ty: int_ty.clone(), i: i.clone()
            }),
            op: BinOp::Add,
            rhs: Box::new(Expr::BinOp {
                lhs: Box::new(int_lit(nblocks)),
                op: BinOp::Sub,
                rhs: Box::new(int_lit(1)),
                ty: int_ty.clone(), i: i.clone()
            }),
            ty: int_ty.clone(), i: i.clone()
        }),
        op: BinOp::FloorDiv,
        rhs: Box::new(int_lit(nblocks)),
        ty: int_ty.clone(), i: i.clone()
    };
    let lo_expr = Expr::BinOp {
        lhs: Box::new(lo),
        op: BinOp::Add,
        rhs: Box::new(Expr::BinOp {
            lhs: Box::new(block_ofs.clone()),
            op: BinOp::Mul,
            rhs: Box::new(block.clone()),
            ty: int_ty.clone(), i: i.clone()
        }),
        ty: int_ty.clone(), i: i.clone()
    };
    let hi_expr = Expr::BinOp {
        lhs: Box::new(Expr::Convert {
            e: Box::new(Expr::BinOp {
                lhs: Box::new(lo_expr.clone()),
                op: BinOp::Add,
                rhs: Box::new(block_ofs),
                ty: int_ty.clone(), i: i.clone()
            }),
            ty: int_ty.clone(),
            i: i.clone()
        }),
        op: BinOp::Min,
        rhs: Box::new(Expr::Convert {
            e: Box::new(hi),
            ty: int_ty.clone(),
            i: i.clone()
        }),
        ty: int_ty.clone(), i: i.clone()
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
        var: loop_idx, lo: lo_expr, hi: hi_expr, step: step,
        body: vec![assign],
        par: LoopPar {nthreads: tpb, reduction: true, tpb, unroll: false},
        i: i.clone()
    }
}

fn outer_multi_block_reduce_loop(
    inner_loop: Stmt,
    temp_id: Name,
    ne: Expr,
    var: Name,
    int_ty: Type,
    nblocks: usize,
    tpb: i64,
    i: &Info
) -> Stmt {
    let init_stmt = Stmt::Definition {
        ty: ne.get_type().clone(),
        id: temp_id,
        expr: ne,
        i: i.clone()
    };
    Stmt::For {
        var,
        lo: Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()},
        hi: Expr::Int {v: nblocks as i128, ty: int_ty.clone(), i: i.clone()},
        step: 1,
        body: vec![init_stmt, inner_loop],
        par: LoopPar {nthreads: nblocks as i64, reduction: false, tpb, unroll: false},
        i: i.clone()
    }
}

fn single_block_reduce_loop(
    var: Name,
    nblocks: usize,
    lhs: Expr,
    op: BinOp,
    rhs: Expr,
    int_ty: Type,
    tpb: i64,
    i: &Info
) -> Stmt {
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
    // For this reduction, we use one block with at most as many threads as in the default number
    // of threads per block. Note that we ignore the user-configured threads per block in this part
    // because more threads should be beneficial performance-wise.
    let par = LoopPar {
        nthreads: i64::min(nblocks as i64, par::DEFAULT_TPB),
        reduction: true,
        tpb,
        unroll: false
    };
    Stmt::For {
        var,
        lo: Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()},
        hi: Expr::Int {v: nblocks as i128, ty: int_ty.clone(), i: i.clone()},
        step: 1, body: vec![assign], par, i: i.clone()
    }
}

/// The 'and' and 'or' operators are short-circuiting in both C and Python. However, we do not want
/// to generate short-circuiting code, as the warp-level reductions will get stuck unless all
/// threads have the same value (because some threads short-circuit, thereby ignoring the warp sync
/// intrinsic call). To work around this, we use the bitwise operations, which are equivalent for
/// boolean values assuming they are encoded as 0 or 1 (not necessarily true in C).
fn non_short_circuiting_op(op: BinOp) -> BinOp {
    match op {
        BinOp::And => BinOp::BitAnd,
        BinOp::Or => BinOp::BitOr,
        _ => op
    }
}

fn extract_bin_op(
    expr: Expr
) -> CompileResult<(Expr, BinOp, Expr, ElemSize, Info)> {
    let i = expr.get_info();
    match expr {
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            match ty {
                Type::Tensor {shape, sz} if shape.is_empty() => {
                    Ok((*lhs, non_short_circuiting_op(op), *rhs, sz, i))
                },
                _ => parpy_compile_error!(i, "Expected the result of reduction \
                                                to be a scalar value, found {0}",
                                                ty.pprint_default())
            }
        },
        Expr::Convert {e, ..} => extract_bin_op(*e),
        _ => {
            parpy_compile_error!(i, "RHS of reduction statement should be a \
                                       binary operation.")
        }
    }
}

fn unwrap_convert(e: &Expr) -> Expr {
    match e {
        Expr::Convert {e, ..} => unwrap_convert(e),
        _ => e.clone()
    }
}

pub fn extract_reduction_operands(
    mut body: Vec<Stmt>,
    i: &Info
) -> CompileResult<(Expr, BinOp, Expr, ElemSize, Info)> {
    // The reduction loop body must contain a single statement.
    if body.len() == 1 {
        // The single statement must be a single (re)assignment.
        if let Stmt::Assign {dst, expr, ..} = body.remove(0) {
            // The right-hand side should be a binary operation, so we extract its constituents.
            let (lhs, op, rhs, sz, i) = extract_bin_op(expr)?;
            // The destination of the assignment must either be a variable or an access.
            match dst {
                Expr::Var {..} | Expr::TensorAccess {..} => {
                    // The assignment destination must be equal to the left-hand side of the
                    // reduction operation.
                    if dst == unwrap_convert(&lhs) {
                        Ok((dst, op, rhs, sz, i))
                    } else {
                        let msg = format!(
                            "Invalid reduction. Left-hand side of binary \
                             operation {0} is not equal to the assignment \
                             target {1}.",
                             lhs.pprint_default(), dst.pprint_default()
                        );
                        parpy_compile_error!(i, "{}", msg)
                    }
                },
                _ => {
                    parpy_compile_error!(i, "Left-hand side of reduction must \
                                               be a variable or tensor access.")
                }
            }
        } else {
            parpy_compile_error!(i, "Reduction for-loop statement must be an \
                                       assignment.")
        }
    } else {
        parpy_compile_error!(i, "Reduction for-loop must contain a single \
                                   statement.")
    }
}

fn split_inter_block_parallel_reductions_stmt(
    opts: &option::CompileOptions,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    let is_parallel_inter_block_reduction = |par: &LoopPar| {
        match classify_parallelism(opts, par) {
            SyncPointKind::InterBlock if par.nthreads > 0 && par.reduction => true,
            _ => false
        }
    };
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if is_parallel_inter_block_reduction(&par) => {
            if par.nthreads % par.tpb == 0 {
                let nblocks = (par.nthreads / par.tpb) as usize;
                let (lhs, op, rhs, sz, i) = extract_reduction_operands(body, &i)?;
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
                let int_ty = lo.get_type().clone();
                let l1 = inner_multi_block_reduce_loop(
                    var, lo, hi, step, nblocks as i128, par.tpb,
                    id.clone(), temp_access.clone(), op.clone(), rhs, &i
                );
                acc.push(outer_multi_block_reduce_loop(
                    l1, temp_id, ne, id.clone(), int_ty.clone(), nblocks, par.tpb, &i
                ));
                acc.push(Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i: i.clone()});
                acc.push(single_block_reduce_loop(
                    id, nblocks, lhs, op, temp_access, int_ty, par.tpb, &i
                ));
                Ok(acc)
            } else {
                parpy_compile_error!(i, "Multi-block reductions must use a \
                                           multiple of {0} threads.",
                                           par.tpb)
            }
        },
        _ => {
            s.sflatten_result(acc, |acc, s| {
                split_inter_block_parallel_reductions_stmt(opts, acc, s)
            })
        }
    }
}

fn split_inter_block_parallel_reductions(
    opts: &option::CompileOptions,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    body.sflatten_result(vec![], |acc, s| {
        split_inter_block_parallel_reductions_stmt(opts, acc, s)
    })
}

fn is_inter_block_sync_point(s: &Stmt) -> bool {
    match s {
        Stmt::SyncPoint {kind: SyncPointKind::InterBlock, ..} => true,
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
    par: LoopPar,
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
            _ => parpy_compile_error!(&i, "Internal error when hoisting \
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
    par: LoopPar,
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
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Expr {..} |
        Stmt::Return {..} | Stmt::SyncPoint {..} | Stmt::Alloc {..} |
        Stmt::AllocShared {..} | Stmt::Free {..} => {
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
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Expr {..} |
        Stmt::Return {..} | Stmt::SyncPoint {..} | Stmt::Alloc {..} |
        Stmt::AllocShared {..} | Stmt::Free {..} => {
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
            let body = body.sflatten(vec![], split_inter_block_synchronization_kernel);
            let mut bodies = body.split_inclusive(is_inter_block_sync_point)
                .map(|chunk| {
                    Stmt::For {
                        var: var.clone(), lo: lo.clone(), hi: hi.clone(), step,
                        body: chunk.to_vec(), par: par.clone(), i: i.clone()}
                })
                .collect::<Vec<Stmt>>();
            acc.append(&mut bodies);
            acc
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} | Stmt::Expr {..} | Stmt::Return {..} | Stmt::SyncPoint {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.sflatten(acc, split_inter_block_synchronization_kernel)
        }
    }
}

fn split_inter_block_synchronization_stmt(acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {ref par, ..} if par.is_parallel() => {
            split_inter_block_synchronization_kernel(acc, s.clone())
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::For {..} | Stmt::While {..} | Stmt::If {..} | Stmt::Expr {..} |
        Stmt::Return {..} | Stmt::Alloc {..} | Stmt::AllocShared {..} |
        Stmt::Free {..} => {
            s.sflatten(acc, split_inter_block_synchronization_stmt)
        }
    }
}

fn split_inter_block_synchronizations(body: Vec<Stmt>) -> Vec<Stmt> {
    body.sflatten(vec![], split_inter_block_synchronization_stmt)
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
        Stmt::SyncPoint {kind: SyncPointKind::InterBlock, ..} if !env.in_parallel => (),
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
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Expr {..} |
        Stmt::Return {..} | Stmt::SyncPoint {..} | Stmt::Alloc {..} |
        Stmt::AllocShared {..} | Stmt::Free {..} => {
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
    stmts.sflatten(vec![], |acc, s| {
        eliminate_unnecessary_synchronization_points_stmt(env.clone(), acc, s)
    })
}

fn eliminate_unnecessary_synchronization_points(body: Vec<Stmt>) -> Vec<Stmt> {
    let env = SyncPointEnv::new();
    eliminate_unnecessary_synchronization_points_stmts(env, body)
}

fn remove_empty_loop_nests_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = remove_empty_loop_nests(body);
            if !body.is_empty() {
                acc.push(Stmt::For {var, lo, hi, step, body, par, i});
            }
            acc
        },
        _ => s.sflatten(acc, remove_empty_loop_nests_stmt)
    }
}

fn remove_empty_loop_nests(body: Vec<Stmt>) -> Vec<Stmt> {
    body.sflatten(vec![], remove_empty_loop_nests_stmt)
}

fn resymbolize_duplicated_loops_stmt(
    mut vars: BTreeSet<Name>,
    s: Stmt
) -> (BTreeSet<Name>, Stmt) {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let (var, body) = if vars.contains(&var) {
                let new_id = var.clone().with_new_sym();
                let sub_env = vec![(var.clone(), new_id.clone())]
                    .into_iter()
                    .collect::<BTreeMap<Name, Name>>();
                let body = body.smap(|s| s.sub_vars(&sub_env));
                (new_id, body)
            } else {
                vars.insert(var.clone());
                (var, body)
            };
            let (vars, body) = body.smap_accum_l(vars, resymbolize_duplicated_loops_stmt);
            (vars, Stmt::For {var, lo, hi, step, body, par, i})
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::If {..} | Stmt::While {..} | Stmt::Expr {..} | Stmt::Return {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
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
    ss: &ScalarSizes,
    mut env: TempDataEnv,
    s: &Stmt
) -> CompileResult<TempDataEnv> {
    match s {
        Stmt::Definition {ty, id, i, ..} if env.temp_vars.contains(&id) => {
            let idx_ty = Type::Tensor {sz: ss.int.clone(), shape: vec![]};
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
                    (e, acc_size * *dim as i128)
                });
            let (ty, res_ty) = match ty {
                Type::Tensor {..} => {
                    let ptr_ty = Type::Pointer {ty: Box::new(ty.clone())};
                    (ptr_ty, ty.clone())
                },
                _ => parpy_compile_error!(i, "Cannot allocate temporary data \
                                                for non-tensor variable {id}.")?
            };
            let new_id = id.clone().with_new_sym();
            let expr = Expr::TensorAccess {
                target: Box::new(Expr::Var {
                    id: new_id.clone(), ty: ty.clone(), i: i.clone()
                }),
                idx: Box::new(idx_expr),
                ty: res_ty,
                i: i.clone()
            };
            let temp_data = TempData {
                id: new_id, ty, expr, size: size as i64
            };
            env.data.insert(id.clone(), temp_data);
            Ok(env)
        },
        Stmt::For {var, body, par, ..} if par.is_parallel() => {
            env.par_structure.push((var.clone(), par.nthreads));
            let mut env = body.sfold_result(Ok(env), |acc, var| {
                collect_variable_temp_data(ss, acc, var)
            })?;
            env.par_structure.pop();
            Ok(env)
        },
        _ => s.sfold_result(Ok(env), |acc, var| {
            collect_variable_temp_data(ss, acc, var)
        })
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
    opts: &CompileOptions,
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
    let ss = ScalarSizes::from_opts(opts);
    let env = body.sfold_result(Ok(TempDataEnv::new(temp_vars)), |acc, var| {
        collect_variable_temp_data(&ss, acc, var)
    })?;

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

fn innermost_body_contains_host_call(
    body: &Vec<Stmt>,
    host_functions: &BTreeSet<Name>
) -> bool {
    match &body[..] {
        [Stmt::For {body, ..}] => {
            innermost_body_contains_host_call(body, &host_functions)
        }
        [Stmt::Expr {e: Expr::Call {id, ..}, ..}] => host_functions.contains(id),
        [Stmt::Expr {e: Expr::PyCallback {..}, ..}] => true,
        _ => false,
    }
}

fn remove_parallelism_of_host_loop_nests_stmt(
    s: Stmt,
    host_functions: &BTreeSet<Name>
) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            let par = if innermost_body_contains_host_call(&body, &host_functions) {
                LoopPar::default()
            } else {
                par
            };
            let body = remove_parallelism_of_host_loop_nests(body, &host_functions);
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        _ => s.smap(|s| remove_parallelism_of_host_loop_nests_stmt(s, &host_functions))
    }
}

/// Removes the parallelism of loop nests whose innermost workload consists of a statement that
/// should run on the host. This can either be an external function that is declared to run on the
/// host, or a callback function (which run in Python code on the host).
fn remove_parallelism_of_host_loop_nests(
    body: Vec<Stmt>,
    host_functions: &BTreeSet<Name>
) -> Vec<Stmt> {
    body.smap(|s| remove_parallelism_of_host_loop_nests_stmt(s, &host_functions))
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
pub fn restructure_inter_block_synchronization(
    ast: Ast,
    mapping: &BTreeMap<Name, TargetConstraint>,
    opts: &option::CompileOptions
) -> CompileResult<Ast> {
    // NOTE: We only apply this to the main function definition, which is the last one in the list.
    // The others are assumed to contain no parallelism.
    let FunDef {id, params, body, res_ty, i} = ast.main;

    // We construct a set of the names of all host functions based on the target mapping results.
    // Host functions are those where the target constraints were identified as 'HostOnly'.
    let host_functions = collect_host_functions(&mapping);

    // Insert a synchronization point at the end of each parallel for-loop, and determine for each
    // of them whether they require inter-block synchronization.
    let body = insert_synchronization_points(body, &host_functions);
    let par = par_tree::build_tree(&body);
    let body = classify_synchronization_points(&par, opts, body);

    // Split up inter-block parallel reductions in two parts. In the first part, the full reduction
    // is split across the blocks, such that each block writes to its own temporary memory
    // location. Then, in the second part, we reduce these values to a single value, stored in the
    // original location.
    let body = split_inter_block_parallel_reductions(opts, body)?;

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

    // Removes any empty loop nests produced as a result of the inter-block transformation.
    let body = remove_empty_loop_nests(body);

    // After hoisting, the code may be restructured such that the definition and the use(s) of a
    // local variable end up in separate parallel kernels. First, we promote assignments to
    // definitions, to properly handle repeated assignments to a local variable ending up in
    // separate kernels. Second, we allocate temporary data for storing local variables, when
    // these are defined and used in separate kernels.
    let body = allocate_temporary_data(&opts, &params, body)?;

    // After the above transformations, we may end up with repeated use of a parallel for-loop. To
    // make sure later transformations work as expected, we need to re-symbolize loop variables.
    let body = resymbolize_duplicated_loops(body);

    // Removes the parallelism of loop nests that contain calls to host code. This includes a call
    // to a host external or a callback function.
    let body = remove_parallelism_of_host_loop_nests(body, &host_functions);

    Ok(Ast {main: FunDef {id, params, body, res_ty, i}, ..ast})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::ir::ast_builder::*;
    use crate::option::*;

    #[test]
    fn cuda_cluster_size_correct_opts() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        opts.max_thread_blocks_per_cluster = 16;
        assert_eq!(determine_cuda_cluster_size(&opts), Some(16));
    }

    #[test]
    fn cuda_cluster_size_invalid_opts() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Metal;
        assert!(determine_cuda_cluster_size(&opts).is_none());
    }

    #[test]
    fn classify_block_local() {
        let opts = CompileOptions::default();
        let p = LoopPar::default().threads(128).unwrap();
        assert_eq!(classify_parallelism(&opts, &p), SyncPointKind::BlockLocal);
    }

    #[test]
    fn classify_block_cluster() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        opts.max_thread_blocks_per_cluster = 8;
        let p = LoopPar::default()
            .tpb(256).unwrap()
            .threads(1024).unwrap();
        assert_eq!(classify_parallelism(&opts, &p), SyncPointKind::BlockCluster);
    }

    #[test]
    fn classify_inter_block_fails_to_fit_cluster() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        opts.max_thread_blocks_per_cluster = 8;
        let p = LoopPar::default()
            .tpb(256).unwrap()
            .threads(4096).unwrap();
        assert_eq!(classify_parallelism(&opts, &p), SyncPointKind::InterBlock);
    }

    #[test]
    fn classify_inter_block() {
        let opts = CompileOptions::default();
        let p = LoopPar::default()
            .tpb(256).unwrap()
            .threads(1024).unwrap();
        assert_eq!(classify_parallelism(&opts, &p), SyncPointKind::InterBlock);
    }

    #[test]
    fn bool_and_non_short_circuiting() {
        assert_eq!(non_short_circuiting_op(BinOp::And), BinOp::BitAnd);
    }

    #[test]
    fn bool_or_non_short_circuiting() {
        assert_eq!(non_short_circuiting_op(BinOp::Or), BinOp::BitOr);
    }

    #[test]
    fn extract_bin_op_invalid_form() {
        let e = unop(UnOp::Sub, int(1, None));
        assert_error_matches(extract_bin_op(e), r"RHS of reduction.*should be.*binary operation");
    }

    #[test]
    fn extract_bin_op_non_scalar_result() {
        let e = binop(int(1, None), BinOp::Add, int(2, None), Some(Type::Void));
        assert_error_matches(extract_bin_op(e), r"Expected.*a scalar value");
    }

    #[test]
    fn unwrap_convert_expr() {
        let e = Expr::Convert {
            e: Box::new(int(1, None)),
            ty: scalar(ElemSize::U16),
            i: i()
        };
        assert_eq!(unwrap_convert(&e), int(1, None));
    }

    #[test]
    fn extract_reduction_operands_invalid_body_size() {
        let s = vec![
            assign(var("x", scalar(ElemSize::I32)), int(1, None)),
            assign(var("y", scalar(ElemSize::F32)), float(1.0, None)),
        ];
        assert_error_matches(
            extract_reduction_operands(s, &i()),
            "must contain a single statement"
        );
    }

    #[test]
    fn extract_reduction_operands_invalid_statement() {
        let s = vec![
            definition(scalar(ElemSize::I32), id("x"), int(1, None))
        ];
        assert_error_matches(
            extract_reduction_operands(s, &i()),
            "must be an assignment"
        );
    }

    #[test]
    fn extract_reduction_operands_invalid_lhs() {
        let lhs = int(1, None);
        let rhs = binop(var("x", scalar(ElemSize::I32)), BinOp::Add, int(1, None), None);
        let s = vec![assign(lhs, rhs)];
        assert_error_matches(
            extract_reduction_operands(s, &i()),
            "Left-hand side of reduction"
        );
    }

    #[test]
    fn extract_reduction_operands_invalid_destination() {
        let lhs = var("x", scalar(ElemSize::I32));
        let rhs = binop(var("y", scalar(ElemSize::I32)), BinOp::Add, int(1, None), None);
        assert_error_matches(
            extract_reduction_operands(vec![assign(lhs, rhs)], &i()),
            "Left-hand side of binary operation.*not equal to the assignment target"
        );
    }

    #[test]
    fn extract_reduction_operands_ok() {
        let lhs = var("x", scalar(ElemSize::I32));
        let rhs = binop(lhs.clone(), BinOp::Add, int(1, None), None);
        let s = vec![assign(lhs.clone(), rhs.clone())];
        let (dst, op, rhs_bop, sz, _) = extract_reduction_operands(s, &i()).unwrap();
        assert_eq!(dst, lhs);
        assert_eq!(op, BinOp::Add);
        assert_eq!(rhs_bop, int(1, None));
        assert_eq!(sz, ElemSize::I32);
    }

    fn for_(id: &str, n: i64, body: Vec<Stmt>) -> Stmt {
        let var = Name::new(id.to_string());
        for_loop(var, n, body)
    }

    fn uvar(id: &str) -> Expr {
        var(id, scalar(ElemSize::Bool))
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
        let body = insert_synchronization_points(body, &BTreeSet::new());
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    fn assert_classify(body: Vec<Stmt>, expected: Vec<Stmt>, opts: Option<option::CompileOptions>) {
        let par = par_tree::build_tree(&body);
        let opts = opts.unwrap_or(option::CompileOptions::default());
        let body = classify_synchronization_points(&par, &opts, body);
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
            sync_point(SyncPointKind::InterBlock)
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
                sync_point(SyncPointKind::InterBlock),
                for_("z", 32, vec![]),
                sync_point(SyncPointKind::InterBlock)
            ]),
            sync_point(SyncPointKind::InterBlock)
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
                    sync_point(SyncPointKind::InterBlock),
                ])
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_loops_classify() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(SyncPointKind::InterBlock),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(SyncPointKind::BlockLocal),
                ])
            ])
        ];
        assert_classify(body, expected, None);
    }

    #[test]
    fn par_seq_par_loops_local_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(uvar("q"), int(3, None))]),
                    sync_point(SyncPointKind::BlockLocal),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(uvar("q"), int(3, None))]),
                    sync_point(SyncPointKind::BlockLocal)
                ])
            ])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_seq_par_loops_inter_block_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(uvar("a"), int(4, None)),
                for_("y", 0, vec![
                    assign(uvar("b"), int(0, None)),
                    for_("z", 2048, vec![assign(uvar("c"), int(3, None))]),
                    sync_point(SyncPointKind::InterBlock),
                    assign(uvar("d"), int(1, None))
                ]),
                assign(uvar("e"), int(2, None))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(uvar("a"), int(4, None))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(uvar("b"), int(0, None)),
                    for_("z", 2048, vec![assign(uvar("c"), int(3, None))])
                ]),
                for_("x", 10, vec![assign(uvar("d"), int(1, None))])
            ]),
            for_("x", 10, vec![assign(uvar("e"), int(2, None))])
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
                        sync_point(SyncPointKind::InterBlock)
                    ]
                )
            ]),
            sync_point(SyncPointKind::InterBlock)
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
                    sync_point(SyncPointKind::InterBlock)
                ])
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_in_while_classify() {
        let body = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(SyncPointKind::InterBlock)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(SyncPointKind::BlockLocal)
                ])
            ])
        ];
        assert_classify(body, expected, None)
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
                            sync_point(SyncPointKind::InterBlock)
                        ])
                    ]),
                    sync_point(SyncPointKind::InterBlock)
                ])
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn cluster_sync_point_classify() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 2048, vec![]),
                sync_point(SyncPointKind::InterBlock)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        let expected_fst = vec![
            for_("x", 10, vec![
                for_("y", 2048, vec![]),
                sync_point(SyncPointKind::InterBlock)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        let expected_snd = vec![
            for_("x", 10, vec![
                for_("y", 2048, vec![]),
                sync_point(SyncPointKind::BlockCluster)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        assert_classify(body.clone(), expected_fst, None);

        let mut opts = option::CompileOptions::default();
        opts.backend = option::CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        assert_classify(body, expected_snd, Some(opts))
    }

    #[test]
    fn cluster_extended_size_sync_point_classify() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 16384, vec![]),
                sync_point(SyncPointKind::InterBlock)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        let expected_fst = vec![
            for_("x", 10, vec![
                for_("y", 16384, vec![]),
                sync_point(SyncPointKind::InterBlock)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        let expected_snd = vec![
            for_("x", 10, vec![
                for_("y", 16384, vec![]),
                sync_point(SyncPointKind::BlockCluster)
            ]),
            sync_point(SyncPointKind::InterBlock)
        ];
        assert_classify(body.clone(), expected_fst.clone(), None);
        let mut opts = option::CompileOptions::default();
        opts.backend = option::CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        assert_classify(body.clone(), expected_fst, Some(opts.clone()));
        opts.max_thread_blocks_per_cluster = 16;
        assert_classify(body, expected_snd, Some(opts.clone()))
    }

    #[test]
    fn par_seq_par_seq_par_classify() {
        let identified_sync_points = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(SyncPointKind::InterBlock)
                        ])
                    ]),
                    sync_point(SyncPointKind::InterBlock)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(SyncPointKind::BlockLocal)
                        ])
                    ]),
                    sync_point(SyncPointKind::InterBlock)
                ])
            ])
        ];
        assert_classify(identified_sync_points, expected, None)
    }

    #[test]
    fn par_seq_par_seq_par_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(uvar("a"), int(1, None)),
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(uvar("b"), int(2, None))]),
                            sync_point(SyncPointKind::BlockLocal)
                        ])
                    ]),
                    sync_point(SyncPointKind::InterBlock)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(uvar("a"), int(1, None))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(uvar("b"), int(2, None))]),
                            sync_point(SyncPointKind::BlockLocal)
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
                assign(uvar("a"), int(1, None)),
                for_("y", 0, vec![
                    assign(uvar("b"), int(2, None)),
                    for_("z", 10, vec![
                        assign(uvar("c"), int(3, None)),
                        for_("w", 0, vec![
                            assign(uvar("d"), int(4, None)),
                            for_("v", 2048, vec![assign(uvar("e"), int(5, None))]),
                            sync_point(SyncPointKind::InterBlock),
                            assign(uvar("f"), int(6, None))
                        ]),
                        assign(uvar("g"), int(7, None))
                    ]),
                    sync_point(SyncPointKind::InterBlock),
                    assign(uvar("h"), int(8, None))
                ]),
                assign(uvar("i"), int(9, None))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(uvar("a"), int(1, None))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(uvar("b"), int(2, None)),
                    for_("z", 10, vec![assign(uvar("c"), int(3, None))])
                ]),
                for_("w", 0, vec![
                    for_("x", 10, vec![
                        for_("z", 10, vec![
                            assign(uvar("d"), int(4, None)),
                            for_("v", 2048, vec![assign(uvar("e"), int(5, None))]),
                        ])
                    ]),
                    for_("x", 10, vec![
                        for_("z", 10, vec![assign(uvar("f"), int(6, None))]),
                    ]),
                ]),
                for_("x", 10, vec![
                    for_("z", 10, vec![assign(uvar("g"), int(7, None))]),
                ]),
                for_("x", 10, vec![assign(uvar("h"), int(8, None))]),
            ]),
            for_("x", 10, vec![assign(uvar("i"), int(9, None))])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn remove_loop_with_empty_body() {
        let body = vec![
            for_("x", 10, vec![])
        ];
        assert_eq!(remove_empty_loop_nests(body), vec![]);
    }

    #[test]
    fn remove_nested_empty_loop() {
        let body = vec![
            for_("x", 10, vec![for_("y", 10, vec![])])
        ];
        assert_eq!(remove_empty_loop_nests(body), vec![]);
    }

    #[test]
    fn remove_inner_nested_empty_loop() {
        let body = vec![
            for_("x", 10, vec![for_("y", 10, vec![
                assign(uvar("a"), int(0, None))
            ])]),
            for_("z", 10, vec![for_("w", 10, vec![])])
        ];
        let expected = vec![
            for_("x", 10, vec![for_("y", 10, vec![
                assign(uvar("a"), int(0, None))
            ])])
        ];
        assert_eq!(remove_empty_loop_nests(body), expected);
    }
}
