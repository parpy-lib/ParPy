use super::ast::*;
use crate::option;
use crate::prickle_compile_error;
use crate::prickle_internal_error;
use crate::utils::reduce;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::smap::SMapAccum;

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReductionScope {
    Warp, Block, Cluster
}

fn classify_reduction(
    opts: &option::CompileOptions,
    nthreads: i64,
    tpb: i64,
    i: &Info
) -> CompileResult<ReductionScope> {
    if nthreads > tpb {
        if opts.backend != option::CompileBackend::Cuda {
            prickle_internal_error!(i, "Found inter-block reduction after \
                                        inter-block transformation in non-CUDA backend.")?
        };
        if !opts.use_cuda_thread_block_clusters {
            prickle_internal_error!(i, "Found multi-block reduction after \
                                        inter-block transformation, but thread \
                                        block clusters were not enabled.\n\
                                        To enable it, set the 'use_cuda_thread_block_clusters' \
                                        field of the compilation options to True.")?
        };
        let nblocks = (nthreads + tpb - 1) / tpb;
        if nblocks > opts.max_thread_blocks_per_cluster {
            let msg = format!(
                "Found multi-block reduction over {0} blocks, but the maximum \
                 number of thread blocks per cluster is set to {1}.\n\
                 To increase the limit, set the 'max_thread_blocks_per_cluster' field \
                 of the compilation options to a larger value.",
                nblocks, opts.max_thread_blocks_per_cluster
            );
            prickle_internal_error!(i, "{}", msg)?
        };
        Ok(ReductionScope::Cluster)
    } else if nthreads > 32 {
        Ok(ReductionScope::Block)
    } else {
        Ok(ReductionScope::Warp)
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
                Type::Scalar {sz} => {
                    Ok((*lhs, non_short_circuiting_op(op), *rhs, sz, i))
                },
                _ => prickle_compile_error!(i, "Expected the result of reduction \
                                                to be a scalar value, found {0}",
                                                ty.pprint_default())
            }
        },
        Expr::Convert {e, ..} => extract_bin_op(*e),
        _ => {
            prickle_compile_error!(i, "RHS of reduction statement should be a \
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
                Expr::Var {..} | Expr::ArrayAccess {..} => {
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
                        prickle_compile_error!(i, "{}", msg)
                    }
                },
                _ => {
                    prickle_compile_error!(i, "Left-hand side of reduction must \
                                               be a variable or tensor access.")
                }
            }
        } else {
            prickle_compile_error!(i, "Reduction for-loop statement must be an \
                                       assignment.")
        }
    } else {
        prickle_compile_error!(i, "Reduction for-loop must contain a single \
                                   statement.")
    }
}

fn reduction_op_neutral_element(
    op: &BinOp,
    sz: &ElemSize,
    i: &Info
) -> CompileResult<Expr> {
    match reduce::neutral_element(op, sz, &i) {
        Some(literal) => Ok(literal),
        None => {
            let op = Expr::print_binop(op, &Type::Void, &Type::Void).unwrap();
            prickle_compile_error!(i, "Parallel reductions not supported for operator {op}.")
        },
    }
}

struct LoopStruct {
    pub var_ty: Type,
    pub var: Name,
    pub init: Expr,
    pub cond: Expr,
    pub incr: Expr,
    pub lhs: Expr,
    pub op: BinOp,
    pub rhs: Expr,
    pub ne: Expr,
    pub nthreads: i64,
    pub tpb: i64,
}

struct ReduceEnv {
    pub for_loop: LoopStruct,
    pub temp_var: Expr,
    pub temp_id: Name,
    pub res_ty: Type,
    pub i: Info
}

fn generate_main_reduction_loop(env: &ReduceEnv, acc: &mut Vec<Stmt>) {
    acc.push(Stmt::Definition {
        ty: env.res_ty.clone(),
        id: env.temp_id.clone(),
        expr: env.for_loop.ne.clone(),
        i: env.i.clone()
    });
    let temp_assign = Stmt::Assign {
        dst: env.temp_var.clone(),
        expr: Expr::BinOp {
            lhs: Box::new(env.temp_var.clone()),
            op: env.for_loop.op.clone(),
            rhs: Box::new(env.for_loop.rhs.clone()),
            ty: env.res_ty.clone(),
            i: env.i.clone()
        },
        i: env.i.clone()
    };
    acc.push(Stmt::For {
        var_ty: env.for_loop.var_ty.clone(),
        var: env.for_loop.var.clone(),
        init: env.for_loop.init.clone(),
        cond: env.for_loop.cond.clone(),
        incr: env.for_loop.incr.clone(),
        body: vec![temp_assign],
        i: env.i.clone()
    });
    acc.push(Stmt::Synchronize {scope: SyncScope::Block, i: env.i.clone()});
}

fn generate_warp_reduction(env: &ReduceEnv, acc: &mut Vec<Stmt>) {
    acc.push(Stmt::WarpReduce {
        value: env.temp_var.clone(),
        op: env.for_loop.op.clone(),
        int_ty: env.for_loop.var_ty.clone(),
        res_ty: env.res_ty.clone(),
        i: env.i.clone()
    });
}

fn generate_shared_memory_zero_initialized(env: &ReduceEnv, acc: &mut Vec<Stmt>) -> Expr {
    let int_ty = env.for_loop.var_ty.clone();
    let i = &env.i;
    let stemp_id = Name::sym_str("stemp");
    let shared_var = Expr::Var {
        id: stemp_id.clone(), ty: env.res_ty.clone(), i: i.clone()
    };
    acc.push(Stmt::AllocShared {
        elem_ty: env.res_ty.clone(), id: stemp_id, sz: 32, i: i.clone()
    });
    let is_first_warp = Expr::BinOp {
        lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: int_ty.clone(), i: i.clone()}),
        op: BinOp::Lt,
        rhs: Box::new(Expr::Int {v: 32, ty: int_ty.clone(), i: i.clone()}),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: i.clone()
    };
    acc.push(Stmt::If {
        cond: is_first_warp,
        thn: vec![Stmt::Assign {
            dst: Expr::ArrayAccess {
                target: Box::new(shared_var.clone()),
                idx: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: int_ty.clone(), i: i.clone()}),
                ty: env.res_ty.clone(),
                i: i.clone()
            },
            expr: env.for_loop.ne.clone(),
            i: i.clone()
        }],
        els: vec![],
        i: i.clone()
    });
    acc.push(Stmt::Synchronize {scope: SyncScope::Block, i: env.i.clone()});
    shared_var
}

fn generate_shared_memory_exchange(
    env: &ReduceEnv,
    shared_var: &Expr,
    acc: &mut Vec<Stmt>
) {
    let i = &env.i;
    let int_ty = env.for_loop.var_ty.clone();
    let warp_op = |op| Expr::BinOp {
        lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: int_ty.clone(), i: i.clone()}),
        op,
        rhs: Box::new(Expr::Int {v: 32, ty: int_ty.clone(), i: i.clone()}),
        ty: int_ty.clone(),
        i: i.clone()
    };
    let is_first_thread_of_warp = Expr::BinOp {
        lhs: Box::new(warp_op(BinOp::Rem)),
        op: BinOp::Eq,
        rhs: Box::new(Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()}),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: i.clone()
    };
    acc.push(Stmt::If {
        cond: is_first_thread_of_warp,
        thn: vec![Stmt::Assign {
            dst: Expr::ArrayAccess {
                target: Box::new(shared_var.clone()),
                idx: Box::new(warp_op(BinOp::Div)),
                ty: env.res_ty.clone(),
                i: i.clone()
            },
            expr: env.temp_var.clone(),
            i: i.clone()
        }],
        els: vec![],
        i: i.clone()
    });
    acc.push(Stmt::Synchronize {scope: SyncScope::Block, i: env.i.clone()});
    acc.push(Stmt::Assign {
        dst: env.temp_var.clone(),
        expr: Expr::ArrayAccess {
            target: Box::new(shared_var.clone()),
            idx: Box::new(warp_op(BinOp::Rem)),
            ty: env.res_ty.clone(),
            i: i.clone()
        },
        i: i.clone()
    });
}

fn generate_block_reduction(env: &ReduceEnv, acc: &mut Vec<Stmt>) -> Expr {
    let shared_var = generate_shared_memory_zero_initialized(env, acc);
    generate_shared_memory_exchange(env, &shared_var, acc);
    generate_warp_reduction(&env, acc);
    shared_var
}

fn generate_cluster_reduction(
    env: &ReduceEnv,
    shared_var: Expr,
    acc: &mut Vec<Stmt>
) {
    let i = &env.i;
    let int_ty = env.for_loop.var_ty.clone();
    let block_idx_in_cluster = Expr::BinOp {
        lhs: Box::new(env.for_loop.init.clone()),
        op: BinOp::Div,
        rhs: Box::new(Expr::Int {
            v: env.for_loop.tpb as i128, ty: int_ty.clone(), i: i.clone()
        }),
        ty: int_ty.clone(),
        i: i.clone()
    };
    let blocks_per_cluster = env.for_loop.nthreads / env.for_loop.tpb;
    acc.push(Stmt::ClusterReduce {
        block_idx: block_idx_in_cluster,
        shared_var: shared_var,
        temp_var: env.temp_var.clone(),
        blocks_per_cluster: blocks_per_cluster as i128,
        op: env.for_loop.op.clone(),
        int_ty: int_ty.clone(),
        res_ty: env.res_ty.clone(),
        i: env.i.clone()
    });
}

// We write the result stored in the temporary variable to the target. If this is a parallel
// reduction over a whole cluster, we only run this on the first block of the cluster.
// cluster, we only run it for the first block of each cluster.
fn generate_reduction_write_result(
    env: &ReduceEnv,
    scope: ReductionScope,
    acc: &mut Vec<Stmt>
) {
    let write_result = Stmt::Assign {
        dst: env.for_loop.lhs.clone(),
        expr: Expr::BinOp {
            lhs: Box::new(env.for_loop.lhs.clone()),
            op: env.for_loop.op.clone(),
            rhs: Box::new(env.temp_var.clone()),
            ty: env.res_ty.clone(),
            i: env.i.clone()
        },
        i: env.i.clone()
    };
    if scope == ReductionScope::Cluster {
        let int_ty = &env.for_loop.var_ty;
        let int_lit = |v: i64| Expr::Int {
            v: v as i128, ty: int_ty.clone(), i: env.i.clone()
        };
        let cluster_block_idx = Expr::BinOp {
            lhs: Box::new(env.for_loop.init.clone()),
            op: BinOp::Div,
            rhs: Box::new(int_lit(env.for_loop.tpb)),
            ty: int_ty.clone(),
            i: env.i.clone()
        };
        // If we are assigning to a global array, we only want one block to write to avoid data
        // races. However, if the result is stored in a variable, we have to update it across all
        // blocks.
        if let Expr::ArrayAccess {..} = &env.for_loop.lhs {
            let is_first_block_of_cluster = Expr::BinOp {
                lhs: Box::new(cluster_block_idx),
                op: BinOp::Eq,
                rhs: Box::new(int_lit(0)),
                ty: int_ty.clone(),
                i: env.i.clone()
            };
            acc.push(Stmt::If {
                cond: is_first_block_of_cluster,
                thn: vec![write_result],
                els: vec![],
                i: env.i.clone()
            });
        } else {
            acc.push(write_result);
        }
    } else {
        acc.push(write_result);
    }
    match scope {
        ReductionScope::Warp => (),
        ReductionScope::Block => {
            acc.push(Stmt::Synchronize {scope: SyncScope::Block, i: env.i.clone()});
        },
        ReductionScope::Cluster => {
            acc.push(Stmt::Synchronize {scope: SyncScope::Cluster, i: env.i.clone()});
        }
    }
}

fn generate_parallel_reduction(
    for_loop: LoopStruct,
    result_sz: ElemSize,
    i: Info,
    scope: ReductionScope
) -> Stmt {
    let temp_id = Name::sym_str("t");
    let res_ty = Type::Scalar {sz: result_sz};
    let temp_var = Expr::Var {
        id: temp_id.clone(), ty: res_ty.clone(), i: i.clone()
    };
    let env = ReduceEnv {
        for_loop, temp_var, temp_id, res_ty, i
    };

    let mut stmts = vec![];
    generate_main_reduction_loop(&env, &mut stmts);
    generate_warp_reduction(&env, &mut stmts);

    // If the parallel reduction involves more than one warp, we perform a reduction among the
    // warps of a thread block.
    if scope == ReductionScope::Block || scope == ReductionScope::Cluster {
        let shared_var = generate_block_reduction(&env, &mut stmts);

        // If the parallel reduction is performed across multiple blocks of a cluster, we perform a
        // reduction among the blocks to ensure they agree on a value.
        if scope == ReductionScope::Cluster {
            generate_cluster_reduction(&env, shared_var, &mut stmts);
        }
    }

    generate_reduction_write_result(&env, scope, &mut stmts);

    Stmt::Scope {body: stmts, i: env.i.clone()}
}

fn expand_parallel_reduction_stmt(
    opts: &option::CompileOptions,
    s: Stmt
) -> CompileResult<Stmt> {
    match s {
        Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i} => {
            let kind = classify_reduction(opts, nthreads, tpb, &i)?;
            let (lhs, op, rhs, sz, i) = extract_reduction_operands(body, &i)?;
            let ne = reduction_op_neutral_element(&op, &sz, &i)?;
            let loop_contents = LoopStruct {
                var_ty, var, init, cond, incr, lhs, op, rhs, ne, nthreads, tpb
            };
            Ok(generate_parallel_reduction(loop_contents, sz, i, kind))
        },
        _ => s.smap_result(|s| expand_parallel_reduction_stmt(opts, s))
    }
}

fn expand_parallel_reductions_stmts(
    opts: &option::CompileOptions,
    stmts: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.smap_result(|s| expand_parallel_reduction_stmt(opts, s))
}

fn expand_parallel_reductions_top(
    opts: &option::CompileOptions,
    t: Top
) -> CompileResult<Top> {
    match t {
        Top::KernelFunDef {attrs, id, params, body} => {
            let body = expand_parallel_reductions_stmts(opts, body)?;
            Ok(Top::KernelFunDef {attrs, id, params, body})
        },
        _ => Ok(t)
    }
}

pub fn expand_parallel_reductions(
    opts: &option::CompileOptions,
    ast: Ast
) -> CompileResult<Ast> {
    ast.smap_result(|t| expand_parallel_reductions_top(opts, t))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::option::*;

    #[test]
    fn classify_inter_block_reduction_non_cuda_backend() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Metal;
        assert_error_matches(
            classify_reduction(&opts, 2048, 1024, &i()),
            r"inter-block reduction.*non-CUDA backend"
        );
    }

    #[test]
    fn classify_inter_block_reduction_clusters_disabled() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = false;
        assert_error_matches(
            classify_reduction(&opts, 2048, 1024, &i()),
            r"thread block clusters.*not enabled"
        );
    }

    #[test]
    fn classify_inter_block_reduction_insufficient_cluster_count() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        opts.max_thread_blocks_per_cluster = 8;
        assert_error_matches(
            classify_reduction(&opts, 2048, 128, &i()),
            r"maximum number of thread blocks per cluster is set to 8"
        );
    }

    #[test]
    fn classify_cluster_reduction() {
        let mut opts = CompileOptions::default();
        opts.backend = CompileBackend::Cuda;
        opts.use_cuda_thread_block_clusters = true;
        opts.max_thread_blocks_per_cluster = 16;
        assert_eq!(classify_reduction(&opts, 2048, 128, &i()), Ok(ReductionScope::Cluster));
    }

    #[test]
    fn classify_block_reduction() {
        let opts = CompileOptions::default();
        assert_eq!(classify_reduction(&opts, 1024, 1024, &i()), Ok(ReductionScope::Block));
    }

    #[test]
    fn classify_warp_reduction() {
        let opts = CompileOptions::default();
        assert_eq!(classify_reduction(&opts, 32, 32, &i()), Ok(ReductionScope::Warp));
    }
}
