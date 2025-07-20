use crate::gpu::ast::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::*;

fn generate_warp_reduction(
    value: Expr, op: BinOp, int_ty: Type, res_ty: Type, i: Info
) -> Stmt {
    let iter_id = Name::sym_str("i");
    let iter_var = Expr::Var {id: iter_id.clone(), ty: int_ty.clone(), i: i.clone()};
    // We use the call expression node to represent the use of a CUDA-specific intrinsic.
    let rhs = Expr::Call {
        id: "__shfl_xor_sync".to_string(),
        args: vec![
            Expr::Int {v: 0xFFFFFFFF, ty: int_ty.clone(), i: i.clone()},
            value.clone(),
            iter_var.clone()
        ],
        ty: value.get_type().clone(),
        i: i.clone()
    };
    let sync_stmt = Stmt::Assign {
        dst: value.clone(),
        expr: Expr::BinOp {
            lhs: Box::new(value),
            op: op,
            rhs: Box::new(rhs),
            ty: res_ty,
            i: i.clone()
        },
        i: i.clone()
    };
    let int_lit = |v| {
        Expr::Int {v, ty: int_ty.clone(), i: i.clone()}
    };
    let cond_expr = Expr::BinOp {
        lhs: Box::new(iter_var.clone()),
        op: BinOp::Gt,
        rhs: Box::new(int_lit(0)),
        ty: Type::Boolean,
        i: i.clone()
    };
    let incr_expr = Expr::BinOp {
        lhs: Box::new(iter_var),
        op: BinOp::Div,
        rhs: Box::new(int_lit(2)),
        ty: int_ty.clone(),
        i: i.clone()
    };
    Stmt::For {
        var_ty: int_ty.clone(), var: iter_id, init: int_lit(16),
        cond: cond_expr, incr: incr_expr, body: vec![sync_stmt],
        i: i.clone()
    }
}

struct ClusterData {
    pub block_idx: Expr,
    pub shared_var: Expr,
    pub temp_var: Expr,
    pub blocks_per_cluster: i128,
    pub op: BinOp,
    pub int_ty: Type,
    pub res_ty: Type,
    pub i: Info
}

fn generate_cluster_init_shared_memory(
    data: &ClusterData,
    block_smem: Expr,
    acc: &mut Vec<Stmt>
) {
    let i = &data.i;
    let int_ty = &data.int_ty;
    let is_first_thread_of_block = Expr::BinOp {
        lhs: Box::new(Expr::ThreadIdx {
            dim: Dim::X, ty: int_ty.clone(), i: i.clone()
        }),
        op: BinOp::Eq,
        rhs: Box::new(Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()}),
        ty: int_ty.clone(),
        i: i.clone()
    };
    acc.push(Stmt::If {
        cond: is_first_thread_of_block,
        thn: vec![Stmt::Assign {
            dst: block_smem,
            expr: data.temp_var.clone(),
            i: data.i.clone()
        }],
        els: vec![],
        i: data.i.clone()
    });
    acc.push(Stmt::Synchronize {scope: SyncScope::Cluster, i: data.i.clone()});
}

fn generate_cluster_iterative_reduction(
    data: &ClusterData,
    block_smem: Expr,
    acc: &mut Vec<Stmt>
) {
    let i = &data.i;
    let int_ty = &data.int_ty;
    let loop_id = Name::sym_str("i");
    let loop_var = Expr::Var {
        id: loop_id.clone(), ty: int_ty.clone(), i: i.clone()
    };
    let mut loop_body = vec![];

    // Create a pointer referring to the shared memory belonging to the other thread block.
    let other_block_idx = Expr::BinOp {
        lhs: Box::new(Expr::BinOp {
            lhs: Box::new(data.block_idx.clone()),
            op: BinOp::Add,
            rhs: Box::new(loop_var.clone()),
            ty: int_ty.clone(),
            i: i.clone()
        }),
        op: BinOp::Rem,
        rhs: Box::new(Expr::Int {v: data.blocks_per_cluster, ty: int_ty.clone(), i: i.clone()}),
        ty: int_ty.clone(),
        i: i.clone()
    };
    let other_smem_id = Name::sym_str("other_block_smem");
    let smem_ty = Type::Pointer {ty: Box::new(data.res_ty.clone()), mem: MemSpace::Device};
    let other_smem_var = Expr::Var {
        id: other_smem_id.clone(), ty: smem_ty.clone(), i: data.i.clone()
    };
    loop_body.push(Stmt::Definition {
        ty: smem_ty.clone(),
        id: other_smem_id,
        expr: Expr::Call {
            id: "this_cluster().map_shared_rank".to_string(),
            args: vec![data.shared_var.clone(), other_block_idx.clone()],
            ty: smem_ty,
            i: data.i.clone()
        },
        i: i.clone()
    });

    // Compute the result after applying the reduction operator on the shared memory of the current
    // block and that of the other thread block in the cluster.
    let combined_smem_data = Expr::BinOp {
        lhs: Box::new(block_smem.clone()),
        op: data.op.clone(),
        rhs: Box::new(Expr::ArrayAccess {
            target: Box::new(other_smem_var.clone()),
            idx: Box::new(other_block_idx.clone()),
            ty: data.res_ty.clone(),
            i: i.clone()
        }),
        ty: data.res_ty.clone(),
        i: i.clone()
    };
    let is_first_thread_of_block = Expr::BinOp {
        lhs: Box::new(Expr::ThreadIdx {
            dim: Dim::X, ty: int_ty.clone(), i: i.clone()
        }),
        op: BinOp::Eq,
        rhs: Box::new(Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()}),
        ty: int_ty.clone(),
        i: i.clone()
    };
    loop_body.push(Stmt::If {
        cond: is_first_thread_of_block,
        thn: vec![Stmt::Assign {
            dst: data.temp_var.clone(),
            expr: combined_smem_data,
            i: i.clone()
        }],
        els: vec![],
        i: i.clone()
    });
    loop_body.push(Stmt::Synchronize {scope: SyncScope::Cluster, i: data.i.clone()});

    // Update the local shared memory value of each block. We store the intermediate results in the
    // temporary variable and synchronize to avoid data races.
    loop_body.push(Stmt::Assign {
        dst: block_smem.clone(),
        expr: data.temp_var.clone(),
        i: i.clone()
    });
    loop_body.push(Stmt::Synchronize {scope: SyncScope::Cluster, i: data.i.clone()});
    
    let int_lit = |v| {
        Expr::Int {v, ty: int_ty.clone(), i: i.clone()}
    };
    let init = int_lit(data.blocks_per_cluster / 2);
    let cond = Expr::BinOp {
        lhs: Box::new(loop_var.clone()),
        op: BinOp::Gt,
        rhs: Box::new(int_lit(0)),
        ty: int_ty.clone(),
        i: i.clone()
    };
    let incr = Expr::BinOp {
        lhs: Box::new(loop_var.clone()),
        op: BinOp::Div,
        rhs: Box::new(int_lit(2)),
        ty: int_ty.clone(),
        i: i.clone()
    };
    acc.push(Stmt::For {
        var_ty: data.int_ty.clone(),
        var: loop_id,
        init, cond, incr,
        body: loop_body,
        i: i.clone()
    });
}

fn generate_cluster_temp_assignment(
    data: &ClusterData,
    block_smem: Expr,
    acc: &mut Vec<Stmt>
) {
    acc.push(Stmt::Assign {
        dst: data.temp_var.clone(),
        expr: block_smem,
        i: data.i.clone()
    });
}

fn generate_cluster_reduction(data: ClusterData) -> Vec<Stmt> {
    let block_smem = Expr::ArrayAccess {
        target: Box::new(data.shared_var.clone()),
        idx: Box::new(data.block_idx.clone()),
        ty: data.res_ty.clone(),
        i: data.i.clone()
    };
    let mut body = vec![];
    generate_cluster_init_shared_memory(&data, block_smem.clone(), &mut body);
    generate_cluster_iterative_reduction(&data, block_smem.clone(), &mut body);
    generate_cluster_temp_assignment(&data, block_smem.clone(), &mut body);
    body
}

fn expand_parallel_reductions_stmt(mut acc: Vec<Stmt>, stmt: Stmt) -> Vec<Stmt> {
    match stmt {
        Stmt::WarpReduce {value, op, int_ty, res_ty, i} => {
            acc.push(generate_warp_reduction(value, op, int_ty, res_ty, i));
            acc
        },
        Stmt::ClusterReduce {
            block_idx, shared_var, temp_var, blocks_per_cluster, op, int_ty, res_ty, i
        } => {
            let data = ClusterData {
                block_idx, shared_var, temp_var, blocks_per_cluster, op,
                int_ty, res_ty, i
            };
            acc.append(&mut generate_cluster_reduction(data));
            acc
        },
        _ => stmt.sflatten(acc, expand_parallel_reductions_stmt)
    }
}

fn expand_parallel_reductions_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.sflatten(vec![], expand_parallel_reductions_stmt)
}

fn expand_parallel_reductions_top(t: Top) -> Top {
    match t {
        Top::KernelFunDef {attrs, id, params, body} => {
            let body = expand_parallel_reductions_stmts(body);
            Top::KernelFunDef {attrs, id, params, body}
        },
        _ => t
    }
}

pub fn expand_parallel_reductions(ast: Ast) -> Ast {
    ast.smap(expand_parallel_reductions_top)
}
