use super::ast::*;
use super::free_vars;
use super::par::{GpuMap, GpuMapping};
use crate::prickle_compile_error;
use crate::option;
use crate::ir::ast as ir_ast;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

#[derive(Debug)]
struct CodegenEnv<'a> {
    gpu_mapping: BTreeMap<Name, GpuMapping>,
    struct_fields: BTreeMap<Name, Vec<Field>>,
    opts: &'a option::CompileOptions,
}

fn from_ir_type(ty: ir_ast::Type) -> Type {
    match ty {
        ir_ast::Type::Tensor {sz, shape} if shape.is_empty() => Type::Scalar {sz},
        ir_ast::Type::Tensor {sz, ..} => {
            Type::Pointer {ty: Box::new(Type::Scalar {sz}), mem: MemSpace::Device}
        },
        ir_ast::Type::Pointer {ty, ..} => {
            Type::Pointer {ty: Box::new(from_ir_type(*ty)), mem: MemSpace::Device}
        },
        ir_ast::Type::Struct {id} => Type::Struct {id},
        ir_ast::Type::Void => Type::Void,
    }
}

fn from_ir_expr(e: ir_ast::Expr) -> CompileResult<Expr> {
    let ty = from_ir_type(e.get_type().clone());
    match e {
        ir_ast::Expr::Var {id, i, ..} => Ok(Expr::Var {id, ty, i}),
        ir_ast::Expr::Bool {v, i, ..} => Ok(Expr::Bool {v, ty, i}),
        ir_ast::Expr::Int {v, i, ..} => Ok(Expr::Int {v, ty, i}),
        ir_ast::Expr::Float {v, i, ..} => Ok(Expr::Float {v, ty, i}),
        ir_ast::Expr::UnOp {op, arg, i, ..} => {
            let arg = Box::new(from_ir_expr(*arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        ir_ast::Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = Box::new(from_ir_expr(*lhs)?);
            let rhs = Box::new(from_ir_expr(*rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        ir_ast::Expr::IfExpr {cond, thn, els, i, ..} => {
            let cond = Box::new(from_ir_expr(*cond)?);
            let thn = Box::new(from_ir_expr(*thn)?);
            let els = Box::new(from_ir_expr(*els)?);
            Ok(Expr::IfExpr {cond, thn, els, ty, i})
        },
        ir_ast::Expr::StructFieldAccess {target, label, i, ..} => {
            let target = Box::new(from_ir_expr(*target)?);
            Ok(Expr::StructFieldAccess {target, label, ty, i})
        },
        ir_ast::Expr::TensorAccess {target, idx, i, ..} => {
            let target = Box::new(from_ir_expr(*target)?);
            let idx = Box::new(from_ir_expr(*idx)?);
            Ok(Expr::ArrayAccess {target, idx, ty, i})
        },
        ir_ast::Expr::Call {id, args, i, ..} => {
            let args = args.into_iter()
                .map(from_ir_expr)
                .collect::<CompileResult<Vec<Expr>>>()?;
            Ok(Expr::Call {id, args, ty, i})
        },
        ir_ast::Expr::Convert {e, ..} => {
            let e = Box::new(from_ir_expr(*e)?);
            Ok(Expr::Convert {e, ty})
        },
    }
}

fn generate_kernel_name(fun_id: &Name, loop_var_id: &Name) -> Name {
    let s = format!("{0}_{1}", fun_id.get_str(), loop_var_id.get_str());
    Name::new(s).with_new_sym()
}

fn remainder_if_shared_dimension(
    idx: Expr, dim_tot: i64, v: i64, multiplier: i64
) -> Expr {
    if dim_tot > v {
        let ty = idx.get_type().clone();
        let int_expr = |v| Expr::Int {v, ty: ty.clone(), i: idx.get_info()};
        Expr::BinOp {
            lhs: Box::new(Expr::BinOp {
                lhs: Box::new(idx.clone()),
                op: BinOp::Div,
                rhs: Box::new(int_expr(multiplier as i128)),
                ty: ty.clone(), i: idx.get_info()
            }),
            op: BinOp::Rem,
            rhs: Box::new(int_expr(v as i128)),
            ty: ty.clone(), i: idx.get_info()
        }
    } else {
        idx
    }
}

fn determine_loop_bounds(
    par: Option<(LaunchArgs, GpuMap)>,
    var: Name,
    lo: ir_ast::Expr,
    hi: ir_ast::Expr,
    step_size: i64
) -> CompileResult<(Expr, Expr, Expr)> {
    let init = from_ir_expr(lo)?;
    let ty = init.get_type().clone();
    let i = init.get_info();
    let var_e = Expr::Var {id: var, ty: ty.clone(), i: i.clone()};
    let cond_op = if step_size > 0 { BinOp::Lt } else { BinOp::Gt };
    let cond = Expr::BinOp {
        lhs: Box::new(var_e.clone()),
        op: cond_op,
        rhs: Box::new(from_ir_expr(hi)?),
        ty: Type::Boolean,
        i: i.clone()
    };
    let step = Expr::Int {
        v: step_size as i128, ty: Type::Scalar {sz: ElemSize::I64}, i: i.clone()
    };
    let fn_incr = |v| Expr::BinOp {
        lhs: Box::new(var_e),
        op: BinOp::Add,
        rhs: Box::new(Expr::BinOp {
            lhs: Box::new(step.clone()),
            op: BinOp::Mul,
            rhs: Box::new(Expr::Int {
                v: v as i128, ty: Type::Scalar {sz: ElemSize::I64}, i: i.clone()
            }),
            ty: ty.clone(), i: i.clone()
        }),
        ty: ty.clone(), i: i.clone()
    };
    match par {
        Some((grid, GpuMap::Thread {n, dim, mult})) => {
            let tot = grid.threads.get_dim(&dim);
            let idx = Expr::ThreadIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n, mult);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add,
                rhs: Box::new(Expr::BinOp {
                    lhs: Box::new(step.clone()),
                    op: BinOp::Mul,
                    rhs: Box::new(rhs),
                    ty: ty.clone(), i: i.clone()
                }),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
        },
        Some((grid, GpuMap::Block {n, dim, mult})) => {
            let tot = grid.blocks.get_dim(&dim);
            let idx = Expr::BlockIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n, mult);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add,
                rhs: Box::new(Expr::BinOp {
                    lhs: Box::new(step.clone()),
                    op: BinOp::Mul,
                    rhs: Box::new(rhs),
                    ty: ty.clone(), i: i.clone()
                }),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
        },
        Some((grid, GpuMap::ThreadBlock {n, nthreads, nblocks, dim})) => {
            let tot_blocks = grid.blocks.get_dim(&dim);
            let idx = Expr::BinOp {
                lhs: Box::new(Expr::BinOp {
                    lhs: Box::new(Expr::BlockIdx {
                        dim, ty: ty.clone(), i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: nthreads as i128, ty: ty.clone(), i: i.clone()
                    }),
                    ty: ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Add,
                rhs: Box::new(Expr::ThreadIdx {dim, ty: ty.clone(), i: i.clone()}),
                ty: ty.clone(),
                i: i.clone()
            };
            let rhs = remainder_if_shared_dimension(idx.clone(), tot_blocks, nblocks, 1);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            // If the total number of threads we are using is not evenly divisible by the number of
            // threads per block, we insert an additional condition to ensure only the intended
            // threads run the code inside the for-loop.
            let cond = if nthreads * nblocks != n {
                Expr::BinOp {
                    lhs: Box::new(cond),
                    op: BinOp::And,
                    rhs: Box::new(Expr::BinOp {
                        lhs: Box::new(idx.clone()),
                        op: BinOp::Lt,
                        rhs: Box::new(Expr::Int {
                            v: n as i128, ty: ty.clone(), i: i.clone()
                        }),
                        ty: ty.clone(), i: i.clone()
                    }),
                    ty: ty.clone(), i: i.clone()
                }
            } else {
                cond
            };
            Ok((init, cond, fn_incr(n)))
        },
        None => Ok((init, cond, fn_incr(1)))
    }
}

fn subtract_from_grid(grid: LaunchArgs, m: &GpuMap) -> LaunchArgs {
    match m {
        GpuMap::Thread {n, dim, ..} => {
            let threads = grid.threads.get_dim(&dim) / n;
            grid.with_threads_dim(dim, threads)
        },
        GpuMap::Block {n, dim, ..} => {
            let blocks = grid.blocks.get_dim(&dim) / n;
            grid.with_blocks_dim(dim, blocks)
        },
        GpuMap::ThreadBlock {nthreads, nblocks, dim, ..} => {
            let threads = grid.threads.get_dim(&dim) / nblocks;
            let blocks = grid.blocks.get_dim(&dim) / nthreads;
            grid.with_blocks_dim(dim, blocks)
                .with_threads_dim(dim, threads)
        }
    }
}

fn generate_sync_scope(kind: ir_ast::SyncPointKind, i: &Info) -> CompileResult<SyncScope> {
    match kind {
        ir_ast::SyncPointKind::BlockLocal => Ok(SyncScope::Block),
        ir_ast::SyncPointKind::BlockCluster => Ok(SyncScope::Cluster),
        ir_ast::SyncPointKind::InterBlock => {
            prickle_compile_error!(i, "Found an inter-block synchronization point \
                                       remaining in parallel code after the \
                                       inter-block transformation. This is likely \
                                       caused by a bug in the compiler.")
        }
    }
}

fn generate_kernel_stmt(
    grid: LaunchArgs,
    map: &[GpuMap],
    mut acc: Vec<Stmt>,
    stmt: ir_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match stmt {
        ir_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = from_ir_type(ty);
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Definition {ty, id, expr, i});
        },
        ir_ast::Stmt::Assign {dst, expr, i} => {
            let dst = from_ir_expr(dst)?;
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Assign {dst, expr, i});
        },
        ir_ast::Stmt::SyncPoint {kind, i} => {
            let scope = generate_sync_scope(kind, &i)?;
            acc.push(Stmt::Synchronize {scope, i});
        },
        ir_ast::Stmt::For {var, lo, hi, step, body, par, i} => {
            let (p, grid, map) = if par.is_parallel() {
                let m = map[0].clone();
                let grid = subtract_from_grid(grid, &m);
                (Some((grid.clone(), m)), grid, &map[1..])
            } else {
                (None, grid, map)
            };
            let var_ty = from_ir_type(lo.get_type().clone());
            let (init, cond, incr) = determine_loop_bounds(p, var.clone(), lo, hi, step)?;
            let body = generate_kernel_stmts(grid, map, vec![], body)?;
            if par.is_parallel() && par.reduction {
                acc.push(Stmt::ParallelReduction {
                    var_ty, var, init, cond, incr, body, nthreads: par.nthreads,
                    tpb: par.tpb, i
                });
            } else {
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i});
            };
        },
        ir_ast::Stmt::If {cond, thn, els, i} => {
            let cond = from_ir_expr(cond)?;
            let thn = generate_kernel_stmts(grid.clone(), map, vec![], thn)?;
            let els = generate_kernel_stmts(grid, map, vec![], els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        },
        ir_ast::Stmt::While {cond, body, i} => {
            let cond = from_ir_expr(cond)?;
            let body = generate_kernel_stmts(grid, map, vec![], body)?;
            acc.push(Stmt::While {cond, body, i});
        },
        ir_ast::Stmt::Return {value, i} => {
            let value = from_ir_expr(value)?;
            acc.push(Stmt::Return {value, i});
        },
        ir_ast::Stmt::Alloc {i, ..} => {
            prickle_compile_error!(i, "Memory allocation is not supported in \
                                     parallel code.")?
        },
        ir_ast::Stmt::Free {i, ..} => {
            prickle_compile_error!(i, "Memory deallocation is not supported in \
                                     parallel code.")?
        },
    };
    Ok(acc)
}

fn generate_kernel_stmts(
    grid: LaunchArgs,
    map: &[GpuMap],
    acc: Vec<Stmt>,
    stmts: Vec<ir_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(acc), |acc, stmt| generate_kernel_stmt(grid.clone(), map, acc?, stmt))
}

fn get_cluster_mapping(
    env: &CodegenEnv,
    inner_mapping: &GpuMap
) -> Option<i64> {
    if env.opts.use_cuda_thread_block_clusters {
        match inner_mapping {
            GpuMap::ThreadBlock {nblocks, ..} => {
                Some(*nblocks)
            },
            _ => None
        }
    } else {
        None
    }
}

fn from_ir_stmt(
    env: &CodegenEnv,
    fun_id: &Name,
    mut host_body: Vec<Stmt>,
    mut kernels: Vec<Top>,
    stmt: ir_ast::Stmt
) -> CompileResult<(Vec<Stmt>, Vec<Top>)> {
    let kernels = match stmt {
        ir_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = from_ir_type(ty);
            let expr = from_ir_expr(expr)?;
            host_body.push(Stmt::Definition {ty, id, expr, i});
            Ok(kernels)
        },
        ir_ast::Stmt::Assign {dst, expr, i} => {
            let dst = from_ir_expr(dst)?;
            let expr = from_ir_expr(expr)?;
            host_body.push(Stmt::Assign {dst, expr, i});
            Ok(kernels)
        },
        ir_ast::Stmt::SyncPoint {i, ..} => {
            prickle_compile_error!(i, "Internal error: Found synchronization point \
                                       outside parallel code")
        },
        ir_ast::Stmt::For {var, lo, hi, step, body, par, i} => {
            let var_ty = from_ir_type(lo.get_type().clone());
            // If this for-loop has been marked for parallelization, we compile its contents to a
            // GPU kernel and insert a kernel launch into the host code. Otherwise, we generate a
            // sequential for-loop and add it to the host code.
            match &env.gpu_mapping.get(&var) {
                Some(m) => {
                    let kernel_id = generate_kernel_name(&fun_id, &var);
                    let stmt = ir_ast::Stmt::For {var, lo, hi, step, body, par, i: i.clone()};
                    let kernel_body = generate_kernel_stmt(
                        m.grid.clone(), &m.get_mapping()[..], vec![], stmt
                    )?;
                    let fv = free_vars::free_variables(&kernel_body);
                    let kernel_params = fv.clone()
                        .into_iter()
                        .map(|(id, ty)| Param {id, ty, i: i.clone()})
                        .collect::<Vec<Param>>();
                    let mut attrs = vec![
                        KernelAttribute::LaunchBounds {threads: m.grid.threads.prod()}
                    ];
                    if let Some(nblocks) = get_cluster_mapping(&env, &m.mapping.last().unwrap()) {
                        let dims = Dim3::default()
                            .with_dim(&Dim::X, nblocks);
                        attrs.push(KernelAttribute::ClusterDims {dims});
                    }
                    let kernel = Top::KernelFunDef {
                        attrs, id: kernel_id.clone(),
                        params: kernel_params, body: kernel_body
                    };
                    kernels.push(kernel);
                    let args = fv.into_iter()
                        .map(|(id, ty)| Expr::Var {id, ty, i: i.clone()})
                        .collect::<Vec<Expr>>();
                    host_body.push(Stmt::KernelLaunch {
                        id: kernel_id, args, grid: m.grid.clone(), i: i.clone()
                    });
                    Ok(kernels)
                },
                None => {
                    let (init, cond, incr) = determine_loop_bounds(None, var.clone(), lo, hi, step)?;
                    let (body, kernels) = from_ir_stmts(env, fun_id, vec![], kernels, body)?;
                    host_body.push(Stmt::For {
                        var_ty, var, init, cond, incr, body, i
                    });
                    Ok(kernels)
                }
            }
        },
        ir_ast::Stmt::If {cond, thn, els, i} => {
            let cond = from_ir_expr(cond)?;
            let (thn, kernels) = from_ir_stmts(env, fun_id, vec![], kernels, thn)?;
            let (els, kernels) = from_ir_stmts(env, fun_id, vec![], kernels, els)?;
            host_body.push(Stmt::If {cond, thn, els, i});
            Ok(kernels)
        },
        ir_ast::Stmt::While {cond, body, i} => {
            let cond = from_ir_expr(cond)?;
            let (body, kernels) = from_ir_stmts(env, fun_id, vec![], kernels, body)?;
            host_body.push(Stmt::While {cond, body, i});
            Ok(kernels)
        },
        ir_ast::Stmt::Return {value, i} => {
            let value = from_ir_expr(value)?;
            host_body.push(Stmt::Return {value, i});
            Ok(kernels)
        },
        ir_ast::Stmt::Alloc {id, elem_ty, sz, i} => {
            let elem_ty = from_ir_type(elem_ty);
            let mem = MemSpace::Device;
            let ty = Type::Pointer {ty: Box::new(elem_ty.clone()), mem};
            let expr = Expr::Int {v: 0, ty: ty.clone(), i: i.clone()};
            host_body.push(Stmt::Definition {ty, id: id.clone(), expr, i: i.clone()});
            host_body.push(Stmt::AllocDevice {id, elem_ty, sz, i});
            Ok(kernels)
        },
        ir_ast::Stmt::Free {id, i} => {
            host_body.push(Stmt::FreeDevice {id, i});
            Ok(kernels)
        }
    }?;
    Ok((host_body, kernels))
}

fn from_ir_stmts(
    env: &CodegenEnv,
    fun_id: &Name,
    host_body: Vec<Stmt>,
    kernels: Vec<Top>,
    stmts: Vec<ir_ast::Stmt>
) -> CompileResult<(Vec<Stmt>, Vec<Top>)> {
    stmts.into_iter()
        .fold(Ok((host_body, kernels)), |acc, stmt| {
            let (host_body, kernels) = acc?;
            from_ir_stmt(env, fun_id, host_body, kernels, stmt)
        })
}

fn unwrap_params(
    env: &CodegenEnv,
    params: Vec<Param>
) -> CompileResult<(Vec<Stmt>, Vec<Param>)> {
    let mut params_init = vec![];
    let mut unwrapped_params = vec![];
    for p in params {
        if let Type::Struct {id} = &p.ty {
            if let Some(fields) = env.struct_fields.get(&id) {
                let field_names = fields.iter()
                    .map(|Field {id, ..}| Name::new(id.clone()).with_new_sym())
                    .collect::<Vec<Name>>();

                let field_exprs = fields.clone()
                    .into_iter()
                    .zip(field_names.clone().into_iter())
                    .map(|(Field {ty, id: label, i}, id)| {
                        (label, Expr::Var {id, ty, i})
                    })
                    .collect::<Vec<(String, Expr)>>();
                let init_expr = Expr::Struct {
                    id: id.clone(), fields: field_exprs, ty: p.ty.clone(),
                    i: p.i.clone()
                };
                params_init.push(Stmt::Definition {
                    ty: p.ty, id: p.id, expr: init_expr, i: p.i
                });

                let mut struct_params = fields.clone()
                    .into_iter()
                    .zip(field_names.into_iter())
                    .map(|(Field {ty, i, ..}, id)| Param {id, ty, i})
                    .collect::<Vec<Param>>();
                unwrapped_params.append(&mut struct_params);
            } else {
                prickle_compile_error!(p.i, "Parameter refers to unknown struct type (internal compiler error)")?;
            }
        } else {
            unwrapped_params.push(p);
        }
    }
    Ok((params_init, unwrapped_params))
}

fn from_ir_param(p: ir_ast::Param) -> Param {
    let ir_ast::Param {id, ty, i} = p;
    Param {id, ty: from_ir_type(ty), i}
}

fn from_ir_fun_def_helper(
    env: &CodegenEnv,
    fun: ir_ast::FunDef,
    is_main: bool
) -> CompileResult<Vec<Top>> {
    let ir_ast::FunDef {id, params, body, res_ty, ..} = fun;
    let params = params.into_iter()
        .map(from_ir_param)
        .collect::<Vec<Param>>();
    let (mut host_body, mut kernel_tops) = from_ir_stmts(env, &id, vec![], vec![], body)?;
    let ret_ty = from_ir_type(res_ty);
    if is_main {
        let (mut params_init, unwrapped_params) = unwrap_params(env, params)?;
        params_init.append(&mut host_body);
        kernel_tops.push(Top::FunDef {
            ret_ty, id, params: unwrapped_params, body: params_init, target: Target::Host
        });
        Ok(kernel_tops)
    } else {
        assert!(kernel_tops.is_empty());
        Ok(vec![Top::FunDef {
            ret_ty, id, params, body: host_body, target: Target::Device
        }])
    }
}

fn from_ir_fun_def(
    env: &CodegenEnv,
    fun: ir_ast::FunDef
) -> CompileResult<Top> {
    let mut v = from_ir_fun_def_helper(env, fun, false)?;
    assert!(v.len() == 1);
    Ok(v.pop().unwrap())
}

fn from_ir_field(f: ir_ast::Field) -> Field {
    let ir_ast::Field {id, ty, i} = f;
    Field {id, ty: from_ir_type(ty), i}
}

fn from_ir_struct(s: ir_ast::StructDef) -> Top {
    Top::StructDef {
        id: s.id,
        fields: s.fields.into_iter()
            .map(|f| from_ir_field(f))
            .collect::<Vec<Field>>()
    }
}

fn struct_top_fields(t: Top) -> (Name, Vec<Field>) {
    if let Top::StructDef {id, fields, ..} = t {
        (id, fields)
    } else {
        unreachable!()
    }
}

pub fn from_general_ir(
    opts: &option::CompileOptions,
    mut ast: ir_ast::Ast,
    gpu_mapping: BTreeMap<Name, GpuMapping>
) -> CompileResult<Ast> {
    let mut tops = ast.structs
        .into_iter()
        .map(from_ir_struct)
        .collect::<Vec<Top>>();
    let struct_fields = tops.clone()
        .into_iter()
        .map(struct_top_fields)
        .collect::<BTreeMap<Name, Vec<Field>>>();
    let env = CodegenEnv {gpu_mapping, struct_fields, opts};

    let main_def = ast.defs.pop().unwrap();
    tops.append(&mut ast.defs
        .into_iter()
        .map(|def| from_ir_fun_def(&env, def))
        .collect::<CompileResult<Vec<Top>>>()?);
    tops.append(&mut from_ir_fun_def_helper(&env, main_def, true)?);
    Ok(tops)
}
