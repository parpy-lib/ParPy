use super::ast::*;
use super::free_vars;
use super::par::{GpuMap, GpuMapping};
use crate::parpy_compile_error;
use crate::option::CompileOptions;
use crate::ir::ast as ir_ast;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

#[derive(Debug)]
struct CodegenEnv<'a> {
    gpu_mapping: BTreeMap<Name, GpuMapping>,
    struct_fields: BTreeMap<Name, Vec<Field>>,
    opts: &'a CompileOptions
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
    let int_ty = init.get_type().clone();
    let i = init.get_info();
    let var_e = Expr::Var {id: var, ty: int_ty.clone(), i: i.clone()};
    let cond_op = if step_size > 0 { BinOp::Lt } else { BinOp::Gt };
    let cond = Expr::BinOp {
        lhs: Box::new(var_e.clone()),
        op: cond_op,
        rhs: Box::new(from_ir_expr(hi)?),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: i.clone()
    };
    let step = Expr::Int {
        v: step_size as i128, ty: int_ty.clone(), i: i.clone()
    };
    let fn_incr = |v| Expr::BinOp {
        lhs: Box::new(var_e),
        op: BinOp::Add,
        rhs: Box::new(Expr::BinOp {
            lhs: Box::new(step.clone()),
            op: BinOp::Mul,
            rhs: Box::new(Expr::Int {
                v: v as i128, ty: int_ty.clone(), i: i.clone()
            }),
            ty: int_ty.clone(), i: i.clone()
        }),
        ty: int_ty.clone(), i: i.clone()
    };
    match par {
        Some((grid, GpuMap::Thread {n, dim, mult})) => {
            let tot = grid.threads.get_dim(&dim);
            let idx = Expr::ThreadIdx {dim, ty: int_ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n, mult);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add,
                rhs: Box::new(Expr::BinOp {
                    lhs: Box::new(step.clone()),
                    op: BinOp::Mul,
                    rhs: Box::new(rhs),
                    ty: int_ty.clone(), i: i.clone()
                }),
                ty: int_ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
        },
        Some((grid, GpuMap::Block {n, dim, mult})) => {
            let tot = grid.blocks.get_dim(&dim);
            let idx = Expr::BlockIdx {dim, ty: int_ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n, mult);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add,
                rhs: Box::new(Expr::BinOp {
                    lhs: Box::new(step.clone()),
                    op: BinOp::Mul,
                    rhs: Box::new(rhs),
                    ty: int_ty.clone(), i: i.clone()
                }),
                ty: int_ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
        },
        Some((grid, GpuMap::ThreadBlock {n, nthreads, nblocks, dim})) => {
            let tot_blocks = grid.blocks.get_dim(&dim);
            let idx = Expr::BinOp {
                lhs: Box::new(Expr::BinOp {
                    lhs: Box::new(Expr::BlockIdx {
                        dim, ty: int_ty.clone(), i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: nthreads as i128, ty: int_ty.clone(), i: i.clone()
                    }),
                    ty: int_ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Add,
                rhs: Box::new(Expr::ThreadIdx {dim, ty: int_ty.clone(), i: i.clone()}),
                ty: int_ty.clone(),
                i: i.clone()
            };
            let rhs = remainder_if_shared_dimension(idx.clone(), tot_blocks, nblocks, 1);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: int_ty.clone(), i: i.clone()
            };
            // If the total number of threads we are using is not evenly divisible by the number of
            // threads per block, we insert an additional condition to ensure only the intended
            // threads run the code inside the for-loop.
            let bool_ty = Type::Scalar {sz: ElemSize::Bool};
            let cond = if nthreads * nblocks != n {
                Expr::BinOp {
                    lhs: Box::new(cond),
                    op: BinOp::And,
                    rhs: Box::new(Expr::BinOp {
                        lhs: Box::new(idx.clone()),
                        op: BinOp::Lt,
                        rhs: Box::new(Expr::Int {
                            v: n as i128, ty: int_ty.clone(), i: i.clone()
                        }),
                        ty: bool_ty.clone(), i: i.clone()
                    }),
                    ty: bool_ty.clone(), i: i.clone()
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
            let threads = grid.threads.get_dim(&dim) / nthreads;
            let blocks = grid.blocks.get_dim(&dim) / nblocks;
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
            parpy_compile_error!(i, "Found an inter-block synchronization point \
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
            parpy_compile_error!(i, "Memory allocation is not supported in \
                                     parallel code.")?
        },
        ir_ast::Stmt::Free {i, ..} => {
            parpy_compile_error!(i, "Memory deallocation is not supported in \
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
        .fold(Ok(acc), |acc, stmt| {
            generate_kernel_stmt(grid.clone(), map, acc?, stmt)
        })
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
            parpy_compile_error!(i, "Internal error: Found synchronization point \
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
                parpy_compile_error!(p.i, "Parameter refers to unknown struct type (internal compiler error)")?;
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

fn from_ir_params(params: Vec<ir_ast::Param>) -> Vec<Param> {
    params.into_iter()
        .map(from_ir_param)
        .collect::<Vec<Param>>()
}

fn from_ir_main_def(
    env: &CodegenEnv,
    fun: ir_ast::FunDef,
) -> CompileResult<Vec<Top>> {
    let ir_ast::FunDef {id, params, body, ..} = fun;
    let params = params.into_iter()
        .map(from_ir_param)
        .collect::<Vec<Param>>();
    let (mut host_body, mut kernel_tops) = from_ir_stmts(env, &id, vec![], vec![], body)?;
    let (mut body, unwrapped_params) = unwrap_params(env, params)?;
    body.append(&mut host_body);
    let ret_ty = Type::Scalar {sz: ElemSize::I32};
    body.push(Stmt::Return {
        value: Expr::Int {v: 0, ty: ret_ty.clone(), i: Info::default()},
        i: Info::default()
    });
    kernel_tops.push(Top::FunDef {
        ret_ty, id, params: unwrapped_params, body, target: Target::Host
    });
    Ok(kernel_tops)
}

fn from_ir_field(f: ir_ast::Field) -> Field {
    let ir_ast::Field {id, ty, i} = f;
    Field {id, ty: from_ir_type(ty), i}
}

fn from_ir_top(
    mut env: CodegenEnv,
    t: ir_ast::Top
) -> CompileResult<(CodegenEnv, Top)> {
    match t {
        ir_ast::Top::StructDef {id, fields, ..} => {
            let fields = fields.into_iter()
                .map(from_ir_field)
                .collect::<Vec<Field>>();
            env.struct_fields.insert(id.clone(), fields.clone());
            Ok((env, Top::StructDef {id, fields}))
        },
        ir_ast::Top::ExtDecl {id, ext_id, params, res_ty, target, header, i: _} => {
            let params = from_ir_params(params);
            let ret_ty = from_ir_type(res_ty);
            Ok((env, Top::ExtDecl {ret_ty, id, ext_id, params, target, header}))
        },
        ir_ast::Top::FunDef {v: ir_ast::FunDef {id, params, body, res_ty, i}} => {
            let params = from_ir_params(params);
            let ret_ty = from_ir_type(res_ty);
            let (body, kernel_tops) = from_ir_stmts(&env, &id, vec![], vec![], body)?;
            if kernel_tops.is_empty() {
                Ok((env, Top::FunDef {ret_ty, id, params, body, target: Target::Device}))
            } else {
                parpy_compile_error!(
                    i,
                    "Found parallel code outside main function definition, \
                     which is not allowed.")
            }
        },
    }
}

pub fn from_general_ir(
    opts: &CompileOptions,
    ast: ir_ast::Ast,
    gpu_mapping: BTreeMap<Name, GpuMapping>
) -> CompileResult<Ast> {
    let env = CodegenEnv {gpu_mapping, struct_fields: BTreeMap::new(), opts};
    let (env, mut tops) = ast.tops.into_iter()
        .fold(Ok((env, vec![])), |acc, t| {
            let (env, mut tops) = acc?;
            let (env, t) = from_ir_top(env, t)?;
            tops.push(t);
            Ok((env, tops))
        })?;
    tops.append(&mut from_ir_main_def(&env, ast.main)?);
    Ok(tops)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;
    use crate::ir::ast_builder as ir;
    use crate::par;
    use crate::test::*;
    use crate::utils::pprint::PrettyPrint;

    fn mk_env<'a>(opts: &'a CompileOptions) -> CodegenEnv<'a> {
        CodegenEnv {
            gpu_mapping: BTreeMap::new(),
            struct_fields: BTreeMap::new(),
            opts
        }
    }

    #[test]
    fn from_scalar_type() {
        assert_eq!(from_ir_type(ir::scalar(ElemSize::F32)), scalar(ElemSize::F32));
    }

    #[test]
    fn from_tensor_vec_type() {
        let ty = ir_ast::Type::Tensor {sz: ElemSize::I64, shape: vec![10]};
        assert_eq!(from_ir_type(ty), pointer(scalar(ElemSize::I64), MemSpace::Device));
    }

    #[test]
    fn from_int_expr() {
        assert_eq!(
            from_ir_expr(ir::int(1, Some(ElemSize::I64))).unwrap(),
            int(1, Some(ElemSize::I64))
        );
    }

    #[test]
    fn gen_kernel_name() {
        let fun_id = id("f");
        let loop_var = id("x");
        let kernel_id = generate_kernel_name(&fun_id, &loop_var);
        assert_eq!(kernel_id.get_str(), "f_x");
        assert!(kernel_id.has_sym());
    }

    #[test]
    fn no_remainder_unique_dim() {
        let idx = int(1, None);
        assert_eq!(remainder_if_shared_dimension(idx.clone(), 10, 10, 1), idx);
    }

    #[test]
    fn remainder_with_shared_dim() {
        let idx = int(1, None);
        let expected = binop(
            binop(
                idx.clone(),
                BinOp::Div,
                int(1, None),
                scalar(ElemSize::I64)
            ),
            BinOp::Rem,
            int(5, None),
            scalar(ElemSize::I64)
        );
        assert_eq!(remainder_if_shared_dimension(idx, 10, 5, 1), expected);
    }

    #[test]
    fn subtract_from_grid_threads() {
        let grid = LaunchArgs::default().with_threads_dim(&Dim::X, 1024);
        let m = GpuMap::Thread {n: 32, mult: 1, dim: Dim::X};
        let expected = LaunchArgs::default().with_threads_dim(&Dim::X, 32);
        assert_eq!(subtract_from_grid(grid, &m), expected);
    }

    #[test]
    fn subtract_from_grid_thread_block() {
        let grid = LaunchArgs::default()
            .with_blocks_dim(&Dim::X, 8)
            .with_threads_dim(&Dim::X, 128);
        let m = GpuMap::ThreadBlock {n: 500, nthreads: 128, nblocks: 4, dim: Dim::X};
        let expected = LaunchArgs::default().with_blocks_dim(&Dim::X, 2);
        assert_eq!(subtract_from_grid(grid, &m), expected);
    }

    #[test]
    fn subtract_from_grid_block() {
        let grid = LaunchArgs::default().with_blocks_dim(&Dim::X, 32);
        let m = GpuMap::Block {n: 8, mult: 4, dim: Dim::X};
        let expected = LaunchArgs::default().with_blocks_dim(&Dim::X, 4);
        assert_eq!(subtract_from_grid(grid, &m), expected);
    }

    #[test]
    fn generate_block_local_sync_scope() {
        let s = generate_sync_scope(ir_ast::SyncPointKind::BlockLocal, &i());
        assert_eq!(s, Ok(SyncScope::Block));
    }

    #[test]
    fn generate_block_cluster_sync_scope() {
        let s = generate_sync_scope(ir_ast::SyncPointKind::BlockCluster, &i());
        assert_eq!(s, Ok(SyncScope::Cluster));
    }

    #[test]
    fn generate_inter_block_sync_scope() {
        let s = generate_sync_scope(ir_ast::SyncPointKind::InterBlock, &i());
        assert_error_matches(s, "inter-block synchronization point remaining");
    }

    fn gen_kernel_stmt(nb: i64, nt: i64, n: i64, s: ir_ast::Stmt) -> CompileResult<Stmt> {
        let grid = LaunchArgs::default()
            .with_blocks_dim(&Dim::X, nb)
            .with_threads_dim(&Dim::X, nt);
        let m = if nb > 1 {
            vec![GpuMap::ThreadBlock {n, nthreads: nt, nblocks: nb, dim: Dim::X}]
        } else {
            vec![GpuMap::Thread {n: nt, mult: 1, dim: Dim::X}]
        };
        generate_kernel_stmt(grid, &m, vec![], s).map(|mut v| v.pop().unwrap())
    }

    fn _gen_for(par: par::LoopPar) -> ir_ast::Stmt {
        ir_ast::Stmt::For {
            var: id("x"),
            lo: ir::int(0, None),
            hi: ir::int(10, None),
            step: 1,
            par,
            body: vec![],
            i: Info::default()
        }
    }

    fn assert_equal_statements(l: Stmt, r: Stmt) {
        assert_eq!(
            l, r, "\nExpected:\n{0}\nActual:\n{1}",
            r.pprint_default(), l.pprint_default()
        );
    }

    #[test]
    fn generate_seq_for_kernel_stmt() {
        let s = _gen_for(par::LoopPar::default());
        let ty = scalar(ElemSize::I64);
        let incr = binop(
            var("x", ty.clone()),
            BinOp::Add,
            binop(
                int(1, None),
                BinOp::Mul,
                int(1, None),
                ty.clone()
            ),
            ty.clone()
        );
        let expected = Stmt::For {
            var_ty: ty.clone(),
            var: id("x"),
            init: int(0, None),
            cond: binop(var("x", ty.clone()), BinOp::Lt, int(10, None), bool_ty()),
            incr, body: vec![], i: Info::default()
        };
        let s = gen_kernel_stmt(1, 1, 1, s).unwrap();
        assert_equal_statements(s, expected);
    }

    #[test]
    fn generate_par_for_kernel_stmt() {
        let s = _gen_for(par::LoopPar::default().threads(128).unwrap());
        let ty = scalar(ElemSize::I64);
        let init = binop(
            int(0, None),
            BinOp::Add,
            binop(int(1, None), BinOp::Mul, thread_idx(Dim::X), ty.clone()),
            ty.clone()
        );
        let incr = binop(
            var("x", ty.clone()),
            BinOp::Add,
            binop(
                int(1, None),
                BinOp::Mul,
                int(128, None),
                ty.clone()
            ),
            ty.clone()
        );
        let expected = Stmt::For {
            var_ty: ty.clone(),
            var: id("x"),
            cond: binop(var("x", ty.clone()), BinOp::Lt, int(10, None), bool_ty()),
            init, incr, body: vec![], i: Info::default()
        };
        assert_equal_statements(gen_kernel_stmt(1, 128, 128, s).unwrap(), expected);
    }

    #[test]
    fn generate_multi_block_reduction_kernel_stmt() {
        let s = _gen_for(par::LoopPar::default().reduce().threads(2000).unwrap());
        let ty = scalar(ElemSize::I64);
        let idx = binop(
            binop(block_idx(Dim::X), BinOp::Mul, int(1024, None), ty.clone()),
            BinOp::Add,
            thread_idx(Dim::X),
            ty.clone()
        );
        let init = binop(int(0, None), BinOp::Add, idx.clone(), ty.clone());
        let cond = binop(
            binop(var("x", ty.clone()), BinOp::Lt, int(10, None), ty.clone()),
            BinOp::And,
            binop(idx.clone(), BinOp::Lt, int(2000, None), ty.clone()),
            ty.clone()
        );
        let incr = binop(
            var("x", ty.clone()),
            BinOp::Add,
            binop(
                int(1, None),
                BinOp::Mul,
                int(2000, None),
                ty.clone()
            ),
            ty.clone()
        );
        let expected = Stmt::ParallelReduction {
            var_ty: scalar(ElemSize::I64),
            var: id("x"),
            init, cond, incr,
            body: vec![],
            nthreads: 2000,
            tpb: 1024,
            i: Info::default()
        };
        assert_equal_statements(gen_kernel_stmt(2, 1024, 2000, s).unwrap(), expected);
    }

    fn cluster_mapping_h(m: GpuMap, use_clusters: bool) -> Option<i64> {
        let mut opts = CompileOptions::default();
        opts.use_cuda_thread_block_clusters = use_clusters;
        let env = mk_env(&opts);
        get_cluster_mapping(&env, &m)
    }

    #[test]
    fn enabled_cluster_mapping() {
        let m = GpuMap::ThreadBlock {n: 1024, nthreads: 1024, nblocks: 1, dim: Dim::X};
        assert_eq!(cluster_mapping_h(m, true), Some(1));
    }

    #[test]
    fn enabled_cluster_mapping_threads() {
        let m = GpuMap::Thread {n: 1024, mult: 1, dim: Dim::X};
        assert_eq!(cluster_mapping_h(m, true), None);
    }

    #[test]
    fn disabled_cluster_mapping() {
        let m = GpuMap::ThreadBlock {n: 1024, nthreads: 1024, nblocks: 1, dim: Dim::X};
        assert_eq!(cluster_mapping_h(m, false), None);
    }

    #[test]
    fn from_host_stmt_sync_point() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let s = ir_ast::Stmt::SyncPoint {kind: ir_ast::SyncPointKind::BlockLocal, i: i()};
        assert_error_matches(
            from_ir_stmt(&env, &id("x"), vec![], vec![], s),
            r"synchronization point outside parallel code"
        );
    }

    #[test]
    fn from_host_seq_for_stmt() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let s = _gen_for(par::LoopPar::default());
        let (mut body, kernels) = from_ir_stmt(&env, &id("f"), vec![], vec![], s).unwrap();
        assert_eq!(kernels.len(), 0);

        let ty = scalar(ElemSize::I64);
        let expected = Stmt::For {
            var_ty: ty.clone(),
            var: id("x"),
            init: int(0, Some(ElemSize::I64)),
            cond: binop(var("x", ty.clone()), BinOp::Lt, int(10, None), scalar(ElemSize::Bool)),
            incr: binop(
                var("x", ty.clone()),
                BinOp::Add,
                binop(int(1, None), BinOp::Mul, int(1, None), ty.clone()),
                ty
            ),
            body: vec![],
            i: Info::default()
        };
        assert_eq!(body.pop().unwrap(), expected);
        assert!(body.is_empty());
    }

    #[test]
    fn from_host_par_for_stmt() {
        let opts = CompileOptions::default();
        let mut env = mk_env(&opts);
        let m = GpuMapping::new(par::DEFAULT_TPB).add_parallelism(10);
        env.gpu_mapping.insert(id("x"), m.clone());
        let s = _gen_for(par::LoopPar::default());
        let (body, kernels) = from_ir_stmt(&env, &id("f"), vec![], vec![], s).unwrap();
        assert_eq!(kernels.len(), 1);

        if let [Stmt::KernelLaunch {id, args, grid, ..}] = &body[..] {
            assert_eq!(id.get_str(), "f_x");
            assert!(id.has_sym());
            assert_eq!(args.len(), 0);
            assert_eq!(grid.clone(), m.grid);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn from_alloc_stmt() {
        let s = ir_ast::Stmt::Alloc {
            id: id("x"),
            elem_ty: ir::scalar(ElemSize::F32),
            sz: 10,
            i: Info::default()
        };
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let (mut body, kernels) = from_ir_stmt(&env, &id("f"), vec![], vec![], s).unwrap();
        assert!(kernels.is_empty());

        let snd = body.pop().unwrap();
        let fst = body.pop().unwrap();
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Device
        };
        let fst_expected = Stmt::Definition {
            ty: ty.clone(),
            id: id("x"),
            expr: Expr::Int {v: 0, ty: ty.clone(), i: i()},
            i: i()
        };
        assert_eq!(fst, fst_expected);
        let snd_expected = Stmt::AllocDevice {
            id: id("x"),
            elem_ty: scalar(ElemSize::F32),
            sz: 10,
            i: i()
        };
        assert_eq!(snd, snd_expected);
    }

    #[test]
    fn unwrap_empty_params() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let (init_stmts, params) = unwrap_params(&env, vec![]).unwrap();
        assert!(init_stmts.is_empty());
        assert!(params.is_empty());
    }

    #[test]
    fn unwrap_known_struct_params() {
        let opts = CompileOptions::default();
        let mut env = mk_env(&opts);
        let fields = vec![
            Field {id: "a".to_string(), ty: scalar(ElemSize::F64), i: i()},
            Field {id: "b".to_string(), ty: scalar(ElemSize::I32), i: i()},
        ];
        env.struct_fields = vec![(id("s"), fields)].into_iter()
            .collect::<BTreeMap<Name, Vec<Field>>>();
        let params = vec![
            Param {id: id("x"), ty: Type::Struct {id: id("s")}, i: i()},
            Param {id: id("y"), ty: scalar(ElemSize::F32), i: i()},
        ];
        let (init_stmts, params) = unwrap_params(&env, params).unwrap();

        if let [Stmt::Definition {ty, id: id_def, expr, ..}] = &init_stmts[..] {
            assert!(matches!(ty, Type::Struct {..}));
            assert_eq!(id_def.clone(), id("x"));
            assert!(matches!(expr, Expr::Struct {..}));
        } else {
            assert!(false)
        };

        let param_tys = params.into_iter()
            .map(|Param {ty, ..}| ty)
            .collect::<Vec<Type>>();
        let expected_types = vec![
            scalar(ElemSize::F64), scalar(ElemSize::I32), scalar(ElemSize::F32)
        ];
        assert_eq!(param_tys, expected_types);
    }

    #[test]
    fn unwrap_unknown_struct_param() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let params = vec![
            Param {id: id("x"), ty: Type::Struct {id: id("s")}, i: i()}
        ];
        assert_error_matches(unwrap_params(&env, params), r"unknown struct type");
    }

    #[test]
    fn from_ir_called_fun() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let v = ir_ast::FunDef {
            id: id("f"), params: vec![], body: vec![], res_ty: ir_ast::Type::Void, i: i()
        };
        let (_, top) = from_ir_top(env, ir_ast::Top::FunDef {v}).unwrap();
        let expected = Top::FunDef {
            ret_ty: Type::Void, id: id("f"), params: vec![], body: vec![],
            target: Target::Device
        };
        assert_eq!(top, expected);
    }

    #[test]
    fn from_ir_main_fun() {
        let opts = CompileOptions::default();
        let env = mk_env(&opts);
        let main = ir_ast::FunDef {
            id: id("f"), params: vec![], body: vec![], res_ty: ir_ast::Type::Void, i: i()
        };
        let mut tops = from_ir_main_def(&env, main).unwrap();
        assert_eq!(tops.len(), 1);
        let expected = Top::FunDef {
            ret_ty: Type::Scalar {sz: ElemSize::I32},
            id: id("f"),
            params: vec![],
            body: vec![Stmt::Return {value: int(0, Some(ElemSize::I32)), i: i()}],
            target: Target::Host
        };
        assert_eq!(tops.pop().unwrap(), expected);
    }
}
