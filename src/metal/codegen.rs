use super::ast::*;
use crate::prickle_internal_error;
use crate::prickle_type_error;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::smap::*;

struct CodegenEnv {
    pub on_device: bool
}

impl CodegenEnv {
    pub fn new(target: &gpu_ast::Target) -> Self {
        match target {
            gpu_ast::Target::Host => CodegenEnv {on_device: false},
            gpu_ast::Target::Device => CodegenEnv {on_device: true}
        }
    }
}

struct TopsAcc {
    pub metal: Vec<Top>,
    pub host: Vec<Top>
}

impl Default for TopsAcc {
    fn default() -> Self {
        TopsAcc {metal: vec![], host: vec![]}
    }
}

fn from_gpu_ir_mem(mem: gpu_ast::MemSpace) -> MemSpace {
    match mem {
        gpu_ast::MemSpace::Host => MemSpace::Host,
        gpu_ast::MemSpace::Device => MemSpace::Device,
    }
}

fn from_gpu_ir_type(env: &CodegenEnv, ty: gpu_ast::Type, i: &Info) -> CompileResult<Type> {
    match ty {
        gpu_ast::Type::Void => Ok(Type::Void),
        gpu_ast::Type::Boolean => Ok(Type::Boolean),
        gpu_ast::Type::Scalar {sz} => {
            match sz {
                ElemSize::F64 if env.on_device => {
                    prickle_type_error!(i, "Metal does not support double-precision \
                                          floating-point numbers.")
                },
                _ => Ok(Type::Scalar {sz})
            }
        },
        gpu_ast::Type::Pointer {ty, mem} => {
            let ty = match from_gpu_ir_type(env, *ty, i) {
                Ok(Type::Pointer {..}) => {
                    prickle_type_error!(i, "Found nested pointer in generated code, \
                                          which is not supported in Metal.")
                },
                Ok(ty) => Ok(Box::new(ty)),
                Err(e) => Err(e)
            }?;
            let mem = from_gpu_ir_mem(mem);
            // The only pointers used in host code (outside the device) are pointers to Metal
            // buffers containing GPU data.
            if env.on_device {
                Ok(Type::Pointer {ty, mem})
            } else {
                Ok(Type::Buffer)
            }
        },
        gpu_ast::Type::Struct {id} => {
            prickle_internal_error!(i, "Found struct type {id} in the Metal backend.")
        }
    }
}

fn from_gpu_ir_expr(env: &CodegenEnv, e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ir_type(env, e.get_type().clone(), &e.get_info())?;
    match e {
        gpu_ast::Expr::Var {id, i, ..} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, i, ..} => Ok(Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, i, ..} => Ok(Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, i, ..} => Ok(Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, i, ..} => {
            let arg = Box::new(from_gpu_ir_expr(env, *arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = Box::new(from_gpu_ir_expr(env, *lhs)?);
            let rhs = Box::new(from_gpu_ir_expr(env, *rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, i, ..} => {
            let cond = Box::new(from_gpu_ir_expr(env, *cond)?);
            let thn = Box::new(from_gpu_ir_expr(env, *thn)?);
            let els = Box::new(from_gpu_ir_expr(env, *els)?);
            Ok(Expr::Ternary {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::StructFieldAccess {i, ..} => {
            prickle_internal_error!(i, "Found struct field access in the Metal backend,\
                                        where structs are not supported.")
        },
        gpu_ast::Expr::ArrayAccess {target, idx, i, ..} => {
            let target = Box::new(from_gpu_ir_expr(env, *target)?);
            let idx = Box::new(from_gpu_ir_expr(env, *idx)?);
            if env.on_device {
                Ok(Expr::ArrayAccess {target, idx, ty, i})
            } else {
                Ok(Expr::HostArrayAccess {target, idx, ty, i})
            }
        },
        gpu_ast::Expr::Call {id, args, i, ..} => {
            let args = args.into_iter()
                .map(|arg| from_gpu_ir_expr(env, arg))
                .collect::<CompileResult<Vec<Expr>>>()?;
            Ok(Expr::Call {id, args, ty, i})
        },
        gpu_ast::Expr::Convert {e, ..} => {
            let e = Box::new(from_gpu_ir_expr(env, *e)?);
            Ok(Expr::Convert {e, ty})
        },
        gpu_ast::Expr::Struct {id, i, ..} => {
            prickle_internal_error!(i, "Found struct {id} in the Metal backend,\
                                        where structs are not supported.")
        },
        gpu_ast::Expr::ThreadIdx {dim, i, ..} => Ok(Expr::ThreadIdx {dim, ty, i}),
        gpu_ast::Expr::BlockIdx {dim, i, ..} => Ok(Expr::BlockIdx {dim, ty, i}),
    }
}

fn from_gpu_ir_stmt(env: &CodegenEnv, s: gpu_ast::Stmt) -> CompileResult<Stmt> {
    match s {
        gpu_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = from_gpu_ir_type(env, ty, &i)?;
            let expr = from_gpu_ir_expr(env, expr)?;
            Ok(Stmt::Definition {ty, id, expr})
        },
        gpu_ast::Stmt::Assign {dst, expr, ..} => {
            let dst = from_gpu_ir_expr(env, dst)?;
            let expr = from_gpu_ir_expr(env, expr)?;
            Ok(Stmt::Assign {dst, expr})
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, i} => {
            let var_ty = from_gpu_ir_type(env, var_ty, &i)?;
            let init = from_gpu_ir_expr(env, init)?;
            let cond = from_gpu_ir_expr(env, cond)?;
            let incr = from_gpu_ir_expr(env, incr)?;
            let body = from_gpu_ir_stmts(env, body)?;
            Ok(Stmt::For {var_ty, var, init, cond, incr, body})
        },
        gpu_ast::Stmt::If {cond, thn, els, ..} => {
            let cond = from_gpu_ir_expr(env, cond)?;
            let thn = from_gpu_ir_stmts(env, thn)?;
            let els = from_gpu_ir_stmts(env, els)?;
            Ok(Stmt::If {cond, thn, els})
        },
        gpu_ast::Stmt::While {cond, body, ..} => {
            let cond = from_gpu_ir_expr(env, cond)?;
            let body = from_gpu_ir_stmts(env, body)?;
            Ok(Stmt::While {cond, body})
        },
        gpu_ast::Stmt::Return {value, ..} => {
            let value = from_gpu_ir_expr(env, value)?;
            Ok(Stmt::Return {value})
        },
        gpu_ast::Stmt::Scope {i, ..} => {
            prickle_internal_error!(i, "Found scope statement that should have \
                                        been eliminated.")
        },
        gpu_ast::Stmt::ParallelReduction {i, ..} => {
            prickle_internal_error!(i, "Found parallel reduction statement \
                                        that should have been eliminated.")
        },
        gpu_ast::Stmt::Synchronize {scope, i} => match scope {
            gpu_ast::SyncScope::Block => Ok(Stmt::ThreadgroupBarrier),
            gpu_ast::SyncScope::Cluster => {
                prickle_internal_error!(i, "Found cluster-level synchronization point, \
                                            which is not supported by the Metal backend.")
            },
        },
        gpu_ast::Stmt::WarpReduce {value, op, res_ty, i, ..} => {
            let value = from_gpu_ir_expr(env, value)?;
            let ty = from_gpu_ir_type(env, res_ty, &i)?;
            Ok(Stmt::Assign {
                dst: value.clone(),
                expr: Expr::SimdOp {op, arg: Box::new(value), ty, i}
            })
        },
        gpu_ast::Stmt::ClusterReduce {i, ..} => {
            prickle_internal_error!(i, "Cluster reductions are not supported in Metal.")
        },
        gpu_ast::Stmt::KernelLaunch {id, args, grid, i} => {
            let is_pointer_type = |ty: &gpu_ast::Type| match ty {
                gpu_ast::Type::Pointer {..} => true,
                _ => false
            };
            // NOTE: We only pass along the arguments that are pointers (i.e., which will be
            // represented as buffers). The other arguments will already have been inlined in the
            // code at an earlier stage.
            let args = args.into_iter()
                .filter(|e| is_pointer_type(e.get_type()))
                .map(|e| from_gpu_ir_expr(env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let gpu_ast::LaunchArgs {blocks, threads} = grid;
            Ok(Stmt::Definition {
                ty: Type::Scalar {sz: ElemSize::I32}, id: Name::sym_str("err"),
                expr: Expr::KernelLaunch {id, blocks, threads, args, i}
            })
        },
        gpu_ast::Stmt::AllocDevice {elem_ty, id, sz, i} => {
            let elem_ty = from_gpu_ir_type(env, elem_ty, &i)?;
            Ok(Stmt::Definition {
                ty: Type::Scalar {sz: ElemSize::I32}, id: Name::sym_str("err"),
                expr: Expr::AllocDevice {id, elem_ty, sz, i}
            })
        },
        gpu_ast::Stmt::AllocShared {elem_ty, id, sz, i} => {
            let elem_ty = from_gpu_ir_type(env, elem_ty, &i)?;
            Ok(Stmt::AllocThreadgroup {elem_ty, id, sz})
        },
        gpu_ast::Stmt::FreeDevice {id, ..} => {
            Ok(Stmt::FreeDevice {id})
        }
        gpu_ast::Stmt::CopyMemory {elem_ty, src, src_mem, dst, dst_mem, sz, i} => {
            let elem_ty = from_gpu_ir_type(env, elem_ty, &i)?;
            let src = from_gpu_ir_expr(env, src)?;
            let src_mem = from_gpu_ir_mem(src_mem);
            let dst = from_gpu_ir_expr(env, dst)?;
            let dst_mem = from_gpu_ir_mem(dst_mem);
            Ok(Stmt::CopyMemory {elem_ty, src, src_mem, dst, dst_mem, sz})
        },
    }
}

fn from_gpu_ir_stmts(env: &CodegenEnv, stmts: Vec<gpu_ast::Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .map(|s| from_gpu_ir_stmt(env, s))
        .collect::<CompileResult<Vec<Stmt>>>()
}

fn from_gpu_ir_param(
    env: &CodegenEnv,
    p: gpu_ast::Param,
    idx: Option<i64>
) -> CompileResult<Param> {
    let gpu_ast::Param {id, ty, i} = p;
    let attr = idx.map(|idx| ParamAttribute::Buffer {idx});
    Ok(Param {id, ty: from_gpu_ir_type(env, ty, &i)?, attr})
}

fn from_gpu_ir_attr(
    attr: gpu_ast::KernelAttribute
) -> CompileResult<KernelAttribute> {
    match attr {
        gpu_ast::KernelAttribute::LaunchBounds {threads} => {
            Ok(KernelAttribute::LaunchBounds {threads})
        },
        gpu_ast::KernelAttribute::ClusterDims {..} => {
            prickle_internal_error!(Info::default(), "Found unsupported cluster dimension \
                                                      attribute in Metal backend.")
        }
    }
}

fn from_gpu_ir_top(mut acc: TopsAcc, top: gpu_ast::Top) -> CompileResult<TopsAcc> {
    match top {
        gpu_ast::Top::KernelFunDef {attrs, id, params, body} => {
            let env = CodegenEnv::new(&gpu_ast::Target::Device);
            let attrs = attrs.into_iter()
                .map(from_gpu_ir_attr)
                .collect::<CompileResult<Vec<KernelAttribute>>>()?;
            let params = params.into_iter()
                .enumerate()
                .map(|(idx, p)| from_gpu_ir_param(&env, p, Some(idx as i64)))
                .collect::<CompileResult<Vec<Param>>>()?;
            let body = from_gpu_ir_stmts(&env, body)?;
            acc.metal.push(Top::KernelDef {attrs, id, params, body});
            Ok(acc)
        },
        gpu_ast::Top::FunDef {ret_ty, id, params, body, target} => {
            let env = CodegenEnv::new(&target);
            let ret_ty = from_gpu_ir_type(&env, ret_ty, &Info::default())?;
            let params = params.into_iter()
                .map(|p| from_gpu_ir_param(&env, p, None))
                .collect::<CompileResult<Vec<Param>>>()?;
            let mut body = from_gpu_ir_stmts(&env, body)?;
            if let gpu_ast::Target::Host = target {
                body.push(Stmt::SubmitWork);
                acc.host.push(Top::FunDef {ret_ty, id, params, body});
            } else {
                acc.metal.push(Top::FunDef {ret_ty, id, params, body});
            };
            Ok(acc)
        },
        gpu_ast::Top::StructDef {id, ..} => {
            prickle_internal_error!(Info::default(), "Found struct definition {id} \
                                                      in the Metal backend, where \
                                                      structs are not supported.")
        }
    }
}

struct GridIds<'a> {
    thread_idx: &'a Name,
    block_idx: &'a Name,
}

fn update_grid_index_references_in_kernel_expr(
    ids: &GridIds,
    e: Expr
) -> Expr {
    match e {
        Expr::ThreadIdx {dim, ty, i} => {
            let id = ids.thread_idx.clone();
            let label = dim.pprint_default();
            Expr::Projection {
                e: Box::new(Expr::Var {
                    id, ty: Type::Scalar {sz: ElemSize::I32}, i: Info::default()
                }),
                label, ty, i
            }
        },
        Expr::BlockIdx {dim, ty, i} => {
            let id = ids.block_idx.clone();
            let label = dim.pprint_default();
            Expr::Projection {
                e: Box::new(Expr::Var {
                    id, ty: Type::Scalar {sz: ElemSize::I32}, i: Info::default()
                }),
                label, ty ,i
            }
        },
        _ => e.smap(|e| update_grid_index_references_in_kernel_expr(ids, e))
    }
}

fn update_grid_index_references_in_kernel_stmt(
    ids: &GridIds,
    s: Stmt
) -> Stmt {
    s.smap(|s| update_grid_index_references_in_kernel_stmt(ids, s))
        .smap(|e| update_grid_index_references_in_kernel_expr(ids, e))
}

fn update_grid_index_references_in_kernel_body(
    ids: GridIds,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    body.smap(|s| update_grid_index_references_in_kernel_stmt(&ids, s))
}

fn add_grid_index_arguments_to_kernels_top(t: Top) -> Top {
    match t {
        Top::KernelDef {attrs, id, mut params, body} => {
            let thread_idx_id = Name::sym_str("thread_idx");
            let block_idx_id = Name::sym_str("block_idx");
            let ids = GridIds {
                thread_idx: &thread_idx_id,
                block_idx: &block_idx_id
            };
            let body = update_grid_index_references_in_kernel_body(ids, body);
            params.push(Param {
                id: thread_idx_id.clone(), ty: Type::Uint3,
                attr: Some(ParamAttribute::ThreadIndex)
            });
            params.push(Param {
                id: block_idx_id.clone(), ty: Type::Uint3,
                attr: Some(ParamAttribute::BlockIndex)
            });
            Top::KernelDef {attrs, id, params, body}
        },
        _ => t
    }
}

fn add_grid_index_arguments_to_kernels(mut tops: TopsAcc) -> TopsAcc {
    tops.metal = tops.metal.smap(add_grid_index_arguments_to_kernels_top);
    tops
}

pub fn from_gpu_ir(ast: gpu_ast::Ast) -> CompileResult<Ast> {
    let includes = vec!["\"prickle_metal.h\"".to_string()];
    let tops_init = Ok(TopsAcc::default());
    let tops = ast.sfold_owned_result(tops_init, from_gpu_ir_top)?;
    let tops = add_grid_index_arguments_to_kernels(tops);
    Ok(Ast {
        includes,
        metal_tops: tops.metal,
        host_tops: tops.host
    })
}
