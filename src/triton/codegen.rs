use super::ast::*;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::gpu::ast as gpu_ast;
use crate::gpu::reduce;
use crate::utils::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

#[derive(Clone, Debug)]
struct CodegenCtx {
    pub nthreads: i64,
    pub mask: Option<Box<Expr>>,
}

impl CodegenCtx {
    fn new(nthreads: i64) -> Self {
        CodegenCtx {nthreads, mask: None}
    }

    fn add_mask(&self, mask: Expr) -> Self {
        let mut ctx = self.clone();
        ctx.mask = match ctx.mask {
            Some(m) => Some(Box::new(Expr::BinOp {
                lhs: m,
                op: BinOp::And,
                rhs: Box::new(mask.clone()),
                ty: Type::Tensor {shape: vec![], sz: ElemSize::Bool},
                i: mask.get_info()
            })),
            None => Some(Box::new(mask))
        };
        ctx
    }
}

fn validate_attr(attr: gpu_ast::KernelAttribute, i: &Info) -> CompileResult<i64> {
    match attr {
        gpu_ast::KernelAttribute::LaunchBounds {threads} => Ok(threads),
        gpu_ast::KernelAttribute::ClusterDims {..} => {
            parpy_compile_error!(i, "Cluster dimensions are not supported in the Triton backend")
        },
        gpu_ast::KernelAttribute::SharedMemory {bytes, ..} if bytes > 0 => {
            parpy_compile_error!(i, "Manual use of shared memory is not supported in the Triton backend")
        },
        _ => Ok(0)
    }
}

fn validate_attrs(attrs: Vec<gpu_ast::KernelAttribute>, i: &Info) -> CompileResult<i64> {
    let valid_fn = |acc, attr| {
        let acc = acc?;
        let n = validate_attr(attr, &i)?;
        Ok(if n > 0 { n } else { acc })
    };
    attrs.into_iter().fold(Ok(0), valid_fn)
}

fn to_type_annot(ty: gpu_ast::Type) -> Option<ElemSize> {
    match ty {
        gpu_ast::Type::Scalar {sz} => Some(sz),
        _ => None
    }
}

fn from_gpu_ast_param(p: gpu_ast::Param) -> Param {
    Param {
        id: p.id,
        ty: to_type_annot(p.ty),
    }
}

fn from_gpu_ast_params(params: Vec<gpu_ast::Param>) -> Vec<Param> {
    params.into_iter()
        .map(from_gpu_ast_param)
        .collect::<Vec<Param>>()
}

fn from_gpu_ast_type(ty: gpu_ast::Type, i: &Info) -> CompileResult<Type> {
    match ty {
        gpu_ast::Type::Void => Ok(Type::Void),
        gpu_ast::Type::Scalar {sz} => Ok(Type::Tensor {shape: vec![], sz}),
        gpu_ast::Type::Pointer {ty, mem: gpu_ast::MemSpace::Device} => {
            match from_gpu_ast_type(*ty, &i) {
                Ok(Type::Tensor {shape, sz}) if shape.is_empty() => Ok(Type::Tensor {shape: vec![1], sz}),
                _ => parpy_internal_error!(i, "Failed to convert pointer to a valid Triton type")
            }
        },
        gpu_ast::Type::Pointer {..} => {
            parpy_internal_error!(i, "Unsupported pointer type")
        },
        gpu_ast::Type::Function {..} => {
            parpy_internal_error!(i, "Function types are not supported in Triton")
        }
    }
}

fn from_gpu_ast_expr(mask: &Option<Box<Expr>>, e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ast_type(e.get_type().clone(), &e.get_info())?;
    match e {
        gpu_ast::Expr::Var {id, ty: _, i} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, ty: _, i} => Ok(Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, ty: _, i} => Ok(Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, ty: _, i} => Ok(Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, ty: _, i} => {
            let arg = Box::new(from_gpu_ast_expr(mask, *arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} => {
            let lhs = Box::new(from_gpu_ast_expr(mask, *lhs)?);
            let rhs = Box::new(from_gpu_ast_expr(mask, *rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::Assign {i, ..} => {
            parpy_internal_error!(i, "Assignments as subexpressions are not supported in Triton")
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, ty: _, i} => {
            let cond = Box::new(from_gpu_ast_expr(mask, *cond)?);
            let thn = Box::new(from_gpu_ast_expr(mask, *thn)?);
            let els = Box::new(from_gpu_ast_expr(mask, *els)?);
            Ok(Expr::Where {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i} => {
            let target = Box::new(from_gpu_ast_expr(mask, *target)?);
            let idx = Box::new(from_gpu_ast_expr(mask, *idx)?);
            let target_ty = target.get_type().clone();
            let ptr = Box::new(Expr::BinOp {
                lhs: target, op: BinOp::Add, rhs: idx, ty: target_ty, i: i.clone()
            });
            Ok(Expr::Load {ptr, mask: mask.clone(), ty, i})
        },
        gpu_ast::Expr::Call {id, args, ty: _, i} => {
            let args = args.into_iter()
                .map(|e| from_gpu_ast_expr(&mask, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            Ok(Expr::Call {id, args, ty, i})
        },
        gpu_ast::Expr::PyCallback {i, ..} => {
            parpy_internal_error!(i, "Found Python callback in GPU code which is not allowed")
        },
        gpu_ast::Expr::Convert {e, ty: gpu_ast::Type::Scalar {sz: elem_sz}} => {
            let i = e.get_info();
            let value = Box::new(from_gpu_ast_expr(mask, *e)?);
            Ok(Expr::Full {shape: vec![], value, elem_sz, ty, i})
        },
        gpu_ast::Expr::Convert {e, ..} => {
            let i = e.get_info();
            parpy_internal_error!(i, "Unsupported conversion in Triton codegen")
        },
        gpu_ast::Expr::ThreadIdx {i, ..} => {
            parpy_internal_error!(i, "Thread indices are not supported")
        },
        gpu_ast::Expr::BlockIdx {dim, ty: _, i} => {
            Ok(Expr::ProgramId {dim, ty, i})
        },
    }
}

fn remove_thread_idx(removed: bool, e: gpu_ast::Expr) -> (bool, gpu_ast::Expr) {
    match e {
        gpu_ast::Expr::ThreadIdx {dim: _, ty, i} => (true, gpu_ast::Expr::Int {v: 0, ty, i}),
        _ => e.smap_accum_l(removed, remove_thread_idx)
    }
}

fn extract_upper_bound(
    ctx: &CodegenCtx,
    var: &Name,
    cond: gpu_ast::Expr
) -> CompileResult<Expr> {
    let fail = |i: Info| {
        parpy_compile_error!(i, "Failed to extract upper-bound of for-loop in Triton codegen")
    };
    if let gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} = cond {
        match (*lhs, op) {
            (gpu_ast::Expr::Var {id, ..}, BinOp::Lt) if id == *var => {
                from_gpu_ast_expr(&ctx.mask, *rhs)
            },
            _ => fail(i)
        }
    } else {
        fail(cond.get_info())
    }
}

fn extract_step(
    ctx: &CodegenCtx,
    var: &Name,
    incr: gpu_ast::Expr
) -> CompileResult<i64> {
    let fail = |i: Info| {
        parpy_compile_error!(i, "Failed to extract step size of for-loop in Triton codegen")
    };
    if let gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} = incr {
        match (*lhs, op) {
            (gpu_ast::Expr::Var {id, ..}, BinOp::Add) if id == *var => {
                match from_gpu_ast_expr(&ctx.mask, *rhs)? {
                    Expr::Int {v, ..} => Ok(v as i64),
                    _ => parpy_compile_error!(i, "Found non-literal step size in Triton codegen")
                }
            },
            _ => fail(i)
        }
    } else {
        fail(incr.get_info())
    }
}

fn extract_loop_bounds(
    ctx: &CodegenCtx,
    var_ty: Type,
    var: Name,
    init: gpu_ast::Expr,
    cond: gpu_ast::Expr,
    incr: gpu_ast::Expr,
    i: &Info
) -> CompileResult<(Name, Expr, Expr, i64, Option<Stmt>)> {
    let (removed_thread, init) = remove_thread_idx(false, init);
    let lo = from_gpu_ast_expr(&ctx.mask, init)?;
    let hi = extract_upper_bound(ctx, &var, cond)?;
    let step = extract_step(ctx, &var, incr)?;
    // NOTE(larshum, 2026-02-11): If we removed the use of a thread index, we know this for-loop
    // runs in parallel over threads, in which case we have to restructure it.
    if removed_thread {
        let new_var = Name::new(format!("{0}_chunk", var.get_str())).with_new_sym();
        let var_assign = Stmt::Assign {
            dst: var,
            expr: Expr::BinOp {
                lhs: Box::new(Expr::Var {
                    id: new_var.clone(),
                    ty: var_ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Add,
                rhs: Box::new(Expr::BinOp {
                    lhs: Box::new(Expr::Arange {
                        lo: 0,
                        hi: ctx.nthreads as usize,
                        ty: var_ty.clone(),
                        i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: (step / ctx.nthreads) as i128,
                        ty: var_ty.clone(),
                        i: i.clone()
                    }),
                    ty: var_ty.clone(),
                    i: i.clone()
                }),
                ty: var_ty.clone(),
                i: i.clone()
            },
            i: i.clone()
        };
        Ok((new_var, lo, hi, step, Some(var_assign)))
    } else {
        Ok((var, lo, hi, step, None))
    }
}

fn get_reduction_operator(op: &BinOp, i: &Info) -> CompileResult<ReduceOp> {
    match op {
        BinOp::Add => Ok(ReduceOp::Sum),
        BinOp::Min => Ok(ReduceOp::Min),
        BinOp::Max => Ok(ReduceOp::Max),
        BinOp::Mul => parpy_compile_error!(i, "Multiplication reductions are \
                                               not supported in Triton codegen"),
        _ => parpy_compile_error!(i, "Unsupported reduction operation")
    }
}

fn from_gpu_ast_stmt(
    ctx: &CodegenCtx,
    mut acc: Vec<Stmt>,
    s: gpu_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        gpu_ast::Stmt::Definition {ty: _, id, expr, i} => {
            let expr = from_gpu_ast_expr(&ctx.mask, expr)?;
            acc.push(Stmt::Assign {dst: id, expr, i});
            Ok(acc)
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, i, ..} => {
            let var_ty = from_gpu_ast_type(var_ty, &i)?;
            let (var, lo, hi, step, init_body) = extract_loop_bounds(ctx, var_ty.clone(), var, init, cond, incr, &i)?;
            let body = match init_body {
                Some(var_assign) => {
                    let inner_ctx = ctx.add_mask(Expr::BinOp {
                        lhs: Box::new(Expr::Var {
                            id: var.clone(),
                            ty: var_ty.clone(),
                            i: i.clone()
                        }),
                        op: BinOp::Lt,
                        rhs: Box::new(hi.clone()),
                        ty: var_ty.clone(),
                        i: i.clone()
                    });
                    let mut body = from_gpu_ast_stmts(&inner_ctx, body)?;
                    body.insert(0, var_assign);
                    Ok(body)
                },
                None => from_gpu_ast_stmts(ctx, body)
            }?;
            acc.push(Stmt::For {var, lo, hi, step, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::If {cond, thn, els, i} => {
            acc.push(Stmt::If {
                cond: from_gpu_ast_expr(&ctx.mask, cond)?,
                thn: from_gpu_ast_stmts(ctx, thn)?,
                els: from_gpu_ast_stmts(ctx, els)?,
                i
            });
            Ok(acc)
        },
        gpu_ast::Stmt::While {cond, body, i} => {
            let cond = from_gpu_ast_expr(&ctx.mask, cond)?;
            let body = from_gpu_ast_stmts(ctx, body)?;
            acc.push(Stmt::While {cond, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Return {value, i} => {
            let value = from_gpu_ast_expr(&ctx.mask, value)?;
            acc.push(Stmt::Return {value, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Scope {body, i: _} => {
            acc.append(&mut from_gpu_ast_stmts(ctx, body)?);
            Ok(acc)
        },
        gpu_ast::Stmt::Expr {e: gpu_ast::Expr::Assign {lhs, rhs, ty, i: _}, i} => {
            let ty = from_gpu_ast_type(ty, &i)?;
            let rhs = from_gpu_ast_expr(&ctx.mask, *rhs)?;
            match *lhs {
                gpu_ast::Expr::Var {id, ty: _, i: _} => {
                    acc.push(Stmt::Assign {
                        dst: id,
                        expr: rhs,
                        i
                    });
                    Ok(acc)
                },
                gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i: _} => {
                    let target = from_gpu_ast_expr(&ctx.mask, *target)?;
                    let idx = from_gpu_ast_expr(&ctx.mask, *idx)?;
                    acc.push(Stmt::Expr {
                        e: Expr::Store {
                            ptr: Box::new(Expr::BinOp {
                                lhs: Box::new(target),
                                op: BinOp::Add,
                                rhs: Box::new(idx),
                                ty: ty.clone(),
                                i: i.clone()
                            }),
                            value: Box::new(rhs),
                            mask: ctx.mask.clone(),
                            ty: Type::Void,
                            i: i.clone()
                        },
                        i
                    });
                    Ok(acc)
                },
                _ => parpy_compile_error!(i, "Unsupported left-hand side of assignment in Triton codegen")
            }
        },
        gpu_ast::Stmt::Expr {e, i} => {
            let e = from_gpu_ast_expr(&ctx.mask, e)?;
            acc.push(Stmt::Expr {e, i});
            Ok(acc)
        },
        gpu_ast::Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, i, ..} => {
            let var_ty = from_gpu_ast_type(var_ty, &i)?;
            let (var, lo, hi, step, init_stmt) = extract_loop_bounds(
                ctx, var_ty.clone(), var, init, cond, incr, &i
            )?;
            let inner_ctx = ctx.add_mask(Expr::BinOp {
                lhs: Box::new(Expr::Var {
                    id: var.clone(),
                    ty: var_ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Lt,
                rhs: Box::new(hi.clone()),
                ty: var_ty.clone(),
                i: i.clone()
            });
            let (l, op, r, sz, i) = reduce::extract_reduction_operands(body, &i)?;
            let l = from_gpu_ast_expr(&inner_ctx.mask, l)?;
            let reduce_op = get_reduction_operator(&op, &i)?;
            if let Expr::Var {ref id, ..} = l {
                let ty = Type::Tensor {shape: vec![], sz};
                let r = from_gpu_ast_expr(&inner_ctx.mask, r)?;
                let reduce_stmt = Stmt::Assign {
                    dst: id.clone(),
                    expr: Expr::BinOp {
                        lhs: Box::new(l),
                        op,
                        rhs: Box::new(Expr::Reduce {
                            op: reduce_op,
                            arg: Box::new(r),
                            ty: ty.clone(),
                            i: i.clone()
                        }),
                        ty,
                        i: i.clone()
                    },
                    i: i.clone()
                };
                let body = init_stmt.into_iter()
                    .chain(vec![reduce_stmt].into_iter())
                    .collect::<Vec<Stmt>>();
                acc.push(Stmt::For {var, lo, hi, step, body, i});
                Ok(acc)
            } else {
                parpy_internal_error!(i, "Invalid form of reduction in Triton codegen")
            }
        },
        gpu_ast::Stmt::Synchronize {scope: gpu_ast::SyncScope::Block, i} => {
            acc.push(Stmt::Barrier {i});
            Ok(acc)
        },
        gpu_ast::Stmt::Synchronize {scope: gpu_ast::SyncScope::Cluster, i} => {
            parpy_compile_error!(i, "Cluster-level parallelism is not supported in the Triton backend")
        },
        gpu_ast::Stmt::WarpReduce {i, ..} | gpu_ast::Stmt::ClusterReduce {i, ..} => {
            parpy_internal_error!(i, "Found intrinsic reductions that are not supported in the Triton backend")
        },
        gpu_ast::Stmt::KernelLaunch {i, ..} => {
            parpy_internal_error!(i, "Found kernel launch in kernel code when compiling to Triton")
        },
        gpu_ast::Stmt::AllocDevice {i, ..} | gpu_ast::Stmt::AllocShared {i, ..} |
        gpu_ast::Stmt::FreeDevice {i, ..} | gpu_ast::Stmt::CopyMemory {i, ..} => {
            parpy_internal_error!(i, "Found unsupported allocation statement in kernel code when compiling to Triton")
        }
    }
}

fn from_gpu_ast_stmts(
    ctx: &CodegenCtx,
    stmts: Vec<gpu_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), |acc, s| from_gpu_ast_stmt(&ctx, acc?, s))
}

fn from_gpu_ast_top(t: gpu_ast::Top) -> CompileResult<Top> {
    match t {
        gpu_ast::Top::ExtDecl {i, ..} => {
            Ok(Top::Import {
                package: "TODO".to_string(),
                as_str: None,
                i
            })
        },
        gpu_ast::Top::KernelFunDef {attrs, id, params, body, i} => {
            let nthreads = validate_attrs(attrs, &i)?;
            if nthreads == 0 {
                parpy_internal_error!(i, "Found kernel parallelized over zero threads")?
            }
            let params = from_gpu_ast_params(params);
            let ctx = CodegenCtx::new(nthreads);
            let body = from_gpu_ast_stmts(&ctx, body)?;
            Ok(Top::TritonFunDef {id, params, body, i})
        },
        gpu_ast::Top::FunDef {i, ..} => {
            Ok(Top::Import {
                package: "TODO".to_string(),
                as_str: None,
                i
            })
        }
    }
}

pub fn from_gpu_ast(ast: gpu_ast::Ast) -> CompileResult<Ast> {
    let tops = ast.into_iter()
        .map(from_gpu_ast_top)
        .collect::<CompileResult<Vec<Top>>>()?;
    Ok(Ast {tops})
}
