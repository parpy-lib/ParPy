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

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct CodegenEnv {
    pub ext_map: BTreeMap<Name, String>,
    pub nthreads: i64,
}

impl Default for CodegenEnv {
    fn default() -> Self {
        CodegenEnv {ext_map: BTreeMap::new(), nthreads: 0}
    }
}

impl CodegenEnv {
    fn add_ext(mut self, ext_id: Name, ext_str: String) -> Self {
        self.ext_map.insert(ext_id, ext_str);
        self
    }

    fn with_nthreads(mut self, nthreads: i64) -> Self {
        self.nthreads = nthreads;
        self
    }
}

fn generate_default_imports() -> Vec<Top> {
    let mk_import = |package_str: &str| Top::Import {
        package: package_str.to_string(),
        as_str: None,
        i: Info::default()
    };
    vec![
        mk_import("triton"),
        mk_import("triton.language"),
    ]
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

fn from_gpu_ast_params(params: Vec<gpu_ast::Param>) -> Vec<Name> {
    params.into_iter()
        .map(|p| p.id)
        .collect::<Vec<Name>>()
}

fn from_gpu_ast_type(ty: gpu_ast::Type, i: &Info) -> CompileResult<Type> {
    match ty {
        gpu_ast::Type::Void => Ok(Type::Void),
        gpu_ast::Type::Scalar {sz} => Ok(Type::Tensor {sz, shape: Shape::Num(1)}),
        gpu_ast::Type::Pointer {ty, mem: gpu_ast::MemSpace::Device} => {
            match from_gpu_ast_type(*ty, &i) {
                Ok(Type::Tensor {sz, shape}) => Ok(Type::Pointer {sz, shape}),
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

fn validate_bin_op(op: &BinOp, i: &Info) -> CompileResult<()> {
    match op {
        BinOp::Pow => parpy_compile_error!(i, "The power operator is not supported in Triton"),
        _ => Ok(())
    }
}

fn extract_element_size(ty: &Type, i: &Info) -> CompileResult<ElemSize> {
    match ty.get_elem_size() {
        Some(sz) => Ok(sz.clone()),
        None => parpy_compile_error!(i, "Invalid type of scalar value")
    }
}

fn from_gpu_ast_kernel_expr(env: &CodegenEnv, e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ast_type(e.get_type().clone(), &e.get_info())?;
    match e {
        gpu_ast::Expr::Var {id, ty: _, i} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, ty: _, i} => {
            Ok(Expr::Bool {v, ty: ty.clone(), i: i.clone()})
        },
        gpu_ast::Expr::Int {v, ty: _, i} => {
            Ok(Expr::Int {v, ty: ty.clone(), i: i.clone()})
        },
        gpu_ast::Expr::Float {v, ty: _, i} => {
            Ok(Expr::Float {v, ty: ty.clone(), i: i.clone()})
        },
        gpu_ast::Expr::UnOp {op, arg, ty: _, i} => {
            let arg = Box::new(from_gpu_ast_kernel_expr(env, *arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} => {
            let lhs = Box::new(from_gpu_ast_kernel_expr(env, *lhs)?);
            let rhs = Box::new(from_gpu_ast_kernel_expr(env, *rhs)?);
            validate_bin_op(&op, &i)?;
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::Assign {i, ..} => {
            parpy_internal_error!(i, "Assignments as subexpressions are not supported in Triton")
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, ty: _, i} => {
            let cond = Box::new(from_gpu_ast_kernel_expr(env, *cond)?);
            let thn = Box::new(from_gpu_ast_kernel_expr(env, *thn)?);
            let els = Box::new(from_gpu_ast_kernel_expr(env, *els)?);
            Ok(Expr::Where {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i} => {
            let target = Box::new(from_gpu_ast_kernel_expr(env, *target)?);
            let idx = Box::new(from_gpu_ast_kernel_expr(env, *idx)?);
            let target_ty = target.get_type().clone();
            let ptr = Box::new(Expr::BinOp {
                lhs: target, op: BinOp::Add, rhs: idx, ty: target_ty, i: i.clone()
            });
            Ok(Expr::Load {ptr, mask: None, ty, i})
        },
        gpu_ast::Expr::Call {id, args, ty: _, i} => {
            let args = args.into_iter()
                .map(|e| from_gpu_ast_kernel_expr(env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            match env.ext_map.get(&id) {
                Some(ext_str) => Ok(Expr::ExtCall {id: ext_str.clone(), args, ty, i}),
                None => Ok(Expr::Call {id, args, ty, i})
            }
        },
        gpu_ast::Expr::PyCallback {i, ..} => {
            parpy_internal_error!(i, "Found Python callback in Triton codegen")
        },
        gpu_ast::Expr::Convert {e, ty: gpu_ast::Type::Scalar {sz: elem_sz}} => {
            let i = e.get_info();
            let value = Box::new(from_gpu_ast_kernel_expr(env, *e)?);
            Ok(Expr::Full {shape: 1, value, elem_sz, ty, i})
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
        gpu_ast::Expr::Convert {e, ty} => {
            match *e {
                gpu_ast::Expr::ThreadIdx {dim: _, ty: ety, i} => {
                    (true, gpu_ast::Expr::Int {v: 0, ty: ety, i})
                },
                _ => {
                    let (removed, e) = remove_thread_idx(removed, *e);
                    (removed, gpu_ast::Expr::Convert {e: Box::new(e), ty})
                }
            }
        },
        _ => e.smap_accum_l(removed, remove_thread_idx)
    }
}

fn extract_upper_bound(
    env: &CodegenEnv,
    var: &Name,
    cond: gpu_ast::Expr
) -> CompileResult<Expr> {
    let fail = |i: Info| {
        parpy_compile_error!(i, "Failed to extract upper-bound of for-loop in Triton codegen")
    };
    if let gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} = cond {
        match (*lhs, op) {
            (gpu_ast::Expr::Var {id, ..}, BinOp::Lt) if id == *var => {
                from_gpu_ast_kernel_expr(env, *rhs)
            },
            _ => fail(i)
        }
    } else {
        fail(cond.get_info())
    }
}

fn extract_step(
    env: &CodegenEnv,
    var: &Name,
    incr: gpu_ast::Expr
) -> CompileResult<usize> {
    let fail = |i: Info| {
        parpy_compile_error!(i, "Failed to extract step size of for-loop in Triton codegen")
    };
    if let gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} = incr {
        match (*lhs, op) {
            (gpu_ast::Expr::Var {id, ..}, BinOp::Add) if id == *var => {
                match from_gpu_ast_kernel_expr(env, *rhs)? {
                    Expr::Int {v, ..} => Ok(v as usize),
                    _ => fail(i)
                }
            },
            _ => fail(i)
        }
    } else {
        fail(incr.get_info())
    }
}

fn extract_loop_bounds(
    env: &CodegenEnv,
    var_ty: Type,
    var: Name,
    init: gpu_ast::Expr,
    cond: gpu_ast::Expr,
    incr: gpu_ast::Expr,
    i: &Info
) -> CompileResult<(Name, Expr, Expr, usize, Option<Stmt>)> {
    let (removed_thread, init) = remove_thread_idx(false, init);
    let lo = from_gpu_ast_kernel_expr(env, init)?;
    let hi = extract_upper_bound(env, &var, cond)?;
    let step = extract_step(env, &var, incr)?;
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
                        hi: env.nthreads as usize,
                        ty: var_ty.clone(),
                        i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: (step / env.nthreads as usize) as i128,
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
        BinOp::Mul => Ok(ReduceOp::Prod),
        _ => parpy_compile_error!(i, "Unsupported reduction operation")
    }
}

fn from_gpu_ast_kernel_stmt(
    env: &CodegenEnv,
    mut acc: Vec<Stmt>,
    s: gpu_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        gpu_ast::Stmt::Definition {ty: _, id, expr, i} => {
            let expr = from_gpu_ast_kernel_expr(env, expr)?;
            acc.push(Stmt::Assign {dst: id, expr, i});
            Ok(acc)
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, i, ..} => {
            let var_ty = from_gpu_ast_type(var_ty, &i)?;
            let (var, lo, hi, step, init_body) = extract_loop_bounds(env, var_ty.clone(), var, init, cond, incr, &i)?;
            let body = match init_body {
                Some(var_assign) => {
                    let mut body = from_gpu_ast_kernel_stmts(&env, body)?;
                    body.insert(0, var_assign);
                    Ok(body)
                },
                None => from_gpu_ast_kernel_stmts(env, body)
            }?;
            acc.push(Stmt::For {var, lo, hi, step, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::If {cond, thn, els, i} => {
            acc.push(Stmt::If {
                cond: from_gpu_ast_kernel_expr(env, cond)?,
                thn: from_gpu_ast_kernel_stmts(env, thn)?,
                els: from_gpu_ast_kernel_stmts(env, els)?,
                i
            });
            Ok(acc)
        },
        gpu_ast::Stmt::While {cond, body, i} => {
            let cond = from_gpu_ast_kernel_expr(env, cond)?;
            let body = from_gpu_ast_kernel_stmts(env, body)?;
            acc.push(Stmt::While {cond, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Return {value, i} => {
            let value = from_gpu_ast_kernel_expr(env, value)?;
            acc.push(Stmt::Return {value, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Scope {body, i: _} => {
            acc.append(&mut from_gpu_ast_kernel_stmts(env, body)?);
            Ok(acc)
        },
        gpu_ast::Stmt::Expr {e: gpu_ast::Expr::Assign {lhs, rhs, ty, i: _}, i} => {
            let ty = from_gpu_ast_type(ty, &i)?;
            let sz = extract_element_size(&ty, &i)?;
            let ptr_ty = Type::Pointer {sz, shape: Shape::Num(1)};
            let rhs = from_gpu_ast_kernel_expr(env, *rhs)?;
            match *lhs {
                gpu_ast::Expr::Var {id, ty: _, i: _} => {
                    acc.push(Stmt::Assign {dst: id, expr: rhs, i});
                    Ok(acc)
                },
                gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i: _} => {
                    let target = from_gpu_ast_kernel_expr(env, *target)?;
                    let idx = from_gpu_ast_kernel_expr(env, *idx)?;
                    acc.push(Stmt::Expr {
                        e: Expr::Store {
                            ptr: Box::new(Expr::BinOp {
                                lhs: Box::new(target),
                                op: BinOp::Add,
                                rhs: Box::new(idx.with_type(ptr_ty.clone())),
                                ty: ptr_ty,
                                i: i.clone()
                            }),
                            value: Box::new(rhs),
                            mask: None,
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
            let e = from_gpu_ast_kernel_expr(env, e)?;
            acc.push(Stmt::Expr {e, i});
            Ok(acc)
        },
        gpu_ast::Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, i, ..} => {
            let var_ty = from_gpu_ast_type(var_ty, &i)?;
            let (var, lo, hi, step, init_stmt) = extract_loop_bounds(
                env, var_ty.clone(), var, init, cond, incr, &i
            )?;
            let (l, op, r, sz, i) = reduce::extract_reduction_operands(body, &i)?;
            let l = from_gpu_ast_kernel_expr(&env, l)?;
            let reduce_op = get_reduction_operator(&op, &i)?;
            if let Expr::Var {ref id, ..} = l {
                let ty = Type::Tensor {sz, shape: Shape::Num(1)};
                let r = from_gpu_ast_kernel_expr(&env, r)?;
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

fn from_gpu_ast_kernel_stmts(
    env: &CodegenEnv,
    stmts: Vec<gpu_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), |acc, s| from_gpu_ast_kernel_stmt(&env, acc?, s))
}

fn from_gpu_ast_host_expr(env: &CodegenEnv, e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ast_type(e.get_type().clone(), &e.get_info())?;
    match e {
        gpu_ast::Expr::Var {id, ty: _, i} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, ty: _, i} => Ok(Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, ty: _, i} => Ok(Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, ty: _, i} => Ok(Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, ty: _, i} => {
            let arg = Box::new(from_gpu_ast_host_expr(env, *arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} => {
            let lhs = Box::new(from_gpu_ast_host_expr(env, *lhs)?);
            let rhs = Box::new(from_gpu_ast_host_expr(env, *rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::Assign {i, ..} => {
            parpy_internal_error!(i, "Assignments as subexpressions are not supported in Triton")
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, ty: _, i} => {
            let cond = Box::new(from_gpu_ast_host_expr(env, *cond)?);
            let thn = Box::new(from_gpu_ast_host_expr(env, *thn)?);
            let els = Box::new(from_gpu_ast_host_expr(env, *els)?);
            Ok(Expr::Where {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {i, ..} => {
            parpy_compile_error!(i, "Data cannot be accessed outside parallel \
                                     code in the Triton backend")
        },
        gpu_ast::Expr::Call {id, args, ty: _, i} => {
            let args = args.into_iter()
                .map(|e| from_gpu_ast_host_expr(env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            match env.ext_map.get(&id) {
                Some(ext_str) => Ok(Expr::ExtCall {id: ext_str.clone(), args, ty, i}),
                None => Ok(Expr::Call {id, args, ty, i})
            }
        },
        gpu_ast::Expr::PyCallback {i, ..} => {
            parpy_internal_error!(i, "Found Python callback in Triton codegen")
        },
        gpu_ast::Expr::Convert {e, ty: gpu_ast::Type::Scalar {sz: elem_sz}} => {
            let i = e.get_info();
            let value = Box::new(from_gpu_ast_host_expr(env, *e)?);
            Ok(Expr::Full {shape: 1, value, elem_sz, ty, i})
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

fn from_gpu_ast_kernel_arg(
    env: &CodegenEnv,
    arg: gpu_ast::Expr
) -> CompileResult<Expr> {
    let is_pointer_type = match arg.get_type() {
        gpu_ast::Type::Pointer {..} => true,
        _ => false
    };
    let arg = from_gpu_ast_host_expr(env, arg)?;
    if is_pointer_type {
        let ty = arg.get_type().clone();
        let i = arg.get_info();
        Ok(Expr::ExtCall {
            id: "_parpy_builtin_to_torch".to_string(),
            args: vec![arg], ty, i
        })
    } else {
        Ok(arg)
    }
}

fn from_gpu_ast_host_stmt(
    env: &CodegenEnv,
    mut acc: Vec<Stmt>,
    s: gpu_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        gpu_ast::Stmt::Definition {ty: _, id, expr, i} => {
            let expr = from_gpu_ast_host_expr(env, expr)?;
            acc.push(Stmt::Assign {dst: id, expr, i});
            Ok(acc)
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, unroll: _, i} => {
            let var_ty = from_gpu_ast_type(var_ty, &i)?;
            let (var, lo, hi, step, init_body) = extract_loop_bounds(
                env, var_ty.clone(), var, init, cond, incr, &i
            )?;
            let body = match init_body {
                Some(_) => {
                    parpy_internal_error!(i, "Found parallel for-loop in host \
                                              code when compiling to Triton")
                },
                None => from_gpu_ast_host_stmts(env, body)
            }?;
            acc.push(Stmt::For {var, lo, hi, step, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::If {cond, thn, els, i} => {
            let cond = from_gpu_ast_host_expr(env, cond)?;
            let thn = from_gpu_ast_host_stmts(env, thn)?;
            let els = from_gpu_ast_host_stmts(env, els)?;
            acc.push(Stmt::If {cond, thn, els, i});
            Ok(acc)
        },
        gpu_ast::Stmt::While {cond, body, i} => {
            let cond = from_gpu_ast_host_expr(env, cond)?;
            let body = from_gpu_ast_host_stmts(env, body)?;
            acc.push(Stmt::While {cond, body, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Return {value, i} => {
            let value = from_gpu_ast_host_expr(env, value)?;
            acc.push(Stmt::Return {value, i});
            Ok(acc)
        },
        gpu_ast::Stmt::Scope {body, i: _} => {
            acc.append(&mut from_gpu_ast_host_stmts(env, body)?);
            Ok(acc)
        },
        gpu_ast::Stmt::Expr {e: gpu_ast::Expr::Assign {lhs, rhs, ..}, i} => {
            match *lhs {
                gpu_ast::Expr::Var {id: dst, ..} => {
                    let expr = from_gpu_ast_host_expr(env, *rhs)?;
                    acc.push(Stmt::Assign {dst, expr, i});
                    Ok(acc)
                },
                gpu_ast::Expr::ArrayAccess {..} => {
                    parpy_compile_error!(i, "Data cannot be accessed outside parallel code")
                },
                _ => parpy_internal_error!(i, "Invalid form of assignment encountered \
                                               in Triton codegen")
            }
        },
        gpu_ast::Stmt::Expr {e, i} => {
            let e = from_gpu_ast_host_expr(env, e)?;
            acc.push(Stmt::Expr {e, i});
            Ok(acc)
        },
        gpu_ast::Stmt::ParallelReduction {i, ..} => {
            parpy_internal_error!(i, "Found parallel reduction in host code when compiling to Triton")
        },
        gpu_ast::Stmt::Synchronize {i, ..} => {
            parpy_internal_error!(i, "Found synchronization in host code when compiling to Triton")
        },
        gpu_ast::Stmt::WarpReduce {i, ..} => {
            parpy_internal_error!(i, "Found warp reduction in host code when compiling to Triton")
        },
        gpu_ast::Stmt::ClusterReduce {i, ..} => {
            parpy_internal_error!(i, "Found cluster reduction in host code when compiling to Triton")
        },
        gpu_ast::Stmt::KernelLaunch {id, args, grid, smem, i} => {
            if smem != 0 {
                parpy_compile_error!(i, "Found kernel launch with non-zero shared memory usage in Triton codegen")?
            }
            let args = args.into_iter()
                .map(|arg| from_gpu_ast_kernel_arg(env, arg))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let nwarps = (grid.threads.prod() / 32) as usize;
            acc.push(Stmt::KernelLaunch {
                id,
                block_dims: grid.blocks,
                args,
                nwarps,
                i
            });
            Ok(acc)
        },
        gpu_ast::Stmt::AllocDevice {elem_ty, id, sz: nelems, i} => {
            let elem_ty = from_gpu_ast_type(elem_ty, &i)?;
            let elem_sz = match elem_ty.get_elem_size() {
                Some(sz) => Ok(sz.clone()),
                None => {
                    parpy_internal_error!(i, "Found allocation statement of non-scalar \
                                              element type in Triton codegen")
                }
            }?;
            acc.push(Stmt::Assign {
                dst: id,
                expr: Expr::AllocBuffer {nelems, elem_sz, ty: elem_ty, i: i.clone()},
                i
            });
            Ok(acc)
        },
        gpu_ast::Stmt::FreeDevice {..} => Ok(acc),
        gpu_ast::Stmt::AllocShared {i, ..} => {
            parpy_internal_error!(i, "Found shared memory allocation in host code when compiling to Triton")
        },
        gpu_ast::Stmt::CopyMemory {i, ..} => {
            parpy_internal_error!(i, "Unsupported node CopyMemory in Triton codegen")
        },
    }
}

fn from_gpu_ast_host_stmts(
    env: &CodegenEnv,
    stmts: Vec<gpu_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), |acc, s| from_gpu_ast_host_stmt(&env, acc?, s))
}

fn from_gpu_ast_top(
    env: CodegenEnv,
    mut tops: Vec<Top>,
    t: gpu_ast::Top
) -> CompileResult<(CodegenEnv, Vec<Top>)> {
    match t {
        gpu_ast::Top::ExtDecl {id, ext_id, target, header, i, ..} => {
            if let Some(h) = header {
                tops.push(Top::Import {
                    package: h,
                    as_str: None,
                    i: i.clone()
                });
            }
            if let Target::Device = target {
                let env = env.add_ext(id, ext_id);
                Ok((env, tops))
            } else {
                parpy_compile_error!(i, "Host externals are not supported in Triton")
            }
        },
        gpu_ast::Top::KernelFunDef {attrs, id, params, body, i} => {
            let nthreads = validate_attrs(attrs, &i)?;
            if nthreads == 0 {
                parpy_internal_error!(i, "Found kernel parallelized over zero threads")?
            }
            let params = from_gpu_ast_params(params);
            let env = env.with_nthreads(nthreads);
            let body = from_gpu_ast_kernel_stmts(&env, body)?;
            tops.push(Top::FunDef {triton_jit: true, id, params, body, i});
            Ok((env, tops))
        },
        gpu_ast::Top::FunDef {ret_ty: _, id, params, body, target: Target::Device, i} => {
            let params = from_gpu_ast_params(params);
            let env = env.with_nthreads(1);
            let body = from_gpu_ast_kernel_stmts(&env, body)?;
            tops.push(Top::FunDef {triton_jit: true, id, params, body, i});
            Ok((env, tops))
        },
        gpu_ast::Top::FunDef {ret_ty: _, id, params, body, target: Target::Host, i} => {
            let params = from_gpu_ast_params(params);
            let body = from_gpu_ast_host_stmts(&env, body)?;
            tops.push(Top::FunDef {triton_jit: false, id, params, body, i});
            Ok((env, tops))
        },
    }
}

pub fn from_gpu_ast(ast: gpu_ast::Ast) -> CompileResult<Ast> {
    let env = CodegenEnv::default();
    let mut tops = generate_default_imports();
    let (_, mut gen_tops) = ast.into_iter()
        .fold(Ok((env, vec![])), |acc, t| {
            let (env, tops) = acc?;
            from_gpu_ast_top(env, tops, t)
        })?;
    tops.append(&mut gen_tops);
    Ok(Ast {tops})
}
