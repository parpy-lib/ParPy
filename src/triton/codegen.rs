use super::ast::*;
use crate::option::CompileOptions;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::gpu::ast as gpu_ast;
use crate::gpu::ast::LaunchArgs;
use crate::gpu::par;
use crate::gpu::reduce;
use crate::utils::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
struct CodegenEnv {
    pub ext_map: BTreeMap<Name, String>,
    pub kernel_dims: BTreeMap<Name, LaunchArgs>,
    pub kernel_params: Vec<Vec<Param>>,
    pub current_grid: LaunchArgs,
    pub sz: ScalarSizes,
}

impl CodegenEnv {
    fn new(opts: &CompileOptions) -> Self {
        CodegenEnv {
            ext_map: BTreeMap::new(),
            kernel_dims: BTreeMap::new(),
            kernel_params: vec![],
            current_grid: LaunchArgs::default(),
            sz: ScalarSizes::from_opts(opts),
        }
    }

    fn add_ext(mut self, ext_id: Name, ext_str: String) -> Self {
        self.ext_map.insert(ext_id, ext_str);
        self
    }

    fn set_active_kernel(mut self, o: Option<&Name>) -> Self {
        match o.map_or(None, |id| self.kernel_dims.get(id)) {
            Some(grid) => self.current_grid = grid.clone(),
            None => self.current_grid = LaunchArgs::default(),
        };
        self
    }
}

fn generate_default_imports() -> Vec<Top> {
    let mk_import = |package_str: &str, as_str: Option<&str>| Top::Import {
        package: package_str.to_string(),
        as_str: as_str.map(|s| s.to_string()),
        i: Info::default()
    };
    vec![
        mk_import("triton", None),
        mk_import("triton.language", Some("tl")),
    ]
}

fn collect_kernel_dims_stmt(mut env: CodegenEnv, s: &gpu_ast::Stmt) -> CodegenEnv {
    match s {
        gpu_ast::Stmt::KernelLaunch {id, grid, ..} => {
            env.kernel_dims.insert(id.clone(), grid.clone());
            env
        },
        _ => s.sfold(env, collect_kernel_dims_stmt)
    }
}

fn collect_kernel_dims_top(env: CodegenEnv, t: &gpu_ast::Top) -> CodegenEnv {
    match t {
        gpu_ast::Top::FunDef {body, target: Target::Host, ..} => {
            body.sfold(env, collect_kernel_dims_stmt)
        },
        _ => env
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

fn validate_attrs(attrs: Vec<gpu_ast::KernelAttribute>, i: &Info) -> CompileResult<()> {
    let valid_fn = |acc, attr| {
        let acc = acc?;
        validate_attr(attr, &i)?;
        Ok(acc)
    };
    attrs.into_iter().fold(Ok(0), valid_fn)?;
    Ok(())
}

fn from_gpu_ast_params(params: Vec<gpu_ast::Param>) -> CompileResult<Vec<Param>> {
    params.into_iter()
        .map(|gpu_ast::Param {id, ty, i}| {
            Ok(Param {
                id: id,
                ty: from_gpu_ast_type(ty, &i)?,
                annot_ty: AnnotType::Any,
                i
            })
        })
        .collect::<CompileResult<Vec<Param>>>()
}

fn from_gpu_ast_type(ty: gpu_ast::Type, i: &Info) -> CompileResult<Type> {
    match ty {
        gpu_ast::Type::Void => Ok(Type::Void),
        gpu_ast::Type::Scalar {sz} => Ok(Type::Tensor {sz, shape: Shape::Num(1)}),
        gpu_ast::Type::Pointer {ty, mem: gpu_ast::MemSpace::Device} => {
            match from_gpu_ast_type(*ty, &i) {
                Ok(ty @ (Type::Tensor {..} | Type::Pointer {..})) => {
                    Ok(Type::Pointer {
                        ty: Box::new(ty),
                        shape: Shape::Num(1)
                    })
                },
                Ok(Type::Function {..}) => {
                    parpy_compile_error!(i, "Function type pointers are not supported in Triton")
                },
                Ok(Type::Void) => parpy_compile_error!(i, "Void pointers are not supported in Triton"),
                _ => parpy_internal_error!(i, "Failed to convert pointer to a valid Triton type")
            }
        },
        gpu_ast::Type::Pointer {ty, mem: gpu_ast::MemSpace::Host} => {
            match *ty {
                gpu_ast::Type::Function {result, args} => {
                    let result = Box::new(from_gpu_ast_type(*result, &i)?);
                    let args = args.into_iter()
                        .map(|ty| from_gpu_ast_type(ty, &i))
                        .collect::<CompileResult<Vec<Type>>>()?;
                    Ok(Type::Function {result, args})
                },
                _ => parpy_internal_error!(i, "Unsupported host pointer type")
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
            let load_expr = Expr::Load {ptr, mask: None, ty: ty.clone(), i: i.clone()};
            match &ty {
                Type::Pointer {..} => {
                    Ok(Expr::Convert { value: Box::new(load_expr), ty, i })
                },
                _ => Ok(load_expr)
            }
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
        gpu_ast::Expr::Convert {e, ty} => {
            let i = e.get_info();
            let ty = from_gpu_ast_type(ty, &i)?;
            let value = Box::new(from_gpu_ast_kernel_expr(env, *e)?);
            Ok(Expr::Convert {value, ty, i})
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
            (gpu_ast::Expr::Var {id, ..}, BinOp::Lt | BinOp::Gt) if id == *var => {
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
) -> CompileResult<i128> {
    let fail = |i: Info| {
        parpy_compile_error!(i, "Failed to extract step size of for-loop in Triton codegen")
    };
    if let gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} = incr {
        match (*lhs, op) {
            (gpu_ast::Expr::Var {id, ..}, BinOp::Add) if id == *var => {
                match from_gpu_ast_kernel_expr(env, *rhs)? {
                    Expr::Int {v, ..} => Ok(v),
                    _ => fail(i)
                }
            },
            _ => fail(i)
        }
    } else {
        fail(incr.get_info())
    }
}

fn refers_to_block_dim(acc: Option<Dim>, e: &Expr) -> Option<Dim> {
    match e {
        Expr::ProgramId {dim, ..} => Some(dim.clone()),
        _ => e.sfold(acc, refers_to_block_dim)
    }
}

fn determine_step_size(
    env: &CodegenEnv,
    step: i128,
    lo: &Expr
) -> i128 {
    let nthreads = env.current_grid.threads.prod() as i128;
    match lo.sfold(None, refers_to_block_dim) {
        Some(dim) => {
            let nblocks = env.current_grid.blocks.get_dim(&dim) as i128;
            step / (nblocks * nthreads)
        },
        None => step / nthreads
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
) -> CompileResult<(Name, Expr, Expr, Expr, Option<Stmt>)> {
    let (removed_thread, init) = remove_thread_idx(false, init);
    let lo = from_gpu_ast_kernel_expr(env, init)?;
    let hi = extract_upper_bound(env, &var, cond)?;
    let step_val = extract_step(env, &var, incr)?;
    let step = Expr::Int {
        v: step_val,
        ty: hi.get_type().clone(),
        i: i.clone()
    };
    // NOTE(larshum, 2026-02-11): If we removed the use of a thread index, we know this for-loop
    // runs in parallel over threads, in which case we have to restructure it.
    if removed_thread {
        let step_size = determine_step_size(&env, step_val, &lo);
        let nthreads = env.current_grid.threads.prod() as i128;
        let new_var = Name::new(format!("{0}_chunk", var.get_str())).with_new_sym();
        let var_assign = Stmt::Definition {
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
                        lo: Box::new(Expr::Int {
                            v: 0,
                            ty: var_ty.clone(),
                            i: i.clone()
                        }),
                        hi: Box::new(Expr::Int {
                            v: nthreads,
                            ty: var_ty.clone(),
                            i: i.clone()
                        }),
                        ty: var_ty.clone(),
                        i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: step_size,
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

fn can_omit_explicit_conversion(s: &ScalarSizes, ty: &Type) -> bool {
    match ty {
        Type::Tensor {sz, ..} => *sz == s.int || *sz == s.float,
        _ => false,
    }
}

fn from_gpu_ast_kernel_stmt(
    env: &CodegenEnv,
    mut acc: Vec<Stmt>,
    s: gpu_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        gpu_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = from_gpu_ast_type(ty, &i)?;
            let expr = from_gpu_ast_kernel_expr(env, expr)?;
            let expr = if can_omit_explicit_conversion(&env.sz, &ty) {
                expr
            } else {
                let i = expr.get_info();
                Expr::Convert {value: Box::new(expr), ty, i}
            };
            acc.push(Stmt::Definition {dst: id, expr, i});
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
            let ptr_ty = Type::Pointer {
                ty: Box::new(ty.clone()),
                shape: Shape::Num(1)
            };
            let rhs = from_gpu_ast_kernel_expr(env, *rhs)?;
            match *lhs {
                gpu_ast::Expr::Var {id, ty: _, i: _} => {
                    acc.push(Stmt::Assign {dst: id, expr: rhs, i});
                    Ok(acc)
                },
                gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i: _} => {
                    let target = from_gpu_ast_kernel_expr(env, *target)?;
                    let idx = from_gpu_ast_kernel_expr(env, *idx)?;
                    acc.push(Stmt::Store {
                        ptr: Expr::BinOp {
                            lhs: Box::new(target),
                            op: BinOp::Add,
                            rhs: Box::new(idx.with_type(ptr_ty.clone())),
                            ty: ptr_ty,
                            i: i.clone()
                        },
                        value: rhs,
                        mask: None,
                        i: i.clone()
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
        gpu_ast::Expr::Convert {e, ty: _} => {
            let i = e.get_info();
            let value = Box::new(from_gpu_ast_host_expr(env, *e)?);
            Ok(Expr::Convert {value, ty, i})
        },
        gpu_ast::Expr::ThreadIdx {i, ..} => {
            parpy_internal_error!(i, "Thread indices are not supported")
        },
        gpu_ast::Expr::BlockIdx {dim, ty: _, i} => {
            Ok(Expr::ProgramId {dim, ty, i})
        },
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
            acc.push(Stmt::Definition {dst: id, expr, i});
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
                .map(|e| from_gpu_ast_host_expr(&env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let nwarps = (grid.threads.prod() / par::WARP_SIZE) as usize;
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
            let elem_sz = match &elem_ty {
                Type::Pointer {..} => Ok(ElemSize::I64),
                Type::Tensor {sz, ..} => Ok(sz.clone()),
                Type::Function {..} |
                Type::List |
                Type::String |
                Type::Void => {
                    parpy_internal_error!(i, "Found allocation of unsupported type \
                                              {elem_ty:?} in Triton codegen")
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

fn sub_buffers_expr(
    callback_params: &BTreeSet<Name>,
    sub_map: &BTreeMap<Name, Expr>,
    e: Expr
) -> Expr {
    match e {
        Expr::Var {id, ty, i} => {
            match sub_map.get(&id) {
                Some(e) => e.clone(),
                None => Expr::Var {id, ty, i}
            }
        },
        Expr::Call {ref id, ..} if callback_params.contains(&id) => e,
        _ => e.smap(|e| sub_buffers_expr(&callback_params, &sub_map, e))
    }
}

fn sub_buffers_stmt(
    callback_params: &BTreeSet<Name>,
    sub_map: &BTreeMap<Name, Expr>,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::Assign {ref dst, expr: Expr::AllocBuffer {..}, ..} => {
            // After each allocation of a temporary buffer, we insert a definition of a variable
            // containing the inner PyTorch tensor.
            if let Some(Expr::Var {id, ty, i}) = sub_map.get(&dst) {
                let torch_def = Stmt::Definition {
                    dst: id.clone(),
                    expr: Expr::ToTorch {
                        e: Box::new(Expr::Var {
                            id: dst.clone(),
                            ty: ty.clone(),
                            i: i.clone()
                        }),
                        ty: ty.clone(),
                        i: i.clone()
                    },
                    i: i.clone()
                };
                acc.push(s);
                acc.push(torch_def);
            } else {
                acc.push(s);
            }
            acc
        }
        _ => {
            s.smap(|e| sub_buffers_expr(&callback_params, &sub_map, e))
                .sflatten(acc, |acc, s| sub_buffers_stmt(&callback_params, &sub_map, acc, s))
        }
    }
}

fn collect_buffer_alloc_subs(
    mut sub_map: BTreeMap<Name, Expr>,
    s: &Stmt
) -> BTreeMap<Name, Expr> {
    match s {
        Stmt::Assign {dst, expr: Expr::AllocBuffer {ty, i, ..}, ..} => {
            let sub_expr = Expr::Var {
                id: dst.clone().with_new_sym(),
                ty: ty.clone(),
                i: i.clone(),
            };
            sub_map.insert(dst.clone(), sub_expr);
            sub_map
        },
        _ => s.sfold(sub_map, collect_buffer_alloc_subs)
    }
}

fn add_buffer_to_torch_conversion(
    params: &Vec<Param>,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    let is_pointer_type = |p: &Param| match &p.ty {
        Type::Pointer {..} => true,
        _ => false
    };
    let is_function_type = |p: &Param| match &p.ty {
        Type::Function {..} => true,
        _ => false
    };
    let pointer_params = params.iter()
        .filter(|p| is_pointer_type(&p))
        .cloned()
        .collect::<Vec<Param>>();
    let callback_params = params.iter()
        .filter(|p| is_function_type(&p))
        .map(|Param {id, ..}| id.clone())
        .collect::<BTreeSet<Name>>();
    let new_ids = pointer_params.iter()
        .map(|p| p.id.clone().with_new_sym())
        .collect::<Vec<Name>>();
    let sub_map = new_ids.iter()
        .zip(pointer_params.iter())
        .map(|(new_id, Param {id, ty, annot_ty: _, i})| {
            (id.clone(), Expr::Var {
                id: new_id.clone(),
                ty: ty.clone(),
                i: i.clone()
            })
        })
        .collect::<BTreeMap<Name, Expr>>();
    let sub_map = body.sfold(sub_map, collect_buffer_alloc_subs);
    // NOTE(larshum, 2026-03-04): Each Triton kernel expects PyTorch tensors as arguments. However,
    // we use ParPy buffers within the host function. Converting a ParPy buffer to a PyTorch
    // buffer is fairly cheap, but if we do this a large number of time it has a big performance
    // impact. To mitigate this, we convert all buffer arguments and allocated intermediate data
    // once, and store their PyTorch tensors in separate variables, which we use immediately
    // instead of converting the arguments when calling the Triton kernel.
    //
    // However, we still track the ParPy buffers, as they are used when invoking a callback
    // function.
    let body = body.sflatten(vec![], |acc, s| {
        sub_buffers_stmt(&callback_params, &sub_map, acc, s)
    });
    let body = new_ids.into_iter()
        .zip(pointer_params.into_iter())
        .map(|(new_id, Param {id, ty, annot_ty: _, i})| Stmt::Definition {
            dst: new_id,
            expr: Expr::ToTorch {
                e: Box::new(Expr::Var {id, ty: ty.clone(), i: i.clone()}),
                ty,
                i: i.clone()
            },
            i
        })
        .chain(body.into_iter())
        .collect::<Vec<Stmt>>();
    Ok(body)
}

fn elem_size_to_triton_signature(sz: &ElemSize) -> String {
    match sz {
        ElemSize::Bool => "i1",
        ElemSize::I8 => "i8",
        ElemSize::I16 => "i16",
        ElemSize::I32 => "i32",
        ElemSize::I64 => "i64",
        ElemSize::U8 => "u8",
        ElemSize::U16 => "u16",
        ElemSize::U32 => "u32",
        ElemSize::U64 => "u64",
        ElemSize::F16 => "fp16",
        ElemSize::F32 => "fp32",
        ElemSize::F64 => "fp64",
    }.to_string()
}

fn type_to_triton_signature(ty: &Type, i: &Info) -> CompileResult<String> {
    match ty {
        Type::Tensor {sz, ..} => Ok(elem_size_to_triton_signature(sz)),
        Type::Pointer {ty, ..} => {
            let ty_str = type_to_triton_signature(ty, i)?;
            Ok(format!("*{ty_str}"))
        },
        Type::Function {..} |
        Type::List |
        Type::String |
        Type::Void => parpy_internal_error!(i, "Failed to generate signature of Triton kernel")
    }
}

fn make_signature_list(
    params: &Vec<Param>,
    i: &Info
) -> CompileResult<Expr> {
    let elems = params.iter()
        .map(|Param {ty, ..}| Ok(Expr::String {
            v: type_to_triton_signature(ty, i)?,
            ty: Type::String,
            i: Info::default()
        }))
        .collect::<CompileResult<Vec<Expr>>>()?;
    Ok(Expr::List {elems, ty: Type::List, i: i.clone()})
}

fn return_kernel_information(
    env: &CodegenEnv,
    s: Stmt
) -> CompileResult<Stmt> {
    // We make the host entry point function of the generated Python code return a list of the
    // defined Triton kernels, instead of simply returning the integer zero. This is used in the
    // native code generation to retrieve a reference to each of the defined Triton kernels for the
    // compilation stage.
    match s {
        Stmt::Return {value: _, i} => {
            let kernels = env.kernel_dims.keys()
                .map(|id| Expr::Var {id: id.clone(), ty: Type::Void, i: i.clone()})
                .collect::<Vec<Expr>>();
            let kernel_signatures = env.kernel_params.iter()
                .map(|params| make_signature_list(params, &i))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let value = Expr::List {
                elems: vec![
                    Expr::List {elems: kernels, ty: Type::List, i: i.clone()},
                    Expr::List {elems: kernel_signatures, ty: Type::List, i: i.clone()},
                ],
                ty: Type::List,
                i: i.clone()
            };
            Ok(Stmt::Return {value, i})
        },
        _ => s.smap_result(|s| return_kernel_information(&env, s))
    }
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
            validate_attrs(attrs, &i)?;
            let params = from_gpu_ast_params(params)?;
            let mut env = env.set_active_kernel(Some(&id));
            let body = from_gpu_ast_kernel_stmts(&env, body)?;
            env.kernel_params.push(params.clone());
            tops.push(Top::KernelFunDef {decorators: vec![], id, params, body, i});
            Ok((env, tops))
        },
        gpu_ast::Top::FunDef {ret_ty: _, id, params, body, target: Target::Device, i} => {
            let params = from_gpu_ast_params(params)?;
            let env = env.set_active_kernel(None);
            let body = from_gpu_ast_kernel_stmts(&env, body)?;
            tops.push(Top::KernelFunDef {decorators: vec![], id, params, body, i});
            Ok((env, tops))
        },
        gpu_ast::Top::FunDef {ret_ty: _, id, params, body, target: Target::Host, i} => {
            let params = from_gpu_ast_params(params)?;
            let body = from_gpu_ast_host_stmts(&env, body)?;
            let body = add_buffer_to_torch_conversion(&params, body)?;

            // We replace the final return statement of the entry point with a return consisting of
            // information on the kernels, including the autotuned kernel function, its signature
            // and other attributes associated with its parameters.
            let body = body.smap_result(|s| return_kernel_information(&env, s))?;
            tops.push(Top::FunDef {id, params, body, i});
            Ok((env, tops))
        },
    }
}

pub fn from_gpu_ast(ast: gpu_ast::Ast, opts: &CompileOptions) -> CompileResult<Ast> {
    let env = ast.sfold(CodegenEnv::new(opts), collect_kernel_dims_top);
    let mut tops = generate_default_imports();
    let (_, mut gen_tops) = ast.into_iter()
        .fold(Ok((env, vec![])), |acc, t| {
            let (env, tops) = acc?;
            from_gpu_ast_top(env, tops, t)
        })?;
    tops.append(&mut gen_tops);
    Ok(Ast {tops})
}
