use super::ast::*;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;
use crate::utils::smap::*;

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct MemoryUse {
    bytes: usize,
    alignment: usize
}

#[derive(Clone, Debug, PartialEq)]
struct Allocation {
    offset: usize,
    bytes: usize
}

#[derive(Clone, Debug)]
struct FunctionState {
    smem_id: Name,
    peak_usage_bytes: usize,
    smem_vars: BTreeMap<Name, MemoryUse>,
    smem_allocations: BTreeMap<Name, Allocation>
}

fn aligned_offset(offset: usize, alignment: usize) -> usize {
    if offset % alignment != 0 {
        offset + (alignment - offset % alignment)
    } else {
        offset
    }
}

impl FunctionState {
    fn add_alloc(mut self, id: &Name, mem_use: MemoryUse) -> Self {
        let offset = self.select_offset(mem_use.alignment);
        let bytes = mem_use.bytes;
        self.smem_allocations.insert(id.clone(), Allocation {offset, bytes});
        self.peak_usage_bytes = usize::max(self.peak_usage_bytes, offset + bytes);
        self
    }

    fn remove_alloc(mut self, id: &Name) -> CompileResult<(Self, usize)> {
        match self.smem_allocations.remove(&id) {
            Some(Allocation {offset, ..}) => Ok((self, offset)),
            None => {
                parpy_internal_error!(
                    Info::default(),
                    "Failed to remove allocation in shared memory analysis"
                )
            }
        }
    }

    fn select_offset(&self, alignment: usize) -> usize {
        self.smem_allocations.values()
            .fold(0, |acc, Allocation {offset, ..}| {
                usize::max(acc, aligned_offset(*offset, alignment))
            })
    }
}

#[derive(Clone, Debug)]
struct SharedMemoryEnv {
    local_state: FunctionState,
    func_smem_use: BTreeMap<Name, MemoryUse>
}

impl SharedMemoryEnv {
    fn add_alloc(mut self, id: &Name, mem_use: MemoryUse) -> Self {
        self.local_state = self.local_state.add_alloc(id, mem_use);
        self
    }

    fn remove_alloc(mut self, id: &Name) -> CompileResult<(Self, usize)> {
        let (local_state, offset) = self.local_state.remove_alloc(id)?;
        self.local_state = local_state;
        Ok((self, offset))
    }
}

fn size_of_elem_size(sz: &ElemSize) -> usize {
    match sz {
        ElemSize::Bool | ElemSize::I8 | ElemSize::U8 => 1,
        ElemSize::I16 | ElemSize::U16 | ElemSize::F16 => 2,
        ElemSize::I32 | ElemSize::U32 | ElemSize::F32 => 4,
        ElemSize::I64 | ElemSize::U64 | ElemSize::F64 => 8,
    }
}

fn size_of_scalar_type(ty: &Type, i: &Info) -> CompileResult<usize> {
    match ty {
        Type::Scalar {sz} => Ok(size_of_elem_size(sz)),
        _ => {
            let ty_str = ty.pprint_default();
            parpy_compile_error!(i, "Expected scalar type in shared memory \
                                     allocation, found {ty_str}.")
        }
    }
}

fn collect_local_shared_memory_allocations(
    mut acc: BTreeMap<Name, MemoryUse>,
    s: &Stmt
) -> CompileResult<BTreeMap<Name, MemoryUse>> {
    match s {
        Stmt::AllocShared {ref id, ref elem_ty, ref sz, ref i, ..} => {
            let elem_sz = size_of_scalar_type(elem_ty, i)?;
            acc.insert(id.clone(), MemoryUse {
                bytes: sz * elem_sz,
                alignment: elem_sz
            });
            Ok(acc)
        },
        _ => s.sfold_result(Ok(acc), collect_local_shared_memory_allocations)
    }
}

fn make_shared_memory_pointer(
    id: &Name,
    offset: usize,
    ty: &Type,
    i: &Info
) -> Expr {
    Expr::Convert {
        e: Box::new(Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: id.clone(),
                ty: ty.clone(),
                i: i.clone()
            }),
            op: BinOp::Add,
            rhs: Box::new(Expr::Int {
                v: offset as i128,
                ty: Type::Scalar {sz: ElemSize::I64},
                i: i.clone()
            }),
            ty: ty.clone(),
            i: i.clone()
        }),
        ty: ty.clone()
    }
}

fn record_shared_memory_use_expr(
    mut env: SharedMemoryEnv,
    e: Expr
) -> CompileResult<(SharedMemoryEnv, Expr)> {
    match e {
        Expr::Var {id, ty, i} => {
            match env.local_state.smem_vars.remove(&id) {
                Some(mem_use) => {
                    let env = env.add_alloc(&id, mem_use);
                    Ok((env, Expr::Var {id, ty, i}))
                },
                None => Ok((env, Expr::Var {id, ty, i}))
            }
        },
        Expr::Call {id, args, ty, i} if env.func_smem_use.contains_key(&id) => {
            let (env, mut args) = args
                .smap_accum_l_result(Ok(env), record_shared_memory_use_expr)?;
            let mem_use = env.func_smem_use.get(&id).cloned().unwrap();
            let env = env.add_alloc(&id, mem_use.clone());
            let (env, offset) = env.remove_alloc(&id)?;
            let args = if offset > 0 {
                let ptr_ty = Type::Pointer {
                    ty: Box::new(Type::Scalar {sz: ElemSize::I8}),
                    mem: MemSpace::Shared
                };
                args.push(make_shared_memory_pointer(
                    &env.local_state.smem_id, offset, &ptr_ty, &i
                ));
                args
            } else {
                args
            };
            Ok((env, Expr::Call {id, args, ty, i}))
        },
        _ => e.smap_accum_l_result(Ok(env), record_shared_memory_use_expr)
    }
}

fn record_shared_memory_use_stmt(
    env: SharedMemoryEnv,
    s: Stmt
) -> CompileResult<(SharedMemoryEnv, Stmt)> {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
            let (env, init) = record_shared_memory_use_expr(env, init)?;
            let (env, cond) = record_shared_memory_use_expr(env, cond)?;
            let (env, incr) = record_shared_memory_use_expr(env, incr)?;
            let (env, body) = record_shared_memory_use_stmts_rev(env, body)?;
            Ok((env, Stmt::For {var_ty, var, init, cond, incr, body, unroll, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let (env, cond) = record_shared_memory_use_expr(env, cond)?;
            let (env, thn) = record_shared_memory_use_stmts_rev(env, thn)?;
            let (env, els) = record_shared_memory_use_stmts_rev(env, els)?;
            Ok((env, Stmt::If {cond, thn, els, i}))
        },
        Stmt::While {cond, body, i} => {
            let (env, cond) = record_shared_memory_use_expr(env, cond)?;
            let (env, body) = record_shared_memory_use_stmts_rev(env, body)?;
            Ok((env, Stmt::While {cond, body, i}))
        },
        Stmt::Scope {body, i} => {
            let (env, body) = record_shared_memory_use_stmts_rev(env, body)?;
            Ok((env, Stmt::Scope {body, i}))
        }
        Stmt::AllocShared {elem_ty, id, sz: _, i} => {
            let (env, offset) = env.remove_alloc(&id)?;
            let ty = Type::Pointer {ty: Box::new(elem_ty), mem: MemSpace::Shared};
            let smem_ptr = make_shared_memory_pointer(&env.local_state.smem_id, offset, &ty, &i);
            let def = Stmt::Definition {
                ty: ty.clone(), id, expr: smem_ptr, i
            };
            Ok((env, def))
        },
        _ => {
            let (env, s) = s.smap_accum_l_result(Ok(env), record_shared_memory_use_stmt)?;
            s.smap_accum_l_result(Ok(env), record_shared_memory_use_expr)
        }
    }
}

fn record_shared_memory_use_stmts_rev(
    env: SharedMemoryEnv,
    stmts: Vec<Stmt>
) -> CompileResult<(SharedMemoryEnv, Vec<Stmt>)> {
    let (env, rev_stmts) = stmts.into_iter()
        .rev()
        .fold(Ok((env, vec![])), |acc, s| {
            let (env, mut rev_stmts) = acc?;
            let (env, s) = record_shared_memory_use_stmt(env, s)?;
            rev_stmts.push(s);
            Ok((env, rev_stmts))
        })?;
    Ok((env, rev_stmts.into_iter().rev().collect::<Vec<Stmt>>()))
}

fn determine_shared_memory_use(
    function_smem: BTreeMap<Name, MemoryUse>,
    body: Vec<Stmt>
) -> CompileResult<(SharedMemoryEnv, Vec<Stmt>)> {
    let alloc_smem = body.sfold_result(Ok(BTreeMap::new()), |acc, s| {
        collect_local_shared_memory_allocations(acc, s)
    })?;
    let env = SharedMemoryEnv {
        local_state: FunctionState {
            smem_id: Name::sym_str("smem"),
            peak_usage_bytes: 0,
            smem_vars: alloc_smem,
            smem_allocations: BTreeMap::new()
        },
        func_smem_use: function_smem
    };
    record_shared_memory_use_stmts_rev(env, body)
}

fn apply_top(
    acc: BTreeMap<Name, MemoryUse>,
    t: Top
) -> CompileResult<(BTreeMap<Name, MemoryUse>, Top)> {
    match t {
        Top::ExtDecl {..} => Ok((acc, t)),
        Top::KernelFunDef {mut attrs, id, params, body, i} => {
            let (env, body) = determine_shared_memory_use(acc, body)?;
            let SharedMemoryEnv {local_state, mut func_smem_use} = env;
            if local_state.peak_usage_bytes > 0 {
                attrs.push(KernelAttribute::SharedMemory {
                    id: local_state.smem_id,
                    bytes: local_state.peak_usage_bytes
                });
            }
            let mem_use = MemoryUse {
                bytes: local_state.peak_usage_bytes,
                alignment: 1
            };
            func_smem_use.insert(id.clone(), mem_use);
            Ok((func_smem_use, Top::KernelFunDef {attrs, id, params, body, i}))
        },
        Top::FunDef {ret_ty, id, mut params, body, target, i} => {
            let (env, body) = determine_shared_memory_use(acc, body)?;
            let SharedMemoryEnv {local_state, mut func_smem_use} = env;
            // If this function uses a non-zero amount of shared memory, we add a parameter
            // containing the shared memory, which is provided by the calling function.
            let params = if local_state.peak_usage_bytes > 0 {
                let ptr_ty = Type::Pointer {
                    ty: Box::new(Type::Scalar {sz: ElemSize::I8}),
                    mem: MemSpace::Shared
                };
                params.push(Param {
                    id: local_state.smem_id,
                    ty: ptr_ty,
                    i: i.clone()
                });
                params
            } else {
                params
            };
            let mem_use = MemoryUse {
                bytes: local_state.peak_usage_bytes,
                alignment: 1
            };
            func_smem_use.insert(id.clone(), mem_use);
            Ok((func_smem_use, Top::FunDef {ret_ty, id, params, body, target, i}))
        },
    }
}

fn set_smem_in_kernel_launch_stmt(
    smem_use: &BTreeMap<Name, MemoryUse>,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::KernelLaunch {id, args, grid, smem, i} => {
            match smem_use.get(&id) {
                Some(MemoryUse {bytes, ..}) => {
                    Stmt::KernelLaunch {id, args, grid, smem: *bytes, i}
                },
                None => Stmt::KernelLaunch {id, args, grid, smem, i}
            }
        },
        _ => s.smap(|s| set_smem_in_kernel_launch_stmt(smem_use, s))
    }
}

fn set_smem_in_kernel_launch(
    smem_use: &BTreeMap<Name, MemoryUse>,
    t: Top
) -> Top {
    match t {
        Top::FunDef {ret_ty, id, params, body, target, i} => {
            let body = body.smap(|s| set_smem_in_kernel_launch_stmt(&smem_use, s));
            Top::FunDef {ret_ty, id, params, body, target, i}
        },
        _ => t
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    let (smem_use, ast) = ast.smap_accum_l_result(Ok(BTreeMap::new()), apply_top)?;
    Ok(ast.smap(|t| set_smem_in_kernel_launch(&smem_use, t)))
}
