/// This file defines a transformation for fusing memory operations targeting the same (static)
/// location in global memory. The purpose of these transformations is to optimize the code by
/// reducing the number of memory operations into global memory. It enables removing reads and
/// writes that the underlying compiler may not do. The transformation consists of three passes:
///
/// 1. Rewrite all assignments to a memory location such that the value is first assigned to a
///    local variable, which is then written to global memory. For instance, if we have a write
///    operation 'x[i] = e', then it is rewritten as two statements:
///     1) A definition introducing a new local variable of the same type 'T' as the expression
///        'e': 'T t = e;'.
///     2) An assignment of the local variable to the memory location: 'x[i] = e'.
///
///    This makes the later passes more straightforward to implement. It should not introduce any
///    extra overheads as the underlying compiler will optimize this temporary variable away when
///    it is only used once.
///
/// 2. We replace any references to memory locations that were previously written to by a reference
///    to the local variable containing its value. For instance, the statement 'y[i] = x[i]' reads
///    from memory. If 'x[i]' was assigned to previously, and its temporary variable 't' is in
///    scope, we rewrite the assignment as 'y[i] = t'. This avoids an extra read from global
///    memory.
///
/// 3. If we have two (or more) subsequent writes to the same location in memory, with no read
///    operations to the same target array in-between, we can remove all writes except the last
///    one. This is because the former ones are unused within the kernel, and they will never be
///    visible after the kernel finishes.

use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFlatten, SFold, SMapAccum};

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ArrayLoc {
    id: Name,
    idx: Expr
}

#[derive(Clone, Debug)]
struct FuseEnv {
    shared_mem_vars: BTreeSet<Name>,
    write_locs: BTreeMap<ArrayLoc, Name>,
    write_state: BTreeMap<Name, BTreeSet<Expr>>,
}

impl Default for FuseEnv {
    fn default() -> FuseEnv {
        FuseEnv {
            shared_mem_vars: BTreeSet::new(),
            write_locs: BTreeMap::new(),
            write_state: BTreeMap::new()
        }
    }
}

fn try_extract_conditional_assignment(thn: &Expr, els: &Expr) -> Option<(Expr, Expr)> {
    if let Expr::Convert {ty: Type::Void, ..} = els {
        if let Expr::Convert {e, ty: Type::Void} = thn {
            if let Expr::Assign {lhs, rhs, ..} = e.as_ref() {
                if let Expr::ArrayAccess {..} = lhs.as_ref() {
                    Some((*lhs.clone(), *rhs.clone()))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

fn store_write_results_in_temporary_variable(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Expr {e: Expr::IfExpr {cond, thn, els, ty: cond_ty, i}, i: s_i} => {
            if let Some((lhs, rhs)) = try_extract_conditional_assignment(&thn, &els) {
                let ty = rhs.get_type().clone();
                let id = Name::sym_str("t");
                acc.push(Stmt::Definition {
                    ty: ty.clone(),
                    id: id.clone(),
                    expr: Expr::IfExpr {
                        cond: cond.clone(),
                        thn: Box::new(rhs),
                        els: Box::new(Expr::Convert {
                            e: Box::new(Expr::Int {
                                v: 0,
                                ty: Type::Scalar {sz: ElemSize::I32},
                                i: i.clone()
                            }),
                            ty: ty.clone()
                        }),
                        ty: ty.clone(),
                        i: i.clone()
                    },
                    i: i.clone()
                });
                let thn = Box::new(Expr::Convert {
                    e: Box::new(Expr::Assign {
                        lhs: Box::new(lhs),
                        rhs: Box::new(Expr::Var {id, ty: ty.clone(), i: i.clone()}),
                        ty: ty.clone(),
                        i: i.clone()
                    }),
                    ty: Type::Void
                });
                acc.push(Stmt::Expr {
                    e: Expr::IfExpr {cond, thn, els, ty: cond_ty, i},
                    i: s_i
                });
            } else {
                acc.push(Stmt::Expr {
                    e: Expr::IfExpr {cond, thn, els, ty: cond_ty, i},
                    i: s_i
                });
            }
            acc
        },
        Stmt::Expr {e: Expr::Assign {lhs, rhs, ty: assign_ty, i}, i: s_i} => {
            if let Expr::ArrayAccess {..} = lhs.as_ref() {
                let ty = rhs.get_type().clone();
                let id = Name::sym_str("t");
                acc.push(Stmt::Definition {
                    ty: ty.clone(),
                    id: id.clone(),
                    expr: *rhs,
                    i: i.clone()
                });
                acc.push(Stmt::Expr {
                    e: Expr::Assign {
                        lhs,
                        rhs: Box::new(Expr::Var {id, ty, i: i.clone()}),
                        ty: assign_ty,
                        i
                    },
                    i: s_i
                });
                acc
            } else {
                acc.push(Stmt::Expr {
                    e: Expr::Assign {
                        lhs, rhs, ty: assign_ty, i
                    },
                    i: s_i
                });
                acc
            }
        },
        _ => s.sflatten(acc, store_write_results_in_temporary_variable)
    }
}

fn memory_operation_array_location(
    env: &FuseEnv,
    e: &Expr
) -> Option<ArrayLoc> {
    if let Expr::ArrayAccess {target, idx, ..} = e {
        match target.as_ref() {
            Expr::Var {id, ..} if !env.shared_mem_vars.contains(&id) => {
                Some(ArrayLoc {
                    id: id.clone(),
                    idx: *idx.clone()
                })
            },
            _ => None
        }
    } else {
        None
    }
}

fn collect_shared_memory_variables(mut env: FuseEnv, s: &Stmt) -> FuseEnv {
    match s {
        Stmt::AllocShared {id, ..} => {
            env.shared_mem_vars.insert(id.clone());
            env
        },
        _ => s.sfold(env, collect_shared_memory_variables)
    }
}

fn replace_reads_with_locals_expr(env: FuseEnv, e: Expr) -> (FuseEnv, Expr) {
    match e {
        Expr::Assign {lhs, rhs, ty, i} => {
            let reconstruct_assign = |lhs, rhs, ty, i| Expr::Assign {
                lhs, rhs: Box::new(rhs), ty, i
            };
            let (mut env, rhs) = replace_reads_with_locals_expr(env, *rhs);
            if let Some(loc) = memory_operation_array_location(&env, &lhs) {
                if let Expr::Var {ref id, ..} = &rhs {
                    env.write_locs.insert(loc, id.clone());
                    (env, reconstruct_assign(lhs, rhs, ty, i))
                } else {
                    (env, reconstruct_assign(lhs, rhs, ty, i))
                }
            } else {
                (env, reconstruct_assign(lhs, rhs, ty, i))
            }
        },
        Expr::ArrayAccess {ref ty, ref i, ..} => {
            if let Some(loc) = memory_operation_array_location(&env, &e) {
                let e = match env.write_locs.get(&loc) {
                    Some(id) => {
                        Expr::Var {id: id.clone(), ty: ty.clone(), i: i.clone()}
                    },
                    None => e
                };
                (env, e)
            } else {
                (env, e)
            }
        },
        _ => e.smap_accum_l(env, replace_reads_with_locals_expr)
    }
}

fn filter_write_locs(
    mut env: FuseEnv,
    inner_write_locs: BTreeMap<ArrayLoc, Name>
) -> FuseEnv {
    // NOTE(larshum, 2025-12-10): We may have performed writes to some arrays within a nested scope
    // (e.g., inside a for-loop). Any arrays that were written to in this scope are ignored
    // following the loop, as their values may have changed relative to any local variables.
    let used_arrays = inner_write_locs.keys()
        .map(|loc| loc.id.clone())
        .collect::<BTreeSet<Name>>();
    env.write_locs = env.write_locs.into_iter()
        .filter(|(loc, _)| !used_arrays.contains(&loc.id))
        .collect::<BTreeMap<ArrayLoc, Name>>();
    env
}

fn replace_reads_with_locals_stmt(env: FuseEnv, s: Stmt) -> (FuseEnv, Stmt) {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
            let inner_env = FuseEnv {write_locs: BTreeMap::new(), ..env.clone()};
            let (inner_env, body) = body.smap_accum_l(inner_env, replace_reads_with_locals_stmt);
            let env = filter_write_locs(env, inner_env.write_locs);
            (env, Stmt::For {var_ty, var, init, cond, incr, body, unroll, i})
        },
        Stmt::If {cond, thn, els, i} => {
            let inner_env = FuseEnv {write_locs: BTreeMap::new(), ..env.clone()};
            let (thn_env, thn) = thn.smap_accum_l(inner_env.clone(), replace_reads_with_locals_stmt);
            let (els_env, els) = els.smap_accum_l(inner_env, replace_reads_with_locals_stmt);
            let env = filter_write_locs(env, thn_env.write_locs);
            let env = filter_write_locs(env, els_env.write_locs);
            (env, Stmt::If {cond, thn, els, i})
        },
        Stmt::While {cond, body, i} => {
            let inner_env = FuseEnv {write_locs: BTreeMap::new(), ..env.clone()};
            let (inner_env, body) = body.smap_accum_l(inner_env, replace_reads_with_locals_stmt);
            let env = filter_write_locs(env, inner_env.write_locs);
            (env, Stmt::While {cond, body, i})
        },
        Stmt::Scope {body, i} => {
            let inner_env = FuseEnv {write_locs: BTreeMap::new(), ..env.clone()};
            let (inner_env, body) = body.smap_accum_l(inner_env, replace_reads_with_locals_stmt);
            let env = filter_write_locs(env, inner_env.write_locs);
            (env, Stmt::Scope {body, i})
        },
        _ => {
            let (env, s) = s.smap_accum_l(env, replace_reads_with_locals_stmt);
            s.smap_accum_l(env, replace_reads_with_locals_expr)
        }
    }
}

fn collect_read_operations_expr(mut env: FuseEnv, e: &Expr) -> FuseEnv {
    match e {
        Expr::Assign {rhs, ..} => {
            // The left-hand side of an assignment never involves a read operation.
            collect_read_operations_expr(env, rhs)
        },
        Expr::ArrayAccess {..} => {
            if let Some(loc) = memory_operation_array_location(&env, &e) {
                // When we find a read from a specific target, we reset the tracker of all write
                // operations referring to this target.
                let ArrayLoc {id, ..} = loc;
                env.write_state.remove(&id);
                env
            } else {
                env
            }
        },
        _ => e.sfold(env, collect_read_operations_expr)
    }
}

fn collect_read_operations_stmt(env: FuseEnv, s: &Stmt) -> FuseEnv {
    let env = s.sfold(env, collect_read_operations_stmt);
    s.sfold(env, collect_read_operations_expr)
}

fn handle_write_operation<F: FnOnce() -> Stmt>(
    mut env: FuseEnv,
    reconstruct_stmt: F,
    loc: ArrayLoc
) -> (FuseEnv, Option<Stmt>) {
    let ArrayLoc {id, idx} = loc;
    match env.write_state.get_mut(&id) {
        Some(id_states) => {
            if id_states.contains(&idx) {
                (env, None)
            } else {
                id_states.insert(idx);
                (env, Some(reconstruct_stmt()))
            }
        },
        None => {
            let mut id_states = BTreeSet::new();
            id_states.insert(idx);
            env.write_state.insert(id, id_states);
            (env, Some(reconstruct_stmt()))
        }
    }
}

fn remove_repeated_writes_stmt(env: FuseEnv, s: Stmt) -> (FuseEnv, Option<Stmt>) {
    match s {
        Stmt::Expr {e: Expr::IfExpr {cond, thn, els, ty, i}, i: s_i} => {
            let reconstruct_stmt = |cond, thn, els, ty, i, s_i| {
                Stmt::Expr {
                    e: Expr::IfExpr {
                        cond, thn, els, ty, i
                    },
                    i: s_i
                }
            };
            if let Some((lhs, _)) = try_extract_conditional_assignment(&thn, &els) {
                if let Some(loc) = memory_operation_array_location(&env, &lhs) {
                    let wrap = || reconstruct_stmt(cond, thn, els, ty, i, s_i);
                    handle_write_operation(env, wrap, loc)
                } else {
                    (env, Some(reconstruct_stmt(cond, thn, els, ty, i, s_i)))
                }
            } else {
                (env, Some(reconstruct_stmt(cond, thn, els, ty, i, s_i)))
            }
        },
        Stmt::Expr {e: Expr::Assign {lhs, rhs, ty, i}, i: s_i} => {
            let reconstruct_stmt = |lhs, rhs, ty, i, s_i| {
                Stmt::Expr {
                    e: Expr::Assign {
                        lhs, rhs, ty, i
                    },
                    i: s_i
                }
            };
            if let Some(loc) = memory_operation_array_location(&env, &lhs) {
                // When we find a write to a specific location, we mark this location as pending.
                // If the location is already marked as pending, we remove this statement.
                let wrap = || reconstruct_stmt(lhs, rhs, ty, i, s_i);
                handle_write_operation(env, wrap, loc)
            } else {
                let env = collect_read_operations_expr(env, &rhs);
                (env, Some(reconstruct_stmt(lhs, rhs, ty, i, s_i)))
            }
        },
        Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
            let env = body.sfold(env, collect_read_operations_stmt);
            let inner_env = FuseEnv {write_state: BTreeMap::new(), ..env.clone()};
            let body = remove_repeated_writes(inner_env, body);
            (env, Some(Stmt::For {var_ty, var, init, cond, incr, body, unroll, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let env = thn.sfold(env, collect_read_operations_stmt);
            let env = els.sfold(env, collect_read_operations_stmt);
            let inner_env = FuseEnv {write_state: BTreeMap::new(), ..env.clone()};
            let thn = remove_repeated_writes(inner_env.clone(), thn);
            let els = remove_repeated_writes(inner_env, els);
            (env, Some(Stmt::If {cond, thn, els, i}))
        },
        Stmt::While {cond, body, i} => {
            let env = body.sfold(env, collect_read_operations_stmt);
            let inner_env = FuseEnv {write_state: BTreeMap::new(), ..env.clone()};
            let body = remove_repeated_writes(inner_env, body);
            (env, Some(Stmt::While {cond, body, i}))
        },
        Stmt::Scope {body, i} => {
            let env = body.sfold(env, collect_read_operations_stmt);
            let inner_env = FuseEnv {write_state: BTreeMap::new(), ..env.clone()};
            let body = remove_repeated_writes(inner_env, body);
            (env, Some(Stmt::Scope {body, i}))
        },
        _ => {
            let env = s.sfold(env, collect_read_operations_expr);
            (env, Some(s))
        }
    }
}

fn remove_repeated_writes(env: FuseEnv, body: Vec<Stmt>) -> Vec<Stmt> {
    let (_, body) = body.into_iter()
        .rev()
        .fold((env, vec![]), |(env, mut acc), s| {
            let (env, o) = remove_repeated_writes_stmt(env, s);
            if let Some(s) = o {
                acc.push(s);
            }
            (env, acc)
        });
    body.into_iter().rev().collect::<Vec<Stmt>>()
}

fn apply_kernel_body(body: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    // 1. For every write to memory, we store the right-hand side expression in a temporary
    //    variable first to simplify the later passes. Importantly, we assume the underlying
    //    compiler can optimize this extra variable away if it ends up being unused.
    let body = body.sflatten(vec![], store_write_results_in_temporary_variable);

    // 2. If we read from memory after writing to it previously, without any writes in-between,  we
    //    can now refer to the temporary variable in which the written value is stored instead.
    //    This can eliminate unnecessary loads from global memory.
    let env = FuseEnv::default();
    let env = body.sfold(env, collect_shared_memory_variables);
    let (env, body) = body.smap_accum_l(env, replace_reads_with_locals_stmt);

    // 3. Finally, if we perform subsequent writes to the same (static) location in memory several
    //    times, without reading from the target in-between, we can eliminate the former write to
    //    improve performance. This is safe because the former write has no impact on the kernel
    //    outcome, and it can never be observed outside the kernel.
    Ok(remove_repeated_writes(env, body))
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = apply_kernel_body(body)?;
            Ok(Top::KernelFunDef {attrs, id, params, body, i})
        },
        _ => Ok(t),
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(apply_top)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::gpu::ast_builder::*;
    use crate::utils::normalize_symbols::NormalizeSym;
    use crate::utils::pprint::*;

    fn pprint_body(body: Vec<Stmt>) -> String {
        pprint_iter(body.iter(), PrettyPrintEnv::default(), "\n").1
    }

    fn assert_eq_bodies(l: Vec<Stmt>, r: Vec<Stmt>) {
        let lstr = pprint_body(l.clone());
        let rstr = pprint_body(r.clone());
        assert_eq!(
            l.normalize_symbols(),
            r.normalize_symbols(),
            "LHS:\n{lstr}\nRHS:\n{rstr}",
        );
    }

    #[test]
    fn write_to_local() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![assign(loc.clone(), float(1.0, Some(ElemSize::F32)))];
        let t = Name::sym_str("t");
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t.clone(),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(loc, Expr::Var {id: t, ty: scalar(ElemSize::F32), i: i()})
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }

    #[test]
    fn eliminate_repeated_write() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            assign(loc.clone(), float(2.0, Some(ElemSize::F32)))
        ];
        let t = Name::sym_str("t");
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: Name::sym_str("t"),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t.clone(),
                expr: float(2.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(loc, Expr::Var {id: t, ty: scalar(ElemSize::F32), i: i()})
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }

    #[test]
    fn read_after_write_use_local_variable() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            Stmt::Expr {
                e: binop(
                    loc.clone(),
                    BinOp::Add,
                    float(2.0, Some(ElemSize::F32)),
                    scalar(ElemSize::F32)
                ),
                i: i()
            }
        ];
        let t = Name::sym_str("t");
        let var_t = Expr::Var {id: t.clone(), ty: scalar(ElemSize::F32), i: i()};
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t.clone(),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(loc.clone(), var_t.clone()),
            Stmt::Expr {
                e: binop(
                   var_t.clone(),
                   BinOp::Add,
                   float(2.0, Some(ElemSize::F32)),
                   scalar(ElemSize::F32)
                ),
                i: i()
            }
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }

    #[test]
    fn read_after_write_in_nested_scope() {
        // The second use should not be replaced with a temporary variable because the for-loop
        // could (and in this case, it actually does) write to the same location.
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::I32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::I32)
        );
        let body = vec![
            assign(loc.clone(), int(1, Some(ElemSize::I32))),
            Stmt::For {
                var_ty: scalar(ElemSize::I32), var: id("i"),
                init: int(0, Some(ElemSize::I32)),
                cond: binop(
                    var("i", scalar(ElemSize::I32)),
                    BinOp::Lt,
                    int(10, Some(ElemSize::I32)),
                    scalar(ElemSize::Bool)
                ),
                incr: binop(
                    var("i", scalar(ElemSize::I32)),
                    BinOp::Add,
                    int(1, Some(ElemSize::I32)),
                    scalar(ElemSize::I32)
                ),
                body: vec![
                    assign(
                        array_access(
                            var("x", pointer(scalar(ElemSize::I32), MemSpace::Device)),
                            var("i", scalar(ElemSize::I32)),
                            scalar(ElemSize::I32)
                        ),
                        var("i", scalar(ElemSize::I32))
                    )
                ],
                unroll: false,
                i: i()
            },
            Stmt::Expr {
                e: binop(
                    loc.clone(),
                    BinOp::Add,
                    int(2, Some(ElemSize::I32)),
                    scalar(ElemSize::I32)
                ),
                i: i()
            }
        ];
        let t = Name::sym_str("t");
        let var_t = Expr::Var {id: t.clone(), ty: scalar(ElemSize::I32), i: i()};
        let t_for = Name::sym_str("t");
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::I32),
                id: t.clone(),
                expr: int(1, Some(ElemSize::I32)),
                i: i()
            },
            assign(loc.clone(), var_t.clone()),
            Stmt::For {
                var_ty: scalar(ElemSize::I32), var: id("i"),
                init: int(0, Some(ElemSize::I32)),
                cond: binop(
                    var("i", scalar(ElemSize::I32)),
                    BinOp::Lt,
                    int(10, Some(ElemSize::I32)),
                    scalar(ElemSize::Bool)
                ),
                incr: binop(
                    var("i", scalar(ElemSize::I32)),
                    BinOp::Add,
                    int(1, Some(ElemSize::I32)),
                    scalar(ElemSize::I32)
                ),
                body: vec![
                    Stmt::Definition {
                        ty: scalar(ElemSize::I32),
                        id: t_for.clone(),
                        expr: var("i", scalar(ElemSize::I32)),
                        i: i()
                    },
                    assign(
                        array_access(
                            var("x", pointer(scalar(ElemSize::I32), MemSpace::Device)),
                            var("i", scalar(ElemSize::I32)),
                            scalar(ElemSize::I32)
                        ),
                        Expr::Var {id: t_for, ty: scalar(ElemSize::I32), i: i()}
                    )
                ],
                unroll: false,
                i: i()
            },
            Stmt::Expr {
                e: binop(
                    loc.clone(),
                    BinOp::Add,
                    int(2, Some(ElemSize::I32)),
                    scalar(ElemSize::I32)
                ),
                i: i()
            }
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }

    #[test]
    fn skip_fusion_for_shared_memory() {
        let dst = var("x", pointer(scalar(ElemSize::F32), MemSpace::Device));
        let loc = array_access(
            dst.clone(),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            Stmt::AllocShared {
                elem_ty: scalar(ElemSize::F32),
                id: Name::new("x".to_string()),
                sz: 10,
                i: i()
            },
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            assign(loc.clone(), float(2.0, Some(ElemSize::F32)))
        ];
        // We always store the RHS of a write to memory in a temporary variable, regardless of
        // whether this is a store to global memory or shared memory. However, for shared memory,
        // we do not remove repeated writes.
        let t1 = Name::sym_str("t");
        let t2 = Name::sym_str("t");
        let expected = vec![
            Stmt::AllocShared {
                elem_ty: scalar(ElemSize::F32),
                id: Name::new("x".to_string()),
                sz: 10,
                i: i()
            },
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t1.clone(),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(
                loc.clone(),
                Expr::Var {id: t1.clone(), ty: scalar(ElemSize::F32), i: i()}
            ),
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t2.clone(),
                expr: float(2.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(
                loc.clone(),
                Expr::Var {id: t2.clone(), ty: scalar(ElemSize::F32), i: i()}
            )
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }

    #[test]
    fn full_write_read_write_fusion() {
        let loc = array_access(
            var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
            int(1, Some(ElemSize::I32)),
            scalar(ElemSize::F32)
        );
        let final_update_expr = |lhs| binop(
            lhs,
            BinOp::Add,
            float(2.0, Some(ElemSize::F32)),
            scalar(ElemSize::F32)
        );
        let body = vec![
            assign(loc.clone(), float(1.0, Some(ElemSize::F32))),
            assign(var("y", scalar(ElemSize::F32)), loc.clone()),
            assign(loc.clone(), final_update_expr(loc.clone()))
        ];
        let t1 = Name::sym_str("t");
        let t2 = Name::sym_str("t");
        let var_t = |id| Expr::Var {id, ty: scalar(ElemSize::F32), i: i()};
        let expected = vec![
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t1.clone(),
                expr: float(1.0, Some(ElemSize::F32)),
                i: i()
            },
            assign(var("y", scalar(ElemSize::F32)), var_t(t1.clone())),
            Stmt::Definition {
                ty: scalar(ElemSize::F32),
                id: t2.clone(),
                expr: final_update_expr(var_t(t1)),
                i: i()
            },
            assign(loc, var_t(t2))
        ];
        assert_eq_bodies(apply_kernel_body(body).unwrap(), expected);
    }
}
