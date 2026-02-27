/// This file defines a fusion of operations that target the same (static) memory location. The
/// purpose is to reduce the number of memory operations, which can signficantly improve
/// performance of the resulting code, as the Triton compiler will not perform such optimizations.
/// This is similar to the fuse_memory implementation for the GPU AST, but it is specifically
/// tailored to the Triton AST, which operates on blocks of data.

use super::ast::*;
use super::constant_fold;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;
use crate::utils::substitute::SubVars;

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct ArangeSubEnv {
    sub: BTreeMap<Name, Expr>,
    base: Option<(Expr, Expr)>,
}

impl Default for ArangeSubEnv {
    fn default() -> Self {
        ArangeSubEnv {
            sub: BTreeMap::new(),
            base: None
        }
    }
}

fn collect_arange_components(e: &Expr) -> Option<(Expr, i128)> {
    match e {
        Expr::Arange {ty, i, ..} => {
            let ofs = Expr::Int {v: 0, ty: ty.clone(), i: i.clone()};
            Some((ofs, 1))
        },
        Expr::BinOp {lhs: ofs, op: BinOp::Add, rhs: scaled_arange, ..} => {
            match &**scaled_arange {
                Expr::Arange {..} => Some((*ofs.clone(), 1)),
                Expr::BinOp {lhs, op: BinOp::Mul, rhs, ..} => {
                    match (&**lhs, &**rhs) {
                        (Expr::Arange {..}, Expr::Int {v, ..}) => {
                            Some((*ofs.clone(), *v))
                        },
                        _ => None
                    }
                },
                _ => None
            }
        },
        _ => None
    }
}

fn substitute_definitions_expr(
    env: &ArangeSubEnv,
    e: Expr
) -> Expr {
    match e {
        Expr::Var {id, ty, i} => {
            match env.sub.get(&id) {
                Some(e) => e.clone(),
                None => Expr::Var {id, ty, i}
            }
        },
        _ => e.smap(|e| substitute_definitions_expr(env, e))
    }
}

fn remove_duplicate_arange_definitions(
    mut env: ArangeSubEnv,
    s: Stmt
) -> (ArangeSubEnv, Stmt) {
    match s {
        Stmt::Definition {dst, expr, i} => {
            let expr = substitute_definitions_expr(&env, expr);
            match collect_arange_components(&expr) {
                Some((ofs, 1)) => {
                    match env.base {
                        Some((ref base_var, ref base_ofs)) => {
                            let ty = ofs.get_type().clone();
                            let sub_expr = Expr::BinOp {
                                lhs: Box::new(base_var.clone()),
                                op: BinOp::Add,
                                rhs: Box::new(Expr::BinOp {
                                    lhs: Box::new(ofs.clone()),
                                    op: BinOp::Sub,
                                    rhs: Box::new(base_ofs.clone()),
                                    ty: ty.clone(),
                                    i: i.clone()
                                }),
                                ty: ty.clone(),
                                i: i.clone()
                            };
                            // Apply constant folding to eliminate no-op expressions, as these
                            // would complicate the identification of equivalent memory locations.
                            let sub_expr = constant_fold::fold_expr(sub_expr);
                            env.sub.insert(dst.clone(), sub_expr);
                            (env, Stmt::Pass {i})
                        },
                        None => {
                            let var = Expr::Var {
                                id: dst.clone(),
                                ty: expr.get_type().clone(),
                                i: i.clone()
                            };
                            env.base = Some((var, ofs));
                            (env, Stmt::Definition {dst, expr, i})
                        }
                    }
                },
                _ => (env, Stmt::Definition {dst, expr, i})
            }
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = substitute_definitions_expr(&env, lo);
            let hi = substitute_definitions_expr(&env, hi);
            let (_, body) = body.smap_accum_l(
                ArangeSubEnv::default(),
                remove_duplicate_arange_definitions
            );
            (env, Stmt::For {var, lo, hi, step, body, i})
        },
        Stmt::While {cond, body, i} => {
            let cond = substitute_definitions_expr(&env, cond);
            let (_, body) = body.smap_accum_l(
                ArangeSubEnv::default(),
                remove_duplicate_arange_definitions
            );
            (env, Stmt::While {cond, body, i})
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = substitute_definitions_expr(&env, cond);
            let (_, thn) = thn.smap_accum_l(
                ArangeSubEnv::default(),
                remove_duplicate_arange_definitions
            );
            let (_, els) = els.smap_accum_l(
                ArangeSubEnv::default(),
                remove_duplicate_arange_definitions
            );
            (env, Stmt::If {cond, thn, els, i})
        }
        _ => {
            let s = s.smap(|e| substitute_definitions_expr(&env, e));
            s.smap_accum_l(env, remove_duplicate_arange_definitions)
        }
    }
}

#[derive(Clone, Debug)]
struct MaskSubEnv {
    vars: BTreeMap<Name, Expr>,
    masks: BTreeMap<Expr, Expr>,
}

impl Default for MaskSubEnv {
    fn default() -> Self {
        MaskSubEnv {
            vars: BTreeMap::new(),
            masks: BTreeMap::new(),
        }
    }
}

fn is_mask_expr(e: &Expr) -> bool {
    match e {
        Expr::BinOp {lhs, op: BinOp::Lt | BinOp::Gt, rhs, ..} => {
            match (&**lhs, &**rhs) {
                (Expr::Var {..}, Expr::Int {..}) => true,
                _ => false
            }
        },
        _ => false
    }
}

fn remove_repeated_mask_definitions(
    mut env: MaskSubEnv,
    s: Stmt
) -> (MaskSubEnv, Stmt) {
    match s {
        Stmt::Definition {dst, expr, i} if is_mask_expr(&expr) => {
            match env.masks.get(&expr) {
                Some(sub_expr) => {
                    env.vars.insert(dst.clone(), sub_expr.clone());
                    (env, Stmt::Pass {i})
                },
                None => {
                    let sub_expr = Expr::Var {
                        id: dst.clone(),
                        ty: expr.get_type().clone(),
                        i: i.clone()
                    };
                    env.masks.insert(expr.clone(), sub_expr);
                    (env, Stmt::Definition {dst, expr, i})
                }
            }
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = lo.sub_vars(&env.vars);
            let hi = hi.sub_vars(&env.vars);
            let (_, body) = body.smap_accum_l(
                env.clone(),
                remove_repeated_mask_definitions
            );
            (env, Stmt::For {var, lo, hi, step, body, i})
        },
        Stmt::While {cond, body, i} => {
            let cond = cond.sub_vars(&env.vars);
            let (_, body) = body.smap_accum_l(
                env.clone(),
                remove_repeated_mask_definitions
            );
            (env, Stmt::While {cond, body, i})
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = cond.sub_vars(&env.vars);
            let (_, thn) = thn.smap_accum_l(
                env.clone(),
                remove_repeated_mask_definitions
            );
            let (_, els) = els.smap_accum_l(
                env.clone(),
                remove_repeated_mask_definitions
            );
            (env, Stmt::If {cond, thn, els, i})
        },
        _ => {
            let s = s.smap(|e: Expr| e.sub_vars(&env.vars));
            s.smap_accum_l(env, remove_repeated_mask_definitions)
        }
    }
}

fn store_expr_in_temporary(mut acc: Vec<Stmt>, e: Expr) -> (Vec<Stmt>, Expr) {
    match e {
        Expr::Var {..} => (acc, e),
        _ => {
            let id = Name::sym_str("t");
            let i = e.get_info();
            let ty = e.get_type().clone();
            acc.push(Stmt::Definition {
                dst: id.clone(),
                expr: e,
                i: i.clone()
            });
            (acc, Expr::Var {id, ty, i})
        }
    }
}

fn store_memory_ops_in_temporary_variable(
    acc: Vec<Stmt>,
    s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::Store {ptr, value, mask, i} => {
            let (mut acc, value) = store_expr_in_temporary(acc, value);
            acc.push(Stmt::Store {ptr, value, mask, i});
            acc
        },
        _ => s.sflatten(acc, store_memory_ops_in_temporary_variable)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Location {
    ptr: Expr,
    mask: Option<Expr>,
}

impl Location {
    fn from(ptr: &Expr, mask: &Option<Expr>) -> Self {
        Location {
            ptr: ptr.clone(),
            mask: mask.clone()
        }
    }
}

fn get_ptr_target<'a>(ptr: &'a Expr) -> Option<&'a Name> {
    if let Expr::BinOp {lhs, op: BinOp::Add, ..} = ptr {
        if let Expr::Var {id, ..} = &**lhs {
            Some(id)
        } else {
            None
        }
    } else {
        None
    }
}

fn sub_load_expr(
    env: &BTreeMap<Name, (Location, Expr)>, e: Expr) -> Expr {
    match e {
        Expr::Load {ptr, mask, ty, i} => {
            if let Some(id) = get_ptr_target(&ptr) {
                let l = Location {
                    ptr: *ptr.clone(),
                    mask: mask.clone().map(|e| *e)
                };
                match env.get(&id) {
                    Some((prev_loc, sub_expr)) if l == *prev_loc => sub_expr.clone(),
                    _ => Expr::Load {ptr, mask, ty, i}
                }
            } else {
                Expr::Load {ptr, mask, ty, i}
            }
        },
        _ => e.smap(|e| sub_load_expr(&env, e))
    }
}

fn filter_sub_load_outer_env(
    env: BTreeMap<Name, (Location, Expr)>,
    inner_env: &BTreeMap<Name, (Location, Expr)>
) -> BTreeMap<Name, (Location, Expr)> {
    // When the inner environment contains an entry for a particular identifier, this means it
    // performs store operations targeting this identifier. We conservatively exclude it from the
    // outer environment in this case, to avoid substituting loads with old data.
    env.into_iter()
        .filter(|(id, _)| !inner_env.contains_key(&id))
        .collect::<BTreeMap<Name, (Location, Expr)>>()
}

fn sub_load_with_temporary_variable(
    mut env: BTreeMap<Name, (Location, Expr)>,
    s: Stmt
) -> (BTreeMap<Name, (Location, Expr)>, Stmt) {
    match s {
        Stmt::Store {ptr, value, mask, i} => {
            if let Some(id) = get_ptr_target(&ptr) {
                let l = Location { ptr: ptr.clone(), mask: mask.clone() };
                env.insert(id.clone(), (l, value.clone()));
            };
            (env, Stmt::Store {ptr, value, mask, i})
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = sub_load_expr(&env, lo);
            let hi = sub_load_expr(&env, hi);
            let (body_env, body) = body.smap_accum_l(
                BTreeMap::new(),
                sub_load_with_temporary_variable
            );
            let env = filter_sub_load_outer_env(env, &body_env);
            (env, Stmt::For {var, lo, hi, step, body, i})
        },
        Stmt::While {cond, body, i} => {
            let cond = sub_load_expr(&env, cond);
            let (body_env, body) = body.smap_accum_l(
                BTreeMap::new(),
                sub_load_with_temporary_variable
            );
            let env = filter_sub_load_outer_env(env, &body_env);
            (env, Stmt::While {cond, body, i})
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = sub_load_expr(&env, cond);
            let (thn_env, thn) = thn.smap_accum_l(
                BTreeMap::new(),
                sub_load_with_temporary_variable
            );
            let (els_env, els) = els.smap_accum_l(
                BTreeMap::new(),
                sub_load_with_temporary_variable
            );
            let env = filter_sub_load_outer_env(env, &thn_env);
            let env = filter_sub_load_outer_env(env, &els_env);
            (env, Stmt::If {cond, thn, els, i})
        },
        _ => {
            let s = s.smap(|e| sub_load_expr(&env, e));
            s.smap_accum_l(env, sub_load_with_temporary_variable)
        }
    }
}

fn filter_load_targets(
    mut env: BTreeMap<Name, Location>,
    e: &Expr
) -> BTreeMap<Name, Location> {
    match e {
        Expr::Load {ptr, ..} => {
            if let Some(id) = get_ptr_target(&ptr) {
                env.remove(id);
            };
            env
        },
        _ => e.sfold(env, filter_load_targets)
    }
}

fn filter_outer_env(
    env: BTreeMap<Name, Location>,
    inner_env: &BTreeMap<Name, Location>
) -> BTreeMap<Name, Location> {
    // We filter the environment corresponding to an outer scope by removing all entries that were
    // removed in the inner scope (because we loaded a value) or changed (because we stored a new
    // value).
    env.into_iter()
        .filter(|(id, l)| match inner_env.get(&id) {
            Some(inner_loc) if inner_loc == l => true,
            _ => false
        })
        .collect::<BTreeMap<Name, Location>>()
}

fn remove_redundant_stores_stmt(
    mut env: BTreeMap<Name, Location>,
    s: Stmt
) -> (BTreeMap<Name, Location>, Stmt) {
    match s {
        Stmt::Store {ptr, value, mask, i} => {
            if let Some(id) = get_ptr_target(&ptr) {
                let l = Location::from(&ptr, &mask);
                match env.get(id) {
                    Some(prev_loc) if *prev_loc == l => (env, Stmt::Pass {i}),
                    _ => {
                        env.insert(id.clone(), l);
                        (env, Stmt::Store {ptr, value, mask, i})
                    },
                }
            } else {
                (env, Stmt::Store {ptr, value, mask, i})
            }
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let env = filter_load_targets(env, &lo);
            let env = filter_load_targets(env, &hi);
            let (body_env, body) = remove_redundant_stores(env.clone(), body);
            let env = filter_outer_env(env, &body_env);
            (env, Stmt::For {var, lo, hi, step, body, i})
        },
        Stmt::While {cond, body, i} => {
            let env = filter_load_targets(env, &cond);
            let (body_env, body) = remove_redundant_stores(env.clone(), body);
            let env = filter_outer_env(env, &body_env);
            (env, Stmt::While {cond, body, i})
        },
        Stmt::If {cond, thn, els, i} => {
            let env = filter_load_targets(env, &cond);
            let (thn_env, thn) = remove_redundant_stores(env.clone(), thn);
            let (els_env, els) = remove_redundant_stores(env.clone(), els);
            let env = filter_outer_env(env, &thn_env);
            let env = filter_outer_env(env, &els_env);
            (env, Stmt::If {cond, thn, els, i})
        },
        _ => {
            let env = s.sfold(env, filter_load_targets);
            (env, s)
        }
    }
}

fn remove_redundant_stores(
    env: BTreeMap<Name, Location>,
    body: Vec<Stmt>
) -> (BTreeMap<Name, Location>, Vec<Stmt>) {
    let (env, body) = body.into_iter()
        .rev()
        .fold((env, vec![]), |acc, s| {
            let (env, mut stmts) = acc;
            let (env, s) = remove_redundant_stores_stmt(env, s);
            stmts.push(s);
            (env, stmts)
        });
    (env, body.into_iter().rev().collect::<Vec<Stmt>>())
}

fn apply_kernel_body(body: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    // 1. Substitute variables defined using arange based on previous definitions, to make it
    //    easier to identify references to equivalent static memory locations.
    let (_, body) = body.smap_accum_l(
        ArangeSubEnv::default(),
        remove_duplicate_arange_definitions
    );

    // 2. Remove repeated definitions of the same mask, making subsequent uses refer to the first
    //    definition of a variable using a particular mask.
    let (_, body) = body.smap_accum_l(
        MaskSubEnv::default(),
        remove_repeated_mask_definitions
    );

    // 3. Store the value of each memory store operation in a temporary variable to enable fusing a
    //    store followed by a load by immediately referring to the temporary variable.
    let body = body.sflatten(vec![], store_memory_ops_in_temporary_variable);

    // 4. Perform a fusion of memory operations, where a load operation referring to the same
    //    pointer and offset as a previous store (using the same mask) is replaced by a temporary
    //    variable containing the value stored.
    let (_, body) = body.smap_accum_l(
        BTreeMap::new(),
        sub_load_with_temporary_variable
    );

    // 5. Eliminate subsequent stores to the same memory location, when we perform no other memory
    //    operations to the same target in-between.
    let (_, body) = remove_redundant_stores(BTreeMap::new(), body);

    Ok(body)
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            let body = apply_kernel_body(body)?;
            Ok(Top::FunDef {triton_jit: true, id, params, body, i})
        },
        _ => Ok(t)
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    Ok(Ast {tops: ast.tops.smap_result(apply_top)?})
}
