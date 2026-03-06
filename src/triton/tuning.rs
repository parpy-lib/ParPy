use super::ast::*;
use crate::gpu::par;
use crate::option::CompileOptions;
use crate::utils::ast::ExprType;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use std::collections::BTreeMap;

fn try_extract_integer(e: &Expr) -> Option<i128> {
    match e {
        Expr::Int {v, ..} => Some(*v),
        _ => None
    }
}

fn try_extract_thread_count(
    step: i128,
    body: &Vec<Stmt>
) -> Option<i128> {
    match body.first() {
        Some(Stmt::Definition {expr: Expr::BinOp {op: BinOp::Add, rhs, ..}, ..}) => {
            match &**rhs {
                Expr::Arange {lo, hi, ..} => {
                    let l = try_extract_integer(&*lo);
                    let h = try_extract_integer(&*hi);
                    let block_size = match (l, h) {
                        (Some(l), Some(h)) => Some(h - l),
                        _ => None
                    };
                    match (block_size, step) {
                        (Some(b), _) if b == step => Some(step),
                        _ => None
                    }
                },
                _ => None
            }
        },
        _ => None
    }
}

fn substitute_int_size_expr(
    thread_count: i128,
    sub_expr: &Expr,
    e: Expr
) -> Expr {
    match e {
        Expr::Arange {lo, hi, ty, i} => {
            let l = try_extract_integer(&lo);
            let h = try_extract_integer(&hi);
            if l == Some(0) && h == Some(thread_count) {
                Expr::Arange {lo, hi: Box::new(sub_expr.clone()), ty, i}
            } else {
                Expr::Arange {lo, hi, ty, i}
            }
        },
        Expr::Full {shape, value, elem_sz, ty, i} => {
            let sh = try_extract_integer(&shape);
            if sh == Some(thread_count) {
                Expr::Full {shape: Box::new(sub_expr.clone()), value, elem_sz, ty, i}
            } else {
                Expr::Full {shape, value, elem_sz, ty, i}
            }
        },
        _ => e.smap(|e| substitute_int_size_expr(thread_count, sub_expr, e))
    }
}

fn substitute_int_size(
    thread_count: i128,
    sub_expr: &Expr,
    s: Stmt
) -> Stmt {
    s.smap(|s| substitute_int_size(thread_count, sub_expr, s))
        .smap(|e| substitute_int_size_expr(thread_count, sub_expr, e))
}

fn add_threaded_block_size_variables(
    mut acc: Option<Name>,
    s: Stmt,
    thread_count: i128
) -> (Option<Name>, Stmt) {
    let rec_call = |acc, s| add_threaded_block_size_variables(acc, s, thread_count);
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            match try_extract_integer(&step) {
                Some(step_val) if step_val == thread_count => {
                    if let Some(thread_count) = try_extract_thread_count(step_val, &body) {
                        let block_size_id = acc.unwrap_or(Name::sym_str("BLOCK_SIZE"));
                        let sub_expr = Expr::Var {
                            id: block_size_id.clone(),
                            ty: lo.get_type().clone(),
                            i: i.clone()
                        };
                        let step = sub_expr.clone();
                        let body = body.smap(|s| substitute_int_size(thread_count, &sub_expr, s));
                        acc = Some(block_size_id.clone());
                        (acc, Stmt::For {var, lo, hi, step, body, i})
                    } else {
                        let (acc, body) = body.smap_accum_l(acc, rec_call);
                        (acc, Stmt::For {var, lo, hi, step, body, i})
                    }
                },
                _ => {
                    let (acc, body) = body.smap_accum_l(acc, rec_call);
                    (acc, Stmt::For {var, lo, hi, step, body, i})
                }
            }
        },
        _ => s.smap_accum_l(acc, rec_call)
    }
}

fn get_pointer_param_ids(params: &Vec<Param>) -> Vec<Name> {
    params.into_iter()
        .filter(|Param {ty, ..}| match ty {
            Type::Pointer {..} => true,
            _ => false
        })
        .map(|Param {id, ..}| id.clone())
        .collect::<Vec<Name>>()
}

fn generate_autotune_decorator(
    block_size_id: Name,
    warp_count: i128,
    params: &Vec<Param>
) -> Decorator {
    let multiples = vec![1, 2, 4, 8];
    let thread_count = warp_count * par::WARP_SIZE as i128;
    let configs = multiples.into_iter()
        .map(|k| {
            let mut mapping = BTreeMap::new();
            mapping.insert(block_size_id.clone(), Expr::Int {
                v: k * thread_count,
                ty: Type::Void,
                i: Info::default()
            });
            AutotuneConfig {mapping, warp_count}
        })
        .collect::<Vec<AutotuneConfig>>();
    let restore_value = get_pointer_param_ids(&params);
    Decorator::Autotune {configs, key: vec![], restore_value}
}

fn add_autotune_decorator_top(
    t: Top,
    warp_counts: &BTreeMap<Name, i128>,
    opts: &CompileOptions
) -> Top {
    match t {
        Top::KernelFunDef {mut decorators, id, mut params, body, i} => {
            match warp_counts.get(&id) {
                Some(wc) => {
                    if opts.triton_autotune {
                        let thread_count = *wc * par::WARP_SIZE as i128;
                        let (block_size_id, body) = body.smap_accum_l(None, |acc, e| {
                            add_threaded_block_size_variables(acc, e, thread_count)
                        });
                        if let Some(bsize_id) = block_size_id {
                            decorators.push(
                                generate_autotune_decorator(bsize_id.clone(), *wc, &params)
                            );
                            params.push(Param {
                                id: bsize_id,
                                ty: Type::Void,
                                annot_ty: AnnotType::Constexpr,
                                i: i.clone()
                            });
                            Top::KernelFunDef {decorators, id, params, body, i}
                        } else {
                            Top::KernelFunDef {decorators, id, params, body, i}
                        }
                    } else {
                        Top::KernelFunDef {decorators, id, params, body, i}
                    }
                },
                None => Top::KernelFunDef {decorators, id, params, body, i}
            }
        },
        _ => t
    }
}

fn collect_warp_counts_stmt(
    mut acc: BTreeMap<Name, i128>,
    s: &Stmt
) -> BTreeMap<Name, i128> {
    match s {
        Stmt::KernelLaunch {id, nwarps, ..} => {
            acc.insert(id.clone(), *nwarps as i128);
            acc
        },
        _ => s.sfold(acc, collect_warp_counts_stmt)
    }
}

fn collect_warp_counts(acc: BTreeMap<Name, i128>, t: &Top) -> BTreeMap<Name, i128> {
    match t {
        Top::FunDef {body, ..} => body.sfold(acc, collect_warp_counts_stmt),
        _ => acc
    }
}

fn contains_autotune_decorator(decorators: &Vec<Decorator>) -> bool {
    decorators.into_iter().any(|d| match d {
        Decorator::Autotune {..} => true,
    })
}

fn get_default_autotune_decorator(
    warp_count: i128,
    params: &Vec<Param>
) -> Decorator {
    let configs = vec![
        AutotuneConfig {mapping: BTreeMap::new(), warp_count}
    ];
    let restore_value = get_pointer_param_ids(&params);
    Decorator::Autotune {configs, key: vec![], restore_value}
}

fn add_default_autotune_decorator_if_none(
    t: Top,
    warp_counts: &BTreeMap<Name, i128>
) -> Top {
    match t {
        Top::KernelFunDef {mut decorators, id, params, body, i} => {
            match warp_counts.get(&id) {
                Some(c) if !contains_autotune_decorator(&decorators) => {
                    decorators.push(get_default_autotune_decorator(*c, &params));
                },
                _ => ()
            };
            Top::KernelFunDef {decorators, id, params, body, i}
        },
        _ => t
    }
}

pub fn apply(ast: Ast, opts: &CompileOptions) -> Ast {
    let warp_counts = ast.tops.sfold(BTreeMap::new(), collect_warp_counts);
    let tops = ast.tops.smap(|t| add_autotune_decorator_top(t, &warp_counts, &opts));
    Ast {tops: tops.smap(|t| add_default_autotune_decorator_if_none(t, &warp_counts))}
}
