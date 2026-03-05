use super::ast::*;
use crate::gpu::par;
use crate::option::CompileOptions;
use crate::utils::ast::ExprType;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use itertools::Itertools;
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
    mut acc: Vec<(Name, i128)>,
    s: Stmt
) -> (Vec<(Name, i128)>, Stmt) {
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            if let Some(step_val) = try_extract_integer(&step) {
                if let Some(thread_count) = try_extract_thread_count(step_val, &body) {
                    let block_size_id = Name::sym_str("BLOCK_SIZE");
                    let sub_expr = Expr::Var {
                        id: block_size_id.clone(),
                        ty: lo.get_type().clone(),
                        i: i.clone()
                    };
                    let step = sub_expr.clone();
                    let body = body.smap(|s| substitute_int_size(thread_count, &sub_expr, s));
                    acc.push((block_size_id, thread_count));
                    (acc, Stmt::For {var, lo, hi, step, body, i})
                } else {
                    let (acc, body) = body.smap_accum_l(acc, add_threaded_block_size_variables);
                    (acc, Stmt::For {var, lo, hi, step, body, i})
                }
            } else {
                let (acc, body) = body.smap_accum_l(acc, add_threaded_block_size_variables);
                (acc, Stmt::For {var, lo, hi, step, body, i})
            }
        },
        _ => s.smap_accum_l(acc, add_threaded_block_size_variables)
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
    block_sizes: Vec<(Name, i128)>,
    params: &Vec<Param>
) -> Option<Decorator> {
    if block_sizes.is_empty() {
        None
    } else {
        let multiples = vec![1, 2, 4, 8];
        let bsz = block_sizes.iter()
            .fold(Some(0), |acc, (_, sz)| {
                let acc = acc?;
                if acc == 0 || acc == *sz {
                    Some(*sz)
                } else {
                    None
                }
            })?;
        let warp_count = bsz / par::WARP_SIZE as i128;
        let configs = block_sizes.iter()
            .map(|(id, n)| {
                multiples.clone()
                    .into_iter()
                    .map(|k| (id.clone(), k * *n))
            })
            .multi_cartesian_product()
            .map(|items| {
                let mapping = items.into_iter()
                    .map(|(id, v)| (id, Expr::Int {v, ty: Type::Void, i: Info::default()}))
                    .collect::<BTreeMap<Name, Expr>>();
                AutotuneConfig {mapping, warp_count}
            })
            .collect::<Vec<AutotuneConfig>>();
        let restore_value = get_pointer_param_ids(&params);
        Some(Decorator::Autotune {configs, key: vec![], restore_value})
    }
}

fn add_autotune_decorator_top(t: Top, opts: &CompileOptions) -> Top {
    match t {
        Top::KernelFunDef {decorators, id, params, body, i} => {
            if opts.triton_autotune {
                let (block_sizes, body) = body.smap_accum_l(vec![], add_threaded_block_size_variables);
                let autotune_decorator = generate_autotune_decorator(block_sizes.clone(), &params);
                let decorators = decorators.into_iter()
                    .chain(autotune_decorator.into_iter())
                    .collect::<Vec<Decorator>>();
                let params = params.into_iter()
                    .chain(block_sizes.into_iter()
                           .map(|(id, _)| Param {
                               id,
                               ty: Type::Void,
                               annot_ty: AnnotType::Constexpr,
                               i: i.clone()
                           }))
                    .collect::<Vec<Param>>();
                Top::KernelFunDef {decorators, id, params, body, i}
            } else {
                Top::KernelFunDef {decorators, id, params, body, i}
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
    thread_counts: &BTreeMap<Name, i128>
) -> Top {
    match t {
        Top::KernelFunDef {mut decorators, id, params, body, i} => {
            match thread_counts.get(&id) {
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
    let tops = ast.tops.smap(|t| add_autotune_decorator_top(t, &opts));
    let thread_counts = tops.sfold(BTreeMap::new(), collect_warp_counts);
    Ast {tops: tops.smap(|t| add_default_autotune_decorator_if_none(t, &thread_counts))}
}
