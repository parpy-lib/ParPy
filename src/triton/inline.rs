use super::ast::*;
use crate::parpy_internal_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;
use crate::utils::substitute::{SubEnv, SubVars};

use std::collections::{BTreeMap, BTreeSet};

struct InlineEnv {
    kernel_ids: BTreeSet<Name>,
    fun_tops: BTreeMap<Name, (Vec<Name>, Vec<Stmt>)>,
}

fn collect_kernel_entry_points_stmt(mut acc: BTreeSet<Name>, s: &Stmt) -> BTreeSet<Name> {
    match s {
        Stmt::KernelLaunch {id, ..} => {
            acc.insert(id.clone());
            acc
        },
        _ => s.sfold(acc, collect_kernel_entry_points_stmt)
    }
}

fn collect_kernel_entry_points_top(acc: BTreeSet<Name>, t: &Top) -> BTreeSet<Name> {
    match t {
        Top::FunDef {triton_jit: false, body, ..} => {
            body.sfold(acc, collect_kernel_entry_points_stmt)
        },
        _ => acc
    }
}

fn collect_kernel_entry_points(ast: &Ast) -> BTreeSet<Name> {
    ast.tops.sfold(BTreeSet::new(), collect_kernel_entry_points_top)
}

fn replace_calls_with_free_variables(
    acc: Vec<(Name, Expr)>,
    e: Expr
) -> (Vec<(Name, Expr)>, Expr) {
    match e {
        Expr::Call {id, args, ty, i} => {
            let free_var = Name::sym_str("t");
            let new_e = Expr::Var {
                id: free_var.clone(),
                ty: ty.clone(),
                i: i.clone()
            };
            let (mut acc, args) = args.smap_accum_l(acc, replace_calls_with_free_variables);
            acc.push((free_var, Expr::Call {id, args, ty, i}));
            (acc, new_e)
        },
        _ => e.smap_accum_l(acc, replace_calls_with_free_variables)
    }
}

fn mk_call_assignments(calls: Vec<(Name, Expr)>) -> Vec<Stmt> {
    calls.into_iter()
        .map(|(id, e)| {
            let i = e.get_info();
            Stmt::Assign {
                dst: id,
                expr: e,
                i
            }
        })
        .collect::<Vec<Stmt>>()
}

fn rewrite_calls_anf_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Definition {dst, expr, i} => {
            let (calls, expr) = replace_calls_with_free_variables(vec![], expr);
            acc.append(&mut mk_call_assignments(calls));
            acc.push(Stmt::Definition {dst, expr, i});
        },
        Stmt::Assign {dst, expr, i} => {
            let (calls, expr) = replace_calls_with_free_variables(vec![], expr);
            acc.append(&mut mk_call_assignments(calls));
            acc.push(Stmt::Assign {dst, expr, i});
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let (calls, lo) = replace_calls_with_free_variables(vec![], lo);
            let (calls, hi) = replace_calls_with_free_variables(calls, hi);
            acc.append(&mut mk_call_assignments(calls));
            let body = body.sflatten(vec![], rewrite_calls_anf_stmt);
            acc.push(Stmt::For {var, lo, hi, step, body, i});
        },
        Stmt::While {cond, body, i} => {
            let (calls, cond) = replace_calls_with_free_variables(vec![], cond);
            let mut call_assigns = mk_call_assignments(calls);
            acc.append(&mut call_assigns.clone());
            let mut body = body.sflatten(vec![], rewrite_calls_anf_stmt);
            // For a while-loop, we could have a condition that involves function calls. Therefore,
            // we perform the function calls both before the loop and at the end of each iteration,
            // storing the result in the same new variable.
            body.append(&mut call_assigns);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let (calls, cond) = replace_calls_with_free_variables(vec![], cond);
            acc.append(&mut mk_call_assignments(calls));
            let thn = thn.sflatten(vec![], rewrite_calls_anf_stmt);
            let els = els.sflatten(vec![], rewrite_calls_anf_stmt);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::Return {value, i} => {
            let (calls, value) = replace_calls_with_free_variables(vec![], value);
            acc.append(&mut mk_call_assignments(calls));
            acc.push(Stmt::Return {value, i});
        },
        Stmt::Expr {e, i} => {
            let (calls, e) = replace_calls_with_free_variables(vec![], e);
            acc.append(&mut mk_call_assignments(calls));
            // If the expression was a function call it does not return a value. Therefore, we can
            // disregard the variable representing its result.
            match e {
                Expr::Var {..} => (),
                _ => acc.push(Stmt::Expr {e, i})
            }
        },
        Stmt::Store {ptr, value, mask, i} => {
            let (calls, ptr) = replace_calls_with_free_variables(vec![], ptr);
            let (calls, value) = replace_calls_with_free_variables(calls, value);
            let (calls, mask) = match mask {
                Some(m) => {
                    let (calls, m) = replace_calls_with_free_variables(calls, m);
                    (calls, Some(m))
                },
                None => (calls, None)
            };
            acc.append(&mut mk_call_assignments(calls));
            acc.push(Stmt::Store {ptr, value, mask, i});
        },
        Stmt::Barrier {..} |
        Stmt::KernelLaunch {..} => {
            acc.push(s);
        },
    };
    acc
}

fn rewrite_calls_anf(t: Top) -> Top {
    match t {
        Top::FunDef {triton_jit, id, params, body, i} => {
            let body = body.sflatten(vec![], rewrite_calls_anf_stmt);
            Top::FunDef {triton_jit, id, params, body, i}
        },
        Top::Import {..} => t
    }
}

fn replace_return_with_assign_to(id: &Name, s: Stmt) -> Stmt {
    match s {
        Stmt::Return {value, i} => Stmt::Assign {dst: id.clone(), expr: value, i},
        _ => s.smap(|s| replace_return_with_assign_to(&id, s))
    }
}

fn inline_function(
    env: &InlineEnv,
    id: Name,
    args: Vec<Expr>,
    dst: Option<Name>,
    i: &Info
) -> CompileResult<Vec<Stmt>> {
    match env.fun_tops.get(&id) {
        Some((params, body)) => {
            let sub_map = params.clone()
                .into_iter()
                .zip(args.into_iter())
                .map(|(id, arg)| (id, arg))
                .collect::<SubEnv<Expr>>();
            let body = body.clone()
                .smap(|s| s.sub_vars(&sub_map));
            Ok(if let Some(dst_id) = dst {
                body.smap(|s| replace_return_with_assign_to(&dst_id, s))
            } else {
                body
            })
        },
        None => parpy_internal_error!(i, "Failed to look up function {id} when \
                                          inlining in Triton codegen")
    }
}

fn inline_calls_stmt(
    env: &InlineEnv,
    mut stmts: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::Definition {dst, expr: Expr::Call {id, args, ty: _, i}, i: _} |
        Stmt::Assign {dst, expr: Expr::Call {id, args, ty: _, i}, i: _} => {
            let body = inline_function(env, id, args, Some(dst), &i)?;
            let mut body = body.sflatten_result(vec![], |acc, s| {
                inline_calls_stmt(env, acc, s)
            })?;
            stmts.append(&mut body);
            Ok(stmts)
        },
        Stmt::Expr {e: Expr::Call {id, args, ty: _, i}, i: _} => {
            let body = inline_function(env, id, args, None, &i)?;
            let mut body = body.sflatten_result(vec![], |acc, s| {
                inline_calls_stmt(env, acc, s)
            })?;
            stmts.append(&mut body);
            Ok(stmts)
        },
        _ => s.sflatten_result(stmts, |acc, s| inline_calls_stmt(env, acc, s))
    }
}

fn inline_calls_top(env: &InlineEnv, t: Top) -> CompileResult<Option<Top>> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            // Any GPU function that is not in the set of kernel identifiers is only indirectly
            // called via other functions. These functions are removed from the resulting AST for
            // brevity.
            if env.kernel_ids.contains(&id) {
                let body = body.sflatten_result(vec![], |acc, s| inline_calls_stmt(env, acc, s))?;
                Ok(Some(Top::FunDef {triton_jit: true, id, params, body, i}))
            } else {
                Ok(None)
            }
        },
        _ => Ok(Some(t))
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    let kernel_ids = collect_kernel_entry_points(&ast);
    let tops = ast.tops.smap(rewrite_calls_anf);
    let fun_tops = tops.clone()
        .into_iter()
        .map(|t| match t {
            Top::FunDef {triton_jit: _, id, params, body, i: _} => {
                Some((id.clone(), (params, body)))
            },
            _ => None
        })
        .flatten()
        .collect::<BTreeMap<Name, (Vec<Name>, Vec<Stmt>)>>();
    let env = InlineEnv { kernel_ids, fun_tops };
    let tops = tops.into_iter()
        .map(|t| inline_calls_top(&env, t))
        .collect::<CompileResult<Vec<Option<Top>>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<Top>>();
    Ok(Ast {tops})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::triton::ast_builder::*;
    use crate::utils::normalize_symbols::NormalizeSym;
    use crate::utils::pprint::*;

    fn pprint_body(stmts: Vec<Stmt>) -> String {
        pprint_iter(stmts.iter(), PrettyPrintEnv::default(), "\n").1
    }

    fn assert_eq_bodies(l: Vec<Stmt>, r: Vec<Stmt>) {
        let lstr = pprint_body(l.clone());
        let rstr = pprint_body(r.clone());
        assert_eq!(
            l.normalize_symbols(),
            r.normalize_symbols(),
            "\nLHS:\n{lstr}\nRHS:\n{rstr}");
    }

    #[test]
    fn function_calls_rewrite_anf() {
        let ty = Type::Tensor {sz: ElemSize::F32, shape: Shape::Num(1)};
        let mk_call = |s| Expr::Call {
            id: id(s), args: vec![], ty: ty.clone(), i: i()
        };
        let e = Expr::BinOp {
            lhs: Box::new(mk_call("f")),
            op: BinOp::Add,
            rhs: Box::new(mk_call("g")),
            ty: ty.clone(),
            i: i()
        };
        let s = Stmt::Assign {
            dst: id("x"),
            expr: e.clone(),
            i: i()
        };
        let stmts = rewrite_calls_anf_stmt(vec![], s);
        let expected = vec![
            Stmt::Assign {dst: id("a"), expr: mk_call("f"), i: i()},
            Stmt::Assign {dst: id("b"), expr: mk_call("g"), i: i()},
            Stmt::Assign {
                dst: id("x"),
                expr: Expr::BinOp {
                    lhs: Box::new(var("a", Some(ty.clone()))),
                    op: BinOp::Add,
                    rhs: Box::new(var("b", Some(ty.clone()))),
                    ty: ty.clone(),
                    i: i()
                },
                i: i()
            },
        ];
        assert_eq_bodies(stmts, expected);
    }

    #[test]
    fn nested_function_call_rewrite_anf() {
        let ty = Type::Tensor {sz: ElemSize::F32, shape: Shape::Num(1)};
        let g_call = Expr::Call {
            id: id("g"),
            args: vec![],
            ty: ty.clone(),
            i: i()
        };
        let s = Stmt::Assign {
            dst: id("x"),
            expr: Expr::Call {
                id: id("f"),
                args: vec![g_call.clone()],
                ty: ty.clone(),
                i: i()
            },
            i: i()
        };
        let stmts = rewrite_calls_anf_stmt(vec![], s);
        let expected = vec![
            Stmt::Assign {dst: id("a"), expr: g_call, i: i()},
            Stmt::Assign {
                dst: id("b"),
                expr: Expr::Call {
                    id: id("f"),
                    args: vec![var("a", Some(ty.clone()))],
                    ty: ty.clone(),
                    i: i()
                },
                i: i()
            },
            Stmt::Assign {
                dst: id("x"),
                expr: var("b", Some(ty.clone())),
                i: i()
            }
        ];
        assert_eq_bodies(stmts, expected);
    }

    #[test]
    fn while_loop_cond_rewrite_anf() {
        let ty = Type::Tensor {sz: ElemSize::F32, shape: Shape::Num(1)};
        let f_call = Expr::Call {
            id: id("f"),
            args: vec![],
            ty: ty.clone(),
            i: i()
        };
        let s = Stmt::While {
            cond: f_call.clone(),
            body: vec![
                Stmt::Assign {dst: id("x"), expr: float(1.0), i: i()}
            ],
            i: i()
        };
        let stmts = rewrite_calls_anf_stmt(vec![], s);
        let expected = vec![
            Stmt::Assign {dst: id("a"), expr: f_call.clone(), i: i()},
            Stmt::While {
                cond: var("a", Some(ty.clone())),
                body: vec![
                    Stmt::Assign {dst: id("x"), expr: float(1.0), i: i()},
                    Stmt::Assign {dst: id("a"), expr: f_call, i: i()},
                ],
                i: i()
            }
        ];
        assert_eq_bodies(stmts, expected);
    }
}
