use super::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
struct UseEnv {
    def: BTreeSet<Name>,
    used: BTreeSet<Name>
}

fn collect_used_err_variables(mut acc: UseEnv, e: &Expr) -> UseEnv {
    match e {
        Expr::Var {ty: Type::Error, id, ..} => {
            acc.used.insert(id.clone());
            acc
        },
        _ => e.sfold(acc, collect_used_err_variables)
    }
}

fn collect_unused_errors_stmt(mut acc: UseEnv, s: &Stmt) -> UseEnv {
    match s {
        Stmt::Definition {ty: Type::Error, id, expr} => {
            acc.def.insert(id.clone());
            if let Some(e) = expr {
                collect_used_err_variables(acc, e)
            } else {
                acc
            }
        },
        _ => {
            let acc = s.sfold(acc, collect_used_err_variables);
            s.sfold(acc, collect_unused_errors_stmt)
        }
    }
}

fn collect_unused_errors(body: &Vec<Stmt>) -> BTreeSet<Name> {
    let acc = UseEnv {def: BTreeSet::new(), used: BTreeSet::new()};
    let mut acc = body.sfold(acc, collect_unused_errors_stmt);
    acc.def.retain(|e| !acc.used.contains(e));
    acc.def
}

fn check_unused_errors(unused: &BTreeSet<Name>, s: Stmt) -> Stmt {
    match s {
        Stmt::Definition {id, expr: Some(e), ..} if unused.contains(&id) =>
            Stmt::CheckError {e},
        _ => s.smap(|s| check_unused_errors(unused, s))
    }
}

fn check_last_error_after_kernel_launch(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::KernelLaunch {..} => {
            acc.push(s);
            acc.push(Stmt::CheckError {e: Expr::GetLastError {
                ty: Type::Error, i: Info::default()
            }});
            acc
        },
        _ => s.sflatten(acc, check_last_error_after_kernel_launch)
    }
}

fn add_error_handling_body(body: Vec<Stmt>) -> Vec<Stmt> {
    let unused_errors = collect_unused_errors(&body);
    let body = body.smap(|s| check_unused_errors(&unused_errors, s));
    body.sflatten(vec![], check_last_error_after_kernel_launch)
}

fn add_error_handling_top(t: Top) -> Top {
    match t {
        Top::FunDef {dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body} => {
            let body = add_error_handling_body(body);
            Top::FunDef {
                dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body
            }
        },
        _ => t
    }
}

pub fn add_error_handling(ast: Ast) -> Ast {
    ast.into_iter()
        .map(add_error_handling_top)
        .collect::<Ast>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cuda::ast_builder::*;

    fn mk_env(def: BTreeSet<Name>, used: BTreeSet<Name>) -> UseEnv {
        UseEnv {def, used}
    }

    fn mk_set<T: Ord>(v: Vec<T>) -> BTreeSet<T> {
        v.into_iter().collect::<BTreeSet<T>>()
    }

    fn mk_default_env() -> UseEnv {
        mk_env(mk_set(vec![]), mk_set(vec![]))
    }

    #[test]
    fn collect_used_non_error_variable() {
        let env = mk_default_env();
        let v = var("x", scalar(ElemSize::F32));
        assert_eq!(collect_used_err_variables(env.clone(), &v), env);
    }

    #[test]
    fn collect_used_error_variable() {
        let env = mk_default_env();
        let v = var("x", Type::Error);
        let expected = mk_env(mk_set(vec![]), mk_set(vec![id("x")]));
        assert_eq!(collect_used_err_variables(env, &v), expected);
    }

    fn gen_stmts() -> Vec<Stmt> {
        vec![
            defn(Type::Error, id("x"), None),
            defn(Type::Error, id("y"), Some(var("x", Type::Error))),
            Stmt::KernelLaunch {
                id: id("kernel"), blocks: Dim3::default(), threads: Dim3::default(),
                stream: Stream::Default, args: vec![]
            }
        ]
    }

    #[test]
    fn collect_used_errors_stmts() {
        let stmts = gen_stmts();
        let res = stmts.sfold(mk_default_env(), collect_unused_errors_stmt);
        let expected = mk_env(mk_set(vec![id("x"), id("y")]), mk_set(vec![id("x")]));
        assert_eq!(res, expected);
    }

    #[test]
    fn unused_errors_stmts() {
        let stmts = gen_stmts();
        assert_eq!(collect_unused_errors(&stmts), mk_set(vec![id("y")]));
    }

    #[test]
    fn check_unused_errors_stmts() {
        let unused = mk_set(vec![id("y")]);
        let stmts = gen_stmts().smap(|s| check_unused_errors(&unused, s));
        let expected = vec![
            defn(Type::Error, id("x"), None),
            Stmt::CheckError {e: var("x", Type::Error)},
            Stmt::KernelLaunch {
                id: id("kernel"), blocks: Dim3::default(), threads: Dim3::default(),
                stream: Stream::Default, args: vec![]
            }
        ];
        assert_eq!(stmts, expected);
    }

    #[test]
    fn add_error_handling_top() {
        let ast = vec![Top::FunDef {
            dev_attr: Attribute::Entry, ret_ty: Type::Void, attrs: vec![],
            id: id("f"), params: vec![], body: gen_stmts()
        }];
        let expected_stmts = vec![
            defn(Type::Error, id("x"), None),
            Stmt::CheckError {e: var("x", Type::Error)},
            Stmt::KernelLaunch {
                id: id("kernel"), blocks: Dim3::default(), threads: Dim3::default(),
                stream: Stream::Default, args: vec![]
            },
            Stmt::CheckError {e: Expr::GetLastError {ty: Type::Error, i: i()}},
        ];
        let expected = vec![Top::FunDef {
            dev_attr: Attribute::Entry, ret_ty: Type::Void, attrs: vec![],
            id: id("f"), params: vec![], body: expected_stmts
        }];
        assert_eq!(add_error_handling(ast), expected);
    }
}
