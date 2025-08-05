use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;
use pyo3::types::*;

use std::collections::BTreeMap;

fn construct_sub_map(
    fun_params: Vec<Param>,
    args: Vec<Expr>,
    fun_id: &Name,
    i: &Info
) -> PyResult<BTreeMap<Name, Expr>> {
    if fun_params.len() == args.len() {
        Ok(fun_params.into_iter()
            .zip(args.into_iter())
            .fold(BTreeMap::new(), |mut acc, (Param {id, ..}, e)| {
                acc.insert(id, e);
                acc
            }))
    } else {
        let msg = format!(
            "Function {0} expected {1} arguments, but received {2}.",
            fun_id, fun_params.len(), args.len()
        );
        py_runtime_error!(i, "{}", msg)
    }
}

fn substitute_variables_expr(e: Expr, sub_map: &BTreeMap<Name, Expr>) -> Expr {
    match e {
        Expr::Var {id, i, ..} if sub_map.contains_key(&id) => {
            let e = sub_map.get(&id).unwrap().clone();
            e.with_info(i)
        },
        Expr::Var {..} | Expr::String {..} | Expr::Bool {..} | Expr::Int {..} |
        Expr::Float {..} | Expr::UnOp {..} | Expr::BinOp {..} |
        Expr::IfExpr {..} | Expr::Subscript {..} | Expr::Slice {..} |
        Expr::Tuple {..} | Expr::Call {..} | Expr::NeutralElement {..} |
        Expr::Builtin {..} | Expr::Convert {..} => {
            e.smap(|e| substitute_variables_expr(e, sub_map))
        }
    }
}

fn substitute_variables_stmt(s: Stmt, sub_map: &BTreeMap<Name, Expr>) -> Stmt {
    s.smap(|s| substitute_variables_stmt(s, sub_map))
        .smap(|e| substitute_variables_expr(e, sub_map))
}

fn inline_function_calls_stmt<'py>(
    mut acc: Vec<Stmt>,
    stmt: Stmt,
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Vec<Stmt>> {
    match stmt {
        Stmt::Call {func, args, i} => {
            if let Some(ast_ref) = tops.get(&func) {
                let t: Top = unsafe { ast_ref.reference::<Top>() }.clone();
                match t {
                    Top::FunDef {v: fun} => {
                        let sub_map = construct_sub_map(fun.params, args, &fun.id, &fun.i)?;
                        let mut body = fun.body.smap(|s| substitute_variables_stmt(s, &sub_map));
                        acc.append(&mut body);
                    },
                    Top::ExtDecl {id, ..} => {
                        py_runtime_error!(i, "Cannot inline call to external function {id}")?
                    },
                }
            } else {
                py_runtime_error!(i, "Reference to unknown function {func}.")?
            }
        },
        Stmt::For {var, lo, hi, step, body, labels, i} => {
            let body = inline_function_calls_stmts(body, tops)?;
            acc.push(Stmt::For {var, lo, hi, step, body, labels, i});
        },
        Stmt::While {cond, body, i} => {
            let body = inline_function_calls_stmts(body, tops)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = inline_function_calls_stmts(thn, tops)?;
            let els = inline_function_calls_stmts(els, tops)?;
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::WithGpuContext {body, i} => {
            let body = inline_function_calls_stmts(body, tops)?;
            acc.push(Stmt::WithGpuContext {body, i});
        },
        Stmt::Scope {body, i} => {
            let body = inline_function_calls_stmts(body, tops)?;
            acc.push(Stmt::Scope {body, i})
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
        Stmt::Label {..} => {
            acc.push(stmt);
        }
    };
    Ok(acc)
}

fn inline_function_calls_stmts<'py>(
    stmts: Vec<Stmt>,
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), |acc, stmt| inline_function_calls_stmt(acc?, stmt, tops))
}

pub fn inline_function_calls<'py>(
    fun: FunDef,
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<FunDef> {
    let body = inline_function_calls_stmts(fun.body, tops)?;
    Ok(FunDef {body, ..fun})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;

    #[test]
    fn sub_vars_empty_map() {
        let s = assignment(var("a", Type::Unknown), var("b", Type::Unknown));
        assert_eq!(substitute_variables_stmt(s.clone(), &BTreeMap::new()), s);
    }

    #[test]
    fn sub_vars_lhs() {
        let s = assignment(var("a", Type::Unknown), var("b", Type::Unknown));
        let mut sub_map = BTreeMap::new();
        sub_map.insert(id("a"), var("c", Type::Unknown));
        let expected = assignment(var("c", Type::Unknown), var("b", Type::Unknown));
        assert_eq!(substitute_variables_stmt(s.clone(), &sub_map), expected);
    }

    #[test]
    fn sub_vars_rhs() {
        let s = assignment(var("a", Type::Unknown), var("b", Type::Unknown));
        let mut sub_map = BTreeMap::new();
        sub_map.insert(id("b"), int(1, None));
        let expected = assignment(var("a", Type::Unknown), int(1, None));
        assert_eq!(substitute_variables_stmt(s, &sub_map), expected);
    }
}
