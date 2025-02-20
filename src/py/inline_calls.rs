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
        Expr::Tuple {..} | Expr::Dict {..} | Expr::Builtin {..} |
        Expr::Convert {..} => {
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
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Vec<Stmt>> {
    match stmt {
        Stmt::Call {func, args, i} => {
            if let Some(ast_ref) = ir_asts.get(&func) {
                let fun: FunDef = unsafe { ast_ref.reference::<FunDef>() }.clone();
                let sub_map = construct_sub_map(fun.params, args, &fun.id, &fun.i)?;
                let mut body = fun.body.smap(|s| substitute_variables_stmt(s, &sub_map));
                acc.append(&mut body);
            } else {
                let msg = format!(
                    "Reference to unknown function {func}.\n{0}",
                    "Perhaps you forgot to decorate the function with @parir.jit?"
                );
                py_runtime_error!(i, "{}", msg)?
            }
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let body = inline_function_calls_stmts(body, ir_asts)?;
            acc.push(Stmt::For {var, lo, hi, step, body, i});
        },
        Stmt::While {cond, body, i} => {
            let body = inline_function_calls_stmts(body, ir_asts)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = inline_function_calls_stmts(thn, ir_asts)?;
            let els = inline_function_calls_stmts(els, ir_asts)?;
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::WithGpuContext {body, i} => {
            let body = inline_function_calls_stmts(body, ir_asts)?;
            acc.push(Stmt::WithGpuContext {body, i});
        },
        Stmt::Label {label, assoc, i} => {
            match assoc {
                Some(s) => {
                    let mut inner_acc = inline_function_calls_stmt(vec![], *s, ir_asts)?;
                    if inner_acc.len() == 1 {
                        let assoc = Some(Box::new(inner_acc.remove(0)));
                        acc.push(Stmt::Label {label, assoc, i})
                    } else {
                        py_runtime_error!(i, "Internal error: found label referring to invalid statement")?
                    }
                },
                None => acc.push(Stmt::Label {label, assoc: None, i})
            }
        },
        Stmt::Definition {..} | Stmt::Assign {..} => {
            acc.push(stmt);
        }
    };
    Ok(acc)
}

fn inline_function_calls_stmts<'py>(
    stmts: Vec<Stmt>,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), |acc, stmt| inline_function_calls_stmt(acc?, stmt, ir_asts))
}

pub fn inline_function_calls<'py>(
    fun: FunDef,
    ir_asts: &BTreeMap<String, Bound<'py, PyCapsule>>
) -> PyResult<FunDef> {
    let body = inline_function_calls_stmts(fun.body, ir_asts)?;
    Ok(FunDef {body, ..fun})
}
