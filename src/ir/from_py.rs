use crate::parir_runtime_error;
use crate::err::*;
use crate::info::Info;
use crate::par::ParSpec;
use crate::py::ast as py_ast;

use super::ast::*;
use super::par;

use std::collections::HashMap;

fn from_stmt(s: py_ast::Stmt) -> Stmt {
    match s {
        py_ast::Stmt::Assign {dst, e, i} => {
            Stmt::Assignment { dst, e, i }
        },
        py_ast::Stmt::For {var, lo, hi, body, i} => {
            Stmt::For {
                id: var, lo, hi,
                body: from_stmts(body),
                properties: LoopProperties::default(),
                i
            }
        }
    }
}

fn from_stmts(stmts: Vec<py_ast::Stmt>) -> Vec<Stmt> {
    stmts.into_iter()
        .map(|stmt| from_stmt(stmt))
        .collect::<Vec<Stmt>>()
}

pub fn from_ast(ast: py_ast::Ast) -> CompileResult<Ast> {
    let (_, ast) = ast.into_iter()
        .fold(Ok((HashMap::new(), vec![])), |acc, def| {
            let (mut env, mut ast) = acc?;
            match def {
                py_ast::Def::FunDef {id, params, body, ..} => {
                    let body = from_stmts(body);
                    env.insert(id, (params, body));
                    Ok((env, ast))
                },
                py_ast::Def::ParFunInst {id, par, i, ..} => {
                    if let Some((params, body)) = env.get(&id) {
                        let (body, nblocks, nthreads) = par::parallelize_loops(body.clone(), par)?;
                        ast.push(Top::KernelDef {
                            id,
                            params: params.clone(),
                            body, nblocks, nthreads,
                            i: i.clone()
                        });
                        Ok((env, ast))
                    } else {
                        parir_runtime_error!(i, "Parallel instantiation of {id} refers to unknown function")
                    }
                }
            }
        })?;
    Ok(ast)
}
