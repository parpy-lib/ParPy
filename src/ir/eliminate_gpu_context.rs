use super::ast::*;
use crate::utils::err::*;
use crate::utils::smap::*;

fn apply_stmt(mut acc: Vec<Stmt>, s: Stmt, in_par: bool) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::For {
            var, lo: lo @ Expr::Int {v: 0, ..}, hi: hi @ Expr::Int {v: 1, ..},
            step: step @ 1, body, par, i
        } if var.get_str() == "_gpu_context" && par.nthreads == 1 => {
            let mut body = body.sflatten_result(vec![], |acc, s| apply_stmt(acc, s, true))?;
            // If a GPU-context for-loop is not the outermost for-loop of a parallel loop nest, it
            // can be removed to clean up the resulting code. Otherwise, we have to keep it to
            // ensure the parallelization behaves as expected.
            if in_par {
                acc.append(&mut body);
            } else {
                acc.push(Stmt::For {var, lo, hi, step, body, par, i});
            }
            Ok(acc)
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let in_par = in_par || par.is_parallel();
            let body = body.sflatten_result(vec![], |acc, s| apply_stmt(acc, s, in_par))?;
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
            Ok(acc)
        },
        _ => s.sflatten_result(acc, |acc, s| apply_stmt(acc, s, in_par))
    }
}

fn apply_fun_def(f: FunDef) -> CompileResult<FunDef> {
    let body = f.body.sflatten_result(vec![], |acc, s| apply_stmt(acc, s, false))?;
    Ok(FunDef {body, ..f})
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {v} => Ok(Top::FunDef {v: apply_fun_def(v)?}),
        Top::StructDef {..} | Top::ExtDecl {..} => Ok(t)
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    let main = apply_fun_def(ast.main)?;
    let tops = ast.tops.smap_result(apply_top)?;
    Ok(Ast {main, tops})
}
