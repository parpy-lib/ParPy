use super::ast::*;
use crate::utils::smap::*;

fn add_error_handling_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::Definition {expr: e @ Expr::AllocDevice {..}, ..} => {
            Stmt::CheckError {e}
        },
        Stmt::Definition {expr: e @ Expr::KernelLaunch {..}, ..} => {
            Stmt::CheckError {e}
        },
        _ => s.smap(add_error_handling_stmt)
    }
}

fn add_error_handling_top(t: Top) -> Top {
    match t {
        Top::FunDef {attrs, is_kernel: false, ret_ty, id, params, body} => {
            let body = body.smap(add_error_handling_stmt);
            Top::FunDef {attrs, is_kernel: false, ret_ty, id, params, body}
        },
        _ => t
    }
}

pub fn add_error_handling(mut ast: Ast) -> Ast {
    ast.tops = ast.tops.smap(add_error_handling_top);
    ast
}
