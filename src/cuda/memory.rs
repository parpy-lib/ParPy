use crate::prickle_compile_error;
use crate::gpu::ast::*;
use crate::utils::err::*;
use crate::utils::smap::*;

fn validate_gpu_memory_access_expr(acc: (), e: &Expr) -> CompileResult<()> {
    match e {
        Expr::ArrayAccess {i, ..} => {
            prickle_compile_error!(i, "Assignments are not allowed outside parallel code.")
        },
        _ => e.sfold_result(Ok(acc), validate_gpu_memory_access_expr)
    }
}

fn validate_gpu_memory_access_stmt(acc: (), s: &Stmt) -> CompileResult<()> {
    let _ = s.sfold_result(Ok(acc), validate_gpu_memory_access_stmt)?;
    s.sfold_result(Ok(acc), validate_gpu_memory_access_expr)
}

fn validate_gpu_memory_access_top(acc: (), t: &Top) -> CompileResult<()> {
    match t {
        Top::FunDef {body, ..} => {
            body.sfold_result(Ok(acc), validate_gpu_memory_access_stmt)
        },
        _ => Ok(())
    }
}

pub fn validate_gpu_memory_access(ast: &Ast) -> CompileResult<()> {
    ast.sfold_result(Ok(()), validate_gpu_memory_access_top)
}
