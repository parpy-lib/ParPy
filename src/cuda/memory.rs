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
        Top::FunDef {body, target: Target::Host, ..} => {
            body.sfold_result(Ok(acc), validate_gpu_memory_access_stmt)
        },
        _ => Ok(())
    }
}

pub fn validate_gpu_memory_access(ast: &Ast) -> CompileResult<()> {
    ast.sfold_result(Ok(()), validate_gpu_memory_access_top)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::assert_error_matches;
    use crate::gpu::ast_builder::*;

    fn assign_stmt() -> Stmt {
        assign(
            array_access(
                var("x", pointer(scalar(ElemSize::F32), MemSpace::Device)),
                int(0, None),
                scalar(ElemSize::F32)
            ),
            binop(
                float(2.718, Some(ElemSize::F32)),
                BinOp::Add,
                float(3.14, Some(ElemSize::F32)),
                scalar(ElemSize::F32)
            )
        )
    }

    #[test]
    fn gpu_access_from_kernel_ok() {
        let ast = vec![Top::KernelFunDef {
            attrs: vec![], id: id("kernel"), params: vec![],
            body: vec![assign_stmt()]
        }];
        assert!(validate_gpu_memory_access(&ast).is_ok());
    }

    #[test]
    fn gpu_access_from_host_fails() {
        let ast = vec![Top::FunDef {
            ret_ty: Type::Void, id: id("f"), params: vec![],
            body: vec![assign_stmt()], target: Target::Host
        }];
        assert_error_matches(validate_gpu_memory_access(&ast), "not allowed outside parallel");
    }

    #[test]
    fn gpu_access_from_device_fun_ok() {
        let ast = vec![Top::FunDef {
            ret_ty: Type::Void, id: id("f"), params: vec![],
            body: vec![assign_stmt()], target: Target::Device
        }];
        assert!(validate_gpu_memory_access(&ast).is_ok());
    }
}
