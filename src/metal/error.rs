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

#[cfg(test)]
mod test {
    use super::*;
    use crate::metal::ast_builder::*;
    use crate::test::*;

    fn alloc_dev_expr() -> Expr {
        Expr::AllocDevice {
            id: id("x"), elem_ty: scalar(ElemSize::F32), sz: 10,
            ty: scalar(ElemSize::I32), i: i()
        }
    }

    fn kernel_launch_expr() -> Expr {
        Expr::KernelLaunch {
            id: id("f"), blocks: Dim3::default(), threads: Dim3::default(),
            args: vec![], ty: scalar(ElemSize::I32), i: i()
        }
    }

    #[test]
    fn add_error_handling_device_alloc() {
        let e = alloc_dev_expr();
        let s = definition(scalar(ElemSize::I32), id("err"), e.clone());
        assert_eq!(add_error_handling_stmt(s), Stmt::CheckError {e});
    }

    #[test]
    fn add_error_handling_kernel_launch() {
        let e = kernel_launch_expr();
        let s = definition(scalar(ElemSize::I32), id("err"), e.clone());
        assert_eq!(add_error_handling_stmt(s), Stmt::CheckError {e});
    }

    #[test]
    fn add_error_handling_multi_stmts() {
        let t = Top::FunDef {
            attrs: vec![FunAttribute::ExternC],
            is_kernel: false,
            ret_ty: scalar(ElemSize::I32),
            id: id("entry"),
            params: vec![],
            body: vec![
                definition(scalar(ElemSize::I32), id("e1"), alloc_dev_expr()),
                definition(scalar(ElemSize::I32), id("y"), int(1, ElemSize::I32)),
                definition(scalar(ElemSize::I32), id("e2"), kernel_launch_expr()),
            ]
        };
        let other_top = Top::VarDef {
            ty: scalar(ElemSize::F32), id: id("z"), init: None
        };
        let ast = Ast {tops: vec![other_top.clone(), t]};
        let expected_top = Top::FunDef {
            attrs: vec![FunAttribute::ExternC],
            is_kernel: false,
            ret_ty: scalar(ElemSize::I32),
            id: id("entry"),
            params: vec![],
            body: vec![
                Stmt::CheckError {e: alloc_dev_expr()},
                definition(scalar(ElemSize::I32), id("y"), int(1, ElemSize::I32)),
                Stmt::CheckError {e: kernel_launch_expr()},
            ],
        };
        let expected_ast = Ast {tops: vec![other_top, expected_top]};
        assert_eq!(add_error_handling(ast), expected_ast);
    }
}
