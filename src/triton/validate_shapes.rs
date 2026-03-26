use super::ast::*;
use crate::parpy_compile_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::*;
use crate::utils::smap::{SFold, SMapAccum};

fn has_scalar_shape(e: &Expr) -> bool {
    e.get_type().get_shape() == Some(&Shape::Num(1))
}

fn with_blocked_shape(ty: Type, shape: &Shape) -> Type {
    match ty {
        Type::Pointer {ty, ..} => Type::Pointer {ty, shape: shape.clone()},
        Type::Tensor {sz, ..} => Type::Tensor {sz, shape: shape.clone()},
        _ => ty
    }
}

fn try_promote_ptr(ptr: Expr, ofs: &Expr, shape: &Shape) -> Expr {
    if has_scalar_shape(&ptr) {
        let ty = ptr.get_type().clone();
        let i = ptr.get_info();
        Expr::BinOp {
            lhs: Box::new(ptr),
            op: BinOp::Add,
            rhs: Box::new(ofs.clone()),
            ty: with_blocked_shape(ty, shape),
            i
        }
    } else {
        ptr
    }
}

fn promote_expr(ofs: &Expr, shape: &Shape, e: Expr) -> Expr {
    match e {
        Expr::Load {ptr, mask, ty, i} if mask.is_some() => {
            let ptr = Box::new(try_promote_ptr(*ptr, &ofs, &shape));
            Expr::Load {ptr, mask, ty, i}
        },
        _ => e.smap(|e| promote_expr(&ofs, &shape, e))
    }
}

fn promote_stmt(ofs: &Expr, shape: &Shape, s: Stmt) -> Stmt {
    match s {
        Stmt::Store {ptr, value, mask, i} if mask.is_some() => {
            let ptr = try_promote_ptr(ptr, &ofs, &shape);
            Stmt::Store {ptr, value, mask, i}
        },
        _ => {
            s.smap(|s| promote_stmt(&ofs, &shape, s))
                .smap(|e| promote_expr(&ofs, &shape, e))
        }
    }
}

fn static_sub(l: &Expr, r: &Expr) -> Option<i128> {
    match (l, r) {
        (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => Some(*lv - *rv),
        _ => None
    }
}

fn extract_blocked_variable(s: &Stmt) -> Option<(Name, Shape)> {
    match s {
        Stmt::Definition {dst, expr, ..} => {
            let shape = expr.get_type().get_shape()?;
            Some((dst.clone(), shape.clone()))
        }
        _ => None
    }
}

fn try_extract_single_iteration_loop_info(
    lo: &Expr,
    hi: &Expr,
    body: &Vec<Stmt>
) -> Option<(Name, Shape, Expr)> {
    // Ensure the loop runs for exactly one iteration.
    if static_sub(hi, lo)? == 1 {
        if body.is_empty() {
            None
        } else {
            // Extract the name of the inner variable introducing the blocking.
            let (var, shape) = extract_blocked_variable(&body[0])?;
            Some((var, shape, lo.clone()))
        }
    } else {
        None
    }
}

fn promote_single_iteration_loop_indices_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            let body = match try_extract_single_iteration_loop_info(&lo, &hi, &body) {
                Some((id, shape, lo)) => {
                    let ty = lo.get_type().clone();
                    let i = lo.get_info();
                    let ofs = Expr::BinOp {
                        lhs: Box::new(Expr::Var {id, ty: ty.clone(), i: i.clone()}),
                        op: BinOp::Sub,
                        rhs: Box::new(lo),
                        ty,
                        i
                    };
                    body.smap(|s| promote_stmt(&ofs, &shape, s))
                },
                None => body.smap(promote_single_iteration_loop_indices_stmt)
            };
            Stmt::For {var, lo, hi, step, body, i}
        },
        _ => s.smap(promote_single_iteration_loop_indices_stmt)
    }
}

fn validate_shapes_stmt(acc: (), s: &Stmt) -> CompileResult<()> {
    match s {
        Stmt::Store {ptr, value, mask: _, i} => {
            // When the pointer has a blocked shape, the value is automatically broadcast to the
            // shape of the pointer in Triton. However, when the pointer is scalar and the value
            // is blocked, the program is rejected by Triton. We detect this and report an error
            // with a reference to the original code.
            if has_scalar_shape(&ptr) {
                if has_scalar_shape(&value) {
                    Ok(())
                } else {
                    parpy_compile_error!(
                        i,
                        concat!(
                            "This assignment updates a memory location multiple times in ",
                            "parallel. Try marking the outer for-loop as a parallel reduction."
                        )
                    )
                }
            } else {
                // If the pointer is a blocked value, the value is automatically broadcast. When
                // the pointer is a blocked value, it will be assigned a mask based on the outer
                // loop introducing the blockedness.
                Ok(())
            }
        },
        _ => s.sfold_result(Ok(acc), validate_shapes_stmt)
    }
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::Import {..} | Top::FunDef {..} => Ok(t),
        Top::KernelFunDef {decorators, id, params, body, i} => {
            let body = body.smap(promote_single_iteration_loop_indices_stmt);
            body.sfold_result(Ok(()), validate_shapes_stmt)?;
            Ok(Top::KernelFunDef {decorators, id, params, body, i})
        },
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    Ok(Ast {tops: ast.tops.smap_result(apply_top)?})
}
