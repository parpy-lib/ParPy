use super::ast::*;
use super::utils::*;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::smap::{SFlatten, SMapAccum};

const TRITON_MAX_NUMEL: usize = 1048576;

fn try_extract_int_literal(e: &Expr) -> Option<i128> {
    match e {
        Expr::Int {v, ..} => Some(*v),
        _ => None
    }
}

fn extract_iteration_count(lo: &Expr, hi: &Expr) -> Option<(i128, i128)> {
    let l = try_extract_int_literal(&lo)?;
    let h = try_extract_int_literal(&hi)?;
    if i128::abs(h-l) as usize > TRITON_MAX_NUMEL {
        None
    } else {
        Some((l, h))
    }
}

fn update_block_size_shape(shape: Shape, block_size: i128) -> Shape {
    match shape {
        Shape::Num(n) if n > 1 => Shape::Num(block_size),
        _ => shape
    }
}

fn update_block_size_type(ty: Type, block_size: i128) -> Type {
    match ty {
        Type::Pointer {ty, shape} => {
            let ty = Box::new(update_block_size_type(*ty, block_size));
            let shape = update_block_size_shape(shape, block_size);
            Type::Pointer {ty, shape}
        },
        Type::Tensor {sz, shape} => {
            let shape = update_block_size_shape(shape, block_size);
            Type::Tensor {sz, shape}
        },
        Type::Function {..} |
        Type::List |
        Type::String |
        Type::Void => ty
    }
}

fn update_block_size_expr(e: Expr, block_size: i128) -> Expr {
    let ty = update_block_size_type(e.get_type().clone(), block_size);
    let e = e.with_type(ty);
    match e {
        Expr::Full {shape: _, value, elem_sz, ty, i} => {
            let shape = Box::new(Expr::Int {
                v: block_size,
                ty: ty.clone(),
                i: i.clone()
            });
            let value = Box::new(update_block_size_expr(*value, block_size));
            Expr::Full {shape, value, elem_sz, ty, i}
        },
        _ => e.smap(|e| update_block_size_expr(e, block_size))
    }
}

fn update_block_size(s: Stmt, block_size: i128) -> Stmt {
    match s {
        _ => {
            s.smap(|s| update_block_size(s, block_size))
                .smap(|e| update_block_size_expr(e, block_size))
        }
    }
}

fn apply_stmt(mut acc: Vec<Stmt>, s: Stmt) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::For {var, lo, hi, step, mut body, i} => {
            // When the number of iterations is known, we remove the for-loop and its
            // first statement. The loop variable is instead defined as one block
            // containing all values referred to by the original loop. Afterward, we
            // simply add the remainder of the loop body as is.
            if !body.is_empty() {
                if let Some((id, step_size)) = try_extract_blocked_initialization(&body[0]) {
                    if let Some((l, h)) = extract_iteration_count(&lo, &hi) {
                        // Remove the first statement of the for-loop body, which involves the
                        // (original) initialization of the iteration variable.
                        body.remove(0);

                        // We set the block size to the smallest power of two greater than the
                        // total number of iterations. Further, we require that it is at least 32
                        // for sanity purposes (the size of a warp).
                        let n_iters = if l < h { h - l } else { l - h };
                        let block_size = (n_iters as usize).next_power_of_two();
                        let block_size = usize::max(block_size, 32) as i128;
                        let ty = lo.get_type().clone();
                        acc.push(Stmt::Definition {
                            dst: id.clone(),
                            expr: Expr::BinOp {
                                lhs: Box::new(lo),
                                op: BinOp::Add,
                                rhs: Box::new(Expr::BinOp {
                                    lhs: Box::new(Expr::Arange {
                                        lo: Box::new(Expr::Int {
                                            v: 0,
                                            ty: ty.clone(),
                                            i: i.clone()
                                        }),
                                        hi: Box::new(Expr::Int {
                                            v: block_size,
                                            ty: ty.clone(),
                                            i: i.clone()
                                        }),
                                        ty: ty.clone(),
                                        i: i.clone()
                                    }),
                                    op: BinOp::Mul,
                                    rhs: Box::new(Expr::Int {
                                        v: step_size, ty: ty.clone(), i: i.clone()
                                    }),
                                    ty: ty.clone(),
                                    i: i.clone()
                                }),
                                ty,
                                i: i.clone()
                            },
                            i: i.clone()
                        });
                        // Push the for-loop body after updating the block size within and
                        // recursively applying this function to the updated body.
                        let body = body.smap(|s| update_block_size(s, block_size));
                        let mut body = body.sflatten_result(vec![], apply_stmt)?;
                        acc.append(&mut body);
                    } else {
                        let body = body.sflatten_result(vec![], apply_stmt)?;
                        acc.push(Stmt::For {var, lo, hi, step, body, i});
                    }
                } else {
                    let body = body.sflatten_result(vec![], apply_stmt)?;
                    acc.push(Stmt::For {var, lo, hi, step, body, i});
                }
            };
            Ok(acc)
        },
        _ => s.sflatten_result(acc, apply_stmt)
    }
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::KernelFunDef {decorators, id, params, body, i} => {
            let body = body.sflatten_result(vec![], apply_stmt)?;
            Ok(Top::KernelFunDef {decorators, id, params, body, i})
        },
        _ => Ok(t)
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    Ok(Ast {tops: ast.tops.smap_result(apply_top)?})
}
