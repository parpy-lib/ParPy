use super::ast::*;
use super::utils::*;
use crate::parpy_internal_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::reduce::ExprLit;
use crate::utils::smap::*;

fn get_elem_size(ty: &Type, i: &Info) -> CompileResult<ElemSize> {
    match ty.get_elem_size() {
        Some(sz) => Ok(sz.clone()),
        None => parpy_internal_error!(i, "Failed to extract element size of literal type")
    }
}

fn get_shape_from_type(ty: &Type) -> usize {
    match ty.get_shape() {
        Some(Shape::Num(n)) => *n,
        _ => 1
    }
}

fn replace_full_with_conversion_expr(e: Expr) -> Expr {
    match e {
        Expr::Full {value, ty, i, ..} if value.get_type().is_blocked() => {
            Expr::Convert {value, ty, i}
        },
        _ => e.smap(replace_full_with_conversion_expr)
    }
}

fn replace_full_with_conversion_stmt(s: Stmt) -> Stmt {
    s.smap(replace_full_with_conversion_stmt)
        .smap(replace_full_with_conversion_expr)
}

fn wrap_blocked_literals_in_full_expr(e: Expr) -> CompileResult<Expr> {
    match e {
        Expr::Bool {ref ty, ..} |
        Expr::Int {ref ty, ..} |
        Expr::Float {ref ty, ..} if ty.is_blocked() => {
            let i = e.get_info();
            let ty = ty.clone();
            let shape = get_shape_from_type(&ty);
            let elem_sz = get_elem_size(&ty, &i)?;
            Ok(Expr::Full {
                shape,
                value: Box::new(e),
                elem_sz,
                ty,
                i
            })
        },
        _ => e.smap_result(wrap_blocked_literals_in_full_expr)
    }
}

fn wrap_blocked_literals_in_full_stmt(s: Stmt) -> CompileResult<Stmt> {
    s.smap_result(wrap_blocked_literals_in_full_stmt)?
        .smap_result(wrap_blocked_literals_in_full_expr)
}

fn make_mask(mask: Option<Expr>, masks: &Vec<Expr>) -> Option<Expr> {
    mask.into_iter()
        .chain(masks.clone().into_iter())
        .reduce(|l, r| {
            let ty = l.get_type().clone();
            let i = l.get_info();
            Expr::BinOp {
                lhs: Box::new(l),
                op: BinOp::And,
                rhs: Box::new(r),
                ty, i
            }
        })
}

fn get_neutral_element(op: &ReduceOp, ty: &Type, i: &Info) -> CompileResult<Expr> {
    let i = i.clone();
    let sz = match ty.get_elem_size() {
        Some(sz) => Ok(sz),
        None => parpy_internal_error!(i, "Encountered invalid type of reduction in Triton codegen")
    }?;
    match op {
        ReduceOp::Min => Ok(Expr::generate_literal(f64::INFINITY, sz, i)),
        ReduceOp::Max => Ok(Expr::generate_literal(f64::NEG_INFINITY, sz, i)),
        ReduceOp::Sum => Ok(Expr::generate_literal(0.0, sz, i)),
        ReduceOp::Prod => Ok(Expr::generate_literal(1.0, sz, i)),
        ReduceOp::Any => parpy_internal_error!(i, "Invalid reduction operation in Triton codegen")
    }
}

fn mask_for_loop_expr(
    loop_cond_var: &Expr,
    e: Expr
) -> CompileResult<Expr> {
    match e {
        Expr::Reduce {op, arg, ty, i} => {
            let arg = Box::new(mask_for_loop_expr(&loop_cond_var, *arg)?);
            let ne = get_neutral_element(&op, &ty, &i)?;
            let arg = Box::new(Expr::Where {
                cond: Box::new(loop_cond_var.clone()),
                thn: arg,
                els: Box::new(ne),
                ty: ty.clone(),
                i: i.clone()
            });
            Ok(Expr::Reduce {op, arg, ty, i})
        },
        Expr::Load {ptr, mask, ty, i} if ty.is_blocked() => {
            let ptr = Box::new(mask_for_loop_expr(&loop_cond_var, *ptr)?);
            let mask = make_mask(mask.map(|e| *e), &vec![loop_cond_var.clone()])
                .map(|e| Box::new(e));
            Ok(Expr::Load {ptr, mask, ty, i})
        },
        _ => e.smap_result(|e| mask_for_loop_expr(&loop_cond_var, e))
    }
}

fn mask_for_loop_stmt(
    mut acc: Vec<Stmt>,
    s: Stmt,
    cond_def: &Stmt,
    loop_cond_var: &Expr,
) -> CompileResult<Vec<Stmt>> {
    let s = s.smap_result(|e| mask_for_loop_expr(&loop_cond_var, e))?;
    match s {
        Stmt::Definition {dst, expr, i} if contains_arange(&expr) => {
            // Insert the definition of the for-loop condition immediately after defining the
            // blocked for-loop variable.
            acc.push(Stmt::Definition {dst, expr, i});
            acc.push(cond_def.clone());
            Ok(acc)
        },
        Stmt::Store {ptr, value, mask, i} => {
            let mask = if ptr.get_type().is_blocked() || value.get_type().is_blocked() {
                make_mask(mask, &vec![loop_cond_var.clone()])
            } else {
                mask
            };
            acc.push(Stmt::Store {ptr, value, mask, i});
            Ok(acc)
        },
        _ => s.sflatten_result(acc, |acc, s| {
            mask_for_loop_stmt(acc, s, &cond_def, &loop_cond_var)
        })
    }
}

fn add_masking_in_parallel_for(
    mut acc: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            let body = if !body.is_empty() {
                match try_extract_blocked_var(&body[0]) {
                    Some(id) => {
                        let ty = hi.get_type().clone();
                        let shape = Shape::Num(get_shape_from_type(&ty));
                        let cond_ty = Type::Tensor {sz: ElemSize::Bool, shape};
                        let op = if step > 0 { BinOp::Lt } else { BinOp::Gt };
                        let boundary_cond = Expr::BinOp {
                            lhs: Box::new(Expr::Var {
                                id: id.clone(), ty: ty.clone(), i: i.clone()
                            }),
                            op,
                            rhs: Box::new(hi.clone()),
                            ty: cond_ty.clone(),
                            i: i.clone()
                        };
                        let for_cond_id = Name::sym_str("for_cond");
                        let for_cond = Expr::Var {
                            id: for_cond_id.clone(),
                            ty: cond_ty.clone(),
                            i: i.clone()
                        };
                        let cond_def = Stmt::Definition {
                            dst: for_cond_id,
                            expr: boundary_cond,
                            i: i.clone()
                        };
                        // We use the mask in any loads and stores that take place in the body of
                        // the for-loop.
                        body.sflatten_result(vec![], |acc, s| {
                            mask_for_loop_stmt(acc, s, &cond_def, &for_cond)
                        })
                    },
                    None => {
                        body.sflatten_result(vec![], |acc, s| {
                            add_masking_in_parallel_for(acc, s)
                        })
                    },
                }
            } else {
                body.sflatten_result(vec![], |acc, s| {
                    add_masking_in_parallel_for(acc, s)
                })
            }?;
            acc.push(Stmt::For {var, lo, hi, step, body, i});
            Ok(acc)
        },
        _ => s.sflatten_result(acc, add_masking_in_parallel_for)
    }
}

fn add_masking_expr(masks: &Vec<Expr>, e: Expr) -> CompileResult<Expr> {
    if masks.is_empty() {
        Ok(e)
    } else {
        match e {
            Expr::Reduce {op, arg, ty, i} if ty.is_blocked() => {
                let mask = make_mask(None, masks).unwrap();
                let ne = get_neutral_element(&op, &ty, &i)?;
                Ok(Expr::Reduce {
                    op,
                    arg: Box::new(Expr::Where {
                        cond: Box::new(mask),
                        thn: arg,
                        els: Box::new(ne),
                        ty: ty.clone(),
                        i: i.clone()
                    }),
                    ty,
                    i
                })
            },
            Expr::Load {ptr, mask, ty, i} if ty.is_blocked() => {
                let mask = make_mask(mask.map(|e| *e), masks)
                    .map(|e| Box::new(e));
                Ok(Expr::Load {ptr, mask, ty, i})
            },
            _ => e.smap_result(|e| add_masking_expr(&masks, e))
        }
    }
}

fn add_masking_stmt(
    acc: (Vec<Expr>, Vec<Stmt>),
    s: Stmt
) -> CompileResult<(Vec<Expr>, Vec<Stmt>)> {
    let (mut masks, mut stmts) = acc;
    match s {
        Stmt::Definition {dst, expr, i} if !masks.is_empty() => {
            let expr = add_masking_expr(&masks, expr)?;
            stmts.push(Stmt::Definition {dst, expr, i});
            Ok((masks, stmts))
        },
        Stmt::Assign {dst, expr, i} if !masks.is_empty() => {
            let expr = add_masking_expr(&masks, expr)?;
            // NOTE(larshum, 2025-02-20): This transformation relies on the fact that we keep a
            // distinction between definitions and assignments, as it is only valid to apply
            // masking to a previously defined variable that we are re-assigning.
            let ty = expr.get_type().clone();
            let mask = make_mask(None, &masks).unwrap();
            let expr = Expr::Where {
                cond: Box::new(mask),
                thn: Box::new(expr),
                els: Box::new(Expr::Var {
                    id: dst.clone(), ty: ty.clone(), i: i.clone()
                }),
                ty: ty,
                i: i.clone()
            };
            stmts.push(Stmt::Assign {dst, expr, i});
            Ok((masks, stmts))
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = add_masking_expr(&masks, lo)?;
            let hi = add_masking_expr(&masks, hi)?;
            let acc = Ok((masks, vec![]));
            let (masks, body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::For {var, lo, hi, step, body, i});
            Ok((masks, stmts))
        },
        Stmt::While {cond, body, i} if cond.get_type().is_blocked() => {
            let cond = add_masking_expr(&masks, cond)?;
            let ty = cond.get_type().clone();
            let cond_id = Name::sym_str("while_cond");
            let cond_var = Expr::Var {
                id: cond_id.clone(),
                ty: ty.clone(),
                i: i.clone()
            };
            stmts.push(Stmt::Definition {
                dst: cond_id.clone(),
                expr: cond.clone(),
                i: i.clone()
            });
            masks.push(cond_var.clone());
            let acc = Ok((masks, vec![]));
            let (mut masks, mut body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            masks.pop();
            body.push(Stmt::Assign {
                dst: cond_id,
                expr: cond,
                i: i.clone()
            });
            stmts.push(Stmt::While {
                cond: Expr::Reduce {
                    op: ReduceOp::Any,
                    arg: Box::new(cond_var),
                    ty: ty,
                    i: i.clone()
                },
                body,
                i
            });
            Ok((masks, stmts))
        },
        Stmt::If {cond, thn, els, i} if cond.get_type().is_blocked() => {
            let cond = add_masking_expr(&masks, cond)?;
            let ty = cond.get_type().clone();
            let cond_id = Name::sym_str("if_cond");
            stmts.push(Stmt::Definition {
                dst: cond_id.clone(),
                expr: cond,
                i: i.clone()
            });
            let thn_cond = Expr::Var {
                id: cond_id.clone(),
                ty: ty.clone(),
                i: i.clone()
            };
            masks.push(thn_cond.clone());
            let acc = Ok((masks, stmts));
            let (mut masks, stmts) = thn.sfold_owned_result(acc, add_masking_stmt)?;
            masks.pop();
            let els_cond = Expr::UnOp {
                op: UnOp::Not,
                arg: Box::new(thn_cond),
                ty: ty.clone(),
                i: i.clone()
            };
            masks.push(els_cond);
            let acc = Ok((masks, stmts));
            let (mut masks, stmts) = els.sfold_owned_result(acc, add_masking_stmt)?;
            masks.pop();
            Ok((masks, stmts))
        },
        Stmt::While {cond, body, i} => {
            let cond = add_masking_expr(&masks, cond)?;
            let acc = Ok((masks, vec![]));
            let (masks, body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::While {cond, body, i});
            Ok((masks, stmts))
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = add_masking_expr(&masks, cond)?;
            let acc = Ok((masks, vec![]));
            let (masks, thn) = thn.sfold_owned_result(acc, add_masking_stmt)?;
            let acc = Ok((masks, vec![]));
            let (masks, els) = els.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::If {cond, thn, els, i});
            Ok((masks, stmts))
        },
        Stmt::Store {ptr, value, mask, i} => {
            let ptr = add_masking_expr(&masks, ptr)?;
            let value = add_masking_expr(&masks, value)?;
            let mask = make_mask(mask, &masks);
            stmts.push(Stmt::Store {ptr, value, mask, i});
            Ok((masks, stmts))
        },
        _ => {
            let s = s.smap_result(|e| add_masking_expr(&masks, e))?;
            stmts.push(s);
            Ok((masks, stmts))
        }
    }
}

fn use_bitwise_ops_in_mask(mask: Expr) -> Expr {
    match mask {
        Expr::BinOp {lhs, op: BinOp::And, rhs, ty, i} => {
            let lhs = Box::new(use_bitwise_ops_in_mask(*lhs));
            let rhs = Box::new(use_bitwise_ops_in_mask(*rhs));
            Expr::BinOp {lhs, op: BinOp::BitAnd, rhs, ty, i}
        },
        Expr::BinOp {lhs, op: BinOp::Or, rhs, ty, i} => {
            let lhs = Box::new(use_bitwise_ops_in_mask(*lhs));
            let rhs = Box::new(use_bitwise_ops_in_mask(*rhs));
            Expr::BinOp {lhs, op: BinOp::BitOr, rhs, ty, i}
        },
        _ => mask.smap(use_bitwise_ops_in_mask)
    }
}

fn use_bitwise_ops_in_masking_expr(e: Expr) -> Expr {
    match e {
        Expr::Load {ptr, mask, ty, i} => {
            let ptr = Box::new(use_bitwise_ops_in_masking_expr(*ptr));
            let mask = mask.map(|e| Box::new(use_bitwise_ops_in_mask(*e)));
            Expr::Load {ptr, mask, ty, i}
        },
        Expr::Where {cond, thn, els, ty, i} => {
            let cond = Box::new(use_bitwise_ops_in_mask(*cond));
            let thn = Box::new(use_bitwise_ops_in_masking_expr(*thn));
            let els = Box::new(use_bitwise_ops_in_masking_expr(*els));
            Expr::Where {cond, thn, els, ty, i}
        },
        _ => e.smap(use_bitwise_ops_in_masking_expr)
    }
}

fn use_bitwise_ops_in_masking(s: Stmt) -> Stmt {
    match s {
        Stmt::Store {ptr, value, mask, i} => {
            let ptr = use_bitwise_ops_in_masking_expr(ptr);
            let value = use_bitwise_ops_in_masking_expr(value);
            let mask = mask.map(use_bitwise_ops_in_mask);
            Stmt::Store {ptr, value, mask, i}
        },
        _ => {
            s.smap(use_bitwise_ops_in_masking_expr)
                .smap(use_bitwise_ops_in_masking)
        }
    }
}

fn transform_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            // Replace uses of tl.full with an explicit conversion when they contain blocked data.
            // This is required for correctness because the argument of a tl.full cannot itself be
            // a blocked value.
            let body = body.smap(replace_full_with_conversion_stmt);

            // If a literal has been assigned a non-unit shape, we rewrite it to use tl.full to
            // specify an explicit shape.
            let body = body.smap_result(wrap_blocked_literals_in_full_stmt)?;

            // Adds masking of blocked operations within a parallel for-loop. This is applied to
            // memory operations (Expr::Load and Stmt::Store), whose results are visible across all
            // threads of the GPU. It also applies to reductions (Expr::Reduce), because threads
            // that are not participating should use a value that does not impact the final result.
            let body = body.sflatten_result(vec![], add_masking_in_parallel_for)?;

            // Rewrite if-statements and while-loops containing block-wide conditions to perform
            // masked operations based on the individual values of a condition. Specifically, we
            // rewrite if-statements so that the code of each branch runs for threads where the
            // condition evaluates accordingly. A while-loop is rewritten to execute as long as any
            // thread evaluates its condition to true.
            //
            // In addition, loads and stores use masks to avoid out-of-bounds access, and
            // similarly, reductions use the neutral element of its operation to avoid having an
            // impact on the result.
            let (_, body) = body.sfold_owned_result(Ok((vec![], vec![])), add_masking_stmt)?;

            // Replaces the use of the boolean 'and' and 'or' operations in Python with the bitwise
            // operators '&' and '|' because recent versions of Triton do not accept the former
            // when used in masks. This applies to the masks of loads and stores and the condition
            // of a tl.where.
            let body = body.smap(use_bitwise_ops_in_masking);
            
            Ok(Top::FunDef {triton_jit: true, id, params, body, i})
        },
        Top::Import {..} | Top::FunDef {..} => Ok(t),
    }
}

pub fn transform(ast: Ast) -> CompileResult<Ast> {
    Ok(Ast {tops: ast.tops.smap_result(transform_top)?})
}
