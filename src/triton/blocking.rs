use super::ast::*;
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

fn is_blocked_type(ty: &Type) -> bool {
    get_shape_from_type(ty) > 1
}

fn wrap_blocked_literals_in_full_expr(e: Expr) -> CompileResult<Expr> {
    match e {
        Expr::Bool {ref ty, ..} |
        Expr::Int {ref ty, ..} |
        Expr::Float {ref ty, ..} if is_blocked_type(&ty) => {
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

fn contains_arange(acc: bool, e: &Expr) -> bool {
    match e {
        Expr::Arange {..} => true,
        _ => e.sfold(acc, contains_arange)
    }
}

fn try_extract_blocked_var(s: &Stmt) -> Option<Name> {
    match s {
        Stmt::Assign {dst, expr, ..} if contains_arange(false, &expr) => {
            Some(dst.clone())
        },
        _ => None
    }
}

fn make_mask(mask: Option<Box<Expr>>, masks: &Vec<Expr>) -> Expr {
    mask.into_iter()
        .map(|e| *e)
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
        .unwrap()
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

fn mask_memory_accesses_expr(cond: &Expr, e: Expr) -> CompileResult<Expr> {
    match e {
        Expr::Reduce {op, arg, ty, i} if is_blocked_type(arg.get_type()) => {
            let arg = Box::new(mask_memory_accesses_expr(&cond, *arg)?);
            let ne = get_neutral_element(&op, &ty, &i)?;
            let arg = Expr::Where {
                cond: Box::new(cond.clone()),
                thn: arg,
                els: Box::new(ne),
                ty: ty.clone(),
                i: i.clone()
            };
            Ok(Expr::Reduce {op, arg: Box::new(arg), ty, i})
        },
        Expr::Load {ptr, mask, ty, i} if is_blocked_type(ptr.get_type()) => {
            let ptr = Box::new(mask_memory_accesses_expr(&cond, *ptr)?);
            let mask = Some(Box::new(make_mask(mask, &vec![cond.clone()])));
            Ok(Expr::Load {ptr, mask, ty, i})
        },
        Expr::Store {ptr, value, mask, ty, i} if is_blocked_type(ptr.get_type()) => {
            let ptr = Box::new(mask_memory_accesses_expr(&cond, *ptr)?);
            let value = Box::new(mask_memory_accesses_expr(&cond, *value)?);
            let mask = Some(Box::new(make_mask(mask, &vec![cond.clone()])));
            Ok(Expr::Store {ptr, value, mask, ty, i})
        },
        _ => e.smap_result(|e| mask_memory_accesses_expr(&cond, e))
    }
}

fn mask_memory_accesses_stmt(cond: &Expr, s: Stmt) -> CompileResult<Stmt> {
    s.smap_result(|s| mask_memory_accesses_stmt(&cond, s))?
        .smap_result(|e| mask_memory_accesses_expr(&cond, e))
}

fn mask_memory_accesses_in_parallel_for(s: Stmt) -> CompileResult<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            let body = if !body.is_empty() {
                match try_extract_blocked_var(&body[0]) {
                    Some(id) => {
                        let ty = hi.get_type().clone();
                        let shape = Shape::Num(get_shape_from_type(&ty));
                        let boundary_cond = Expr::BinOp {
                            lhs: Box::new(Expr::Var {id, ty: ty.clone(), i: i.clone()}),
                            op: BinOp::Lt,
                            rhs: Box::new(hi.clone()),
                            ty: Type::Tensor {sz: ElemSize::Bool, shape},
                            i: i.clone()
                        };
                        body.smap_result(|s| mask_memory_accesses_stmt(&boundary_cond, s))
                    },
                    None => Ok(body)
                }
            } else {
                Ok(body)
            }?;
            let body = body.smap_result(mask_memory_accesses_in_parallel_for)?;
            Ok(Stmt::For {var, lo, hi, step, body, i})
        },
        _ => s.smap_result(mask_memory_accesses_in_parallel_for)
    }
}

fn add_masking_expr(masks: &Vec<Expr>, e: Expr) -> CompileResult<Expr> {
    if masks.is_empty() {
        Ok(e)
    } else {
        match e {
            Expr::Reduce {op, arg, ty, i} => {
                let mask = make_mask(None, masks);
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
            Expr::Load {ptr, mask, ty, i} => {
                let mask = Some(Box::new(make_mask(mask, masks)));
                Ok(Expr::Load {ptr, mask, ty, i})
            },
            Expr::Store {ptr, value, mask, ty, i} => {
                let mask = Some(Box::new(make_mask(mask, masks)));
                Ok(Expr::Store {ptr, value, mask, ty, i})
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
        Stmt::Assign {dst, expr, i} if !masks.is_empty() => {
            let expr = add_masking_expr(&masks, expr)?;
            // TODO: this rewrite is only safe when 'dst' has been defined earlier in the
            // program. Therefore, we need to track this information somehow.
            let ty = expr.get_type().clone();
            let where_expr = Expr::Where {
                cond: Box::new(make_mask(None, &masks)),
                thn: Box::new(expr),
                els: Box::new(Expr::Var {
                    id: dst.clone(), ty: ty.clone(), i: i.clone()
                }),
                ty: ty,
                i: i.clone()
            };
            stmts.push(Stmt::Assign {dst, expr: where_expr, i});
            Ok((masks, stmts))
        },
        Stmt::For {var, lo, hi, step, mut body, i} if is_blocked_type(lo.get_type()) => {
            let ty = lo.get_type().clone();
            let shape = Shape::Num(get_shape_from_type(&ty));
            let var_expr = Expr::Var {id: var.clone(), ty: ty.clone(), i: i.clone()};
            stmts.push(Stmt::Assign {
                dst: var.clone(),
                expr: lo,
                i: i.clone()
            });
            body.push(Stmt::Assign {
                dst: var,
                expr: Expr::BinOp {
                    lhs: Box::new(var_expr.clone()),
                    op: BinOp::Add,
                    rhs: Box::new(Expr::Int {
                        v: step as i128, ty: ty.clone(), i: i.clone()
                    }),
                    ty: ty.clone(),
                    i: i.clone()
                },
                i: i.clone()
            });
            let cond = Expr::BinOp {
                lhs: Box::new(var_expr),
                op: BinOp::Lt,
                rhs: Box::new(hi),
                ty: Type::Tensor {sz: ElemSize::Bool, shape},
                i: i.clone()
            };
            let while_loop = Stmt::While {cond, body, i: i.clone()};
            add_masking_stmt((masks, stmts), while_loop)
        },
        Stmt::While {cond, body, i} if is_blocked_type(cond.get_type()) => {
            let ty = cond.get_type().clone();
            let cond_id = Name::sym_str("while_cond");
            let cond_var = Expr::Var {
                id: cond_id.clone(),
                ty: ty.clone(),
                i: i.clone()
            };
            let cond_upd = Stmt::Assign {
                dst: cond_id.clone(),
                expr: cond,
                i: i.clone()
            };
            stmts.push(cond_upd.clone());
            masks.push(cond_var.clone());
            let acc = Ok((masks, vec![]));
            let (mut masks, mut body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            masks.pop();
            body.push(cond_upd.clone());
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
        Stmt::If {cond, thn, els, i} if is_blocked_type(cond.get_type()) => {
            let ty = cond.get_type().clone();
            let cond_id = Name::sym_str("if_cond");
            stmts.push(Stmt::Assign {
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
        Stmt::For {var, lo, hi, step, body, i} => {
            let acc = Ok((masks, vec![]));
            let (masks, body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::For {var, lo, hi, step, body, i});
            Ok((masks, stmts))
        },
        Stmt::While {cond, body, i} => {
            let acc = Ok((masks, vec![]));
            let (masks, body) = body.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::While {cond, body, i});
            Ok((masks, stmts))
        },
        Stmt::If {cond, thn, els, i} => {
            let acc = Ok((masks, vec![]));
            let (masks, thn) = thn.sfold_owned_result(acc, add_masking_stmt)?;
            let acc = Ok((masks, vec![]));
            let (masks, els) = els.sfold_owned_result(acc, add_masking_stmt)?;
            stmts.push(Stmt::If {cond, thn, els, i});
            Ok((masks, stmts))
        },
        _ => {
            let s = s.smap_result(|e| add_masking_expr(&masks, e))?;
            stmts.push(s);
            Ok((masks, stmts))
        }
    }
}

fn transform_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            // If a literal has been assigned a non-unit shape, we rewrite it to use tl.full to
            // specify an explicit shape.
            let body = body.smap_result(wrap_blocked_literals_in_full_stmt)?;

            // Adds masking of memory operations (tl.load and tl.store) when the pointers are
            // blocked, by checking that the loop variable on which the index is (presumably) based
            // is within range of the for-loop that introduced it.
            let body = body.smap_result(mask_memory_accesses_in_parallel_for)?;

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
            
            Ok(Top::FunDef {triton_jit: true, id, params, body, i})
        },
        Top::Import {..} | Top::FunDef {..} => Ok(t),
    }
}

pub fn transform(ast: Ast) -> CompileResult<Ast> {
    Ok(Ast {tops: ast.tops.smap_result(transform_top)?})
}
