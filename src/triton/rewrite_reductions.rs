use crate::parpy_compile_error;
use crate::gpu::ast::*;
use crate::gpu::reduce as gpu_reduce;
use crate::utils::err::*;
use crate::utils::name::Name;
use crate::utils::reduce;
use crate::utils::smap::*;

fn apply_stmt(mut acc: Vec<Stmt>, s: Stmt) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, unroll, i} => {
            let (l, op, r, sz, i) = gpu_reduce::extract_reduction_operands(body, &i)?;
            let ne = match reduce::neutral_element(&op, &sz, &i) {
                Some(ne) => Ok(ne),
                None => {
                    parpy_compile_error!(i, "Binary operation of reduction has \
                                             unknown neutral element")
                }
            }?;
            let ty = Type::Scalar {sz};
            let temp_id = Name::sym_str("temp");
            let temp_var = Expr::Var {
                id: temp_id.clone(),
                ty: ty.clone(),
                i: i.clone()
            };
            acc.push(Stmt::Definition {
                ty: ty.clone(),
                id: temp_id.clone(),
                expr: ne,
                i: i.clone()
            });
            let new_body = vec![Stmt::Expr {
                e: Expr::Assign {
                    lhs: Box::new(temp_var.clone()),
                    rhs: Box::new(Expr::BinOp {
                        lhs: Box::new(temp_var.clone()),
                        op: op.clone(),
                        rhs: Box::new(r),
                        ty: ty.clone(),
                        i: i.clone()
                    }),
                    ty: ty.clone(),
                    i: i.clone()
                },
                i: i.clone()
            }];
            acc.push(Stmt::ParallelReduction {
                var_ty, var, init, cond, incr, body: new_body, nthreads,
                tpb, unroll, i: i.clone()
            });
            acc.push(Stmt::Expr {
                e: Expr::Assign {
                    lhs: Box::new(l.clone()),
                    rhs: Box::new(Expr::BinOp {
                        lhs: Box::new(l),
                        op,
                        rhs: Box::new(temp_var),
                        ty: ty.clone(),
                        i: i.clone()
                    }),
                    ty: ty.clone(),
                    i: i.clone()
                },
                i: i.clone()
            });
            Ok(acc)
        },
        _ => s.sflatten_result(acc, apply_stmt)
    }
}

fn apply_stmts(stmts: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.sflatten_result(vec![], apply_stmt)
}

fn apply_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::ExtDecl {..} | Top::FunDef {..} => Ok(t),
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = apply_stmts(body)?;
            Ok(Top::KernelFunDef {attrs, id, params, body, i})
        },
    }
}

pub fn apply(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(apply_top)
}
