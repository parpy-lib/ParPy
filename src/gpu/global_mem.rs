/// This file contains an analysis finding places where many threads within a block write to the
/// same global memory address. If all threads of a block perform a write to the same global memory
/// location, this may produce an invalid result. For instance, the tests in test/test_syrk.py
/// reproduces this problem in CUDA running on an A100 when using 512 or 1024 threads (when this
/// transformation is not included).
///
/// The analysis is performed in two steps. First, we collect a set of all variables whose value
/// depends on the thread index. This comes either from the variable in the innermost for-loop or
/// any expressions computed based on it. Second, we transform global writes to memory where the
/// index is independent of the thread index, so that only the first thread performs the write.

use super::ast::*;
use crate::parpy_compile_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeSet;

fn thread_index_dependent_expr_helper(
    acc: bool,
    vars: &BTreeSet<Name>,
    expr: &Expr
) -> bool {
    if acc {
        acc
    } else {
        match expr {
            Expr::Var {id, ..} if vars.contains(&id) => true,
            Expr::ThreadIdx {..} => true,
            _ => expr.sfold(acc, |acc, e| thread_index_dependent_expr_helper(acc, vars, e))
        }
    }
}

fn thread_index_dependent_expr(
    vars: &BTreeSet<Name>,
    expr: &Expr
) -> bool {
    thread_index_dependent_expr_helper(false, vars, expr)
}

fn extract_assign_target_id(e: &Expr) -> CompileResult<Name> {
    match e {
        Expr::Var {id, ..} => Ok(id.clone()),
        Expr::ArrayAccess {target, ..} => extract_assign_target_id(&target),
        _ => {
            parpy_compile_error!(e.get_info(), "Unexpected target of assignment")
        }
    }
}

fn find_thread_index_dependent_variables_stmt(
    acc: CompileResult<BTreeSet<Name>>,
    stmt: &Stmt
) -> CompileResult<BTreeSet<Name>> {
    let mut acc = acc?;
    match stmt {
        Stmt::Definition {id, expr, ..} if thread_index_dependent_expr(&acc, expr) => {
            acc.insert(id.clone());
            Ok(acc)
        },
        Stmt::Expr {e, ..} => {
            match e {
                Expr::Assign {lhs, rhs, ..} if thread_index_dependent_expr(&acc, rhs) => {
                    let target_id = extract_assign_target_id(lhs)?;
                    acc.insert(target_id);
                },
                _ => ()
            };
            Ok(acc)
        },
        Stmt::For {var, init, body, ..} => {
            if thread_index_dependent_expr(&acc, init) {
                acc.insert(var.clone());
            }
            body.sfold(Ok(acc), find_thread_index_dependent_variables_stmt)
        },
        Stmt::Definition {..} | Stmt::While {..} | Stmt::If {..} | Stmt::Return {..} |
        Stmt::Scope {..} | Stmt::ParallelReduction {..} | Stmt::Synchronize {..} |
        Stmt::WarpReduce {..} | Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} |
        Stmt::AllocDevice {..} | Stmt::AllocShared {..} | Stmt::FreeDevice {..} |
        Stmt::CopyMemory {..} => {
            stmt.sfold(Ok(acc), find_thread_index_dependent_variables_stmt)
        }
    }
}

fn find_thread_index_dependent_variables_top(
    acc: CompileResult<BTreeSet<Name>>,
    top: &Top
) -> CompileResult<BTreeSet<Name>> {
    match top {
        Top::KernelFunDef {body, ..} => {
            body.sfold(acc, find_thread_index_dependent_variables_stmt)
        },
        Top::ExtDecl {..} | Top::FunDef {..} => acc
    }
}

fn find_thread_index_dependent_variables(ast: &Ast) -> CompileResult<BTreeSet<Name>> {
    ast.sfold(Ok(BTreeSet::new()), find_thread_index_dependent_variables_top)
}

fn write_temporary_result_on_first_thread_only(
    acc: &mut Vec<Stmt>,
    lhs: Expr,
    rhs: Expr,
    int_ty: &Type,
    i: &Info
) {
    let temp_id = Name::sym_str("t");
    let ty = lhs.get_type().clone();
    let temp_var = Expr::Var {
        id: temp_id.clone(),
        ty: ty.clone(),
        i: lhs.get_info().clone()
    };
    let assign_rhs_to_fresh_temp = Stmt::Definition {
        ty: ty.clone(), id: temp_id, expr: rhs, i: i.clone()
    };
    acc.push(assign_rhs_to_fresh_temp);
    // NOTE(larshum, 2025-11-03): By converting the results of both sides to void, we indicate that
    // the resulting values are being ignored (in C). By doing this, we ensure that they are
    // ignored.
    acc.push(Stmt::Expr {
        e: Expr::IfExpr {
            cond: Box::new(Expr::BinOp {
                lhs: Box::new(Expr::ThreadIdx {
                    dim: Dim::X, ty: int_ty.clone(), i: i.clone()
                }),
                op: BinOp::Eq,
                rhs: Box::new(Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()}),
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: i.clone()
            }),
            thn: Box::new(Expr::Convert {
                e: Box::new(Expr::Assign {
                    lhs: Box::new(lhs),
                    rhs: Box::new(temp_var.clone()),
                    ty: ty.clone(),
                    i: i.clone()
                }),
                ty: Type::Void
            }),
            els: Box::new(Expr::Convert {
                e: Box::new(Expr::Int {v: 0, ty: ty.clone(), i: i.clone()}),
                ty: Type::Void
            }),
            ty: Type::Void,
            i: i.clone()
        },
        i: i.clone()
    });
    acc.push(Stmt::Synchronize {scope: SyncScope::Block, i: i.clone()});
}

fn transform_thread_independent_memory_writes_stmt(
    mut acc: Vec<Stmt>,
    stmt: Stmt,
    vars: &BTreeSet<Name>
) -> Vec<Stmt> {
    match stmt {
        Stmt::Expr {e: Expr::Assign {lhs, rhs, ty, i}, i: stmt_i} => {
            if let Expr::ArrayAccess {idx, ..} = lhs.as_ref() {
                if thread_index_dependent_expr(&vars, idx.as_ref()) {
                    acc.push(Stmt::Expr {
                        e: Expr::Assign {
                            lhs: Box::new(*lhs.clone()),
                            rhs: Box::new(*rhs),
                            ty,
                            i: i.clone()
                        },
                        i
                    });
                } else {
                    let int_ty = idx.get_type();
                    write_temporary_result_on_first_thread_only(
                        &mut acc, *lhs.clone(), *rhs, int_ty, &i
                    );
                }
            } else {
                acc.push(Stmt::Expr {
                    e: Expr::Assign {lhs, rhs, ty, i},
                    i: stmt_i
                });
            }
            acc
        }
        Stmt::Definition {..} | Stmt::For {..} | Stmt::If {..} | Stmt::While {..} |
        Stmt::Return {..} | Stmt::Scope {..} | Stmt::Expr {..} |
        Stmt::ParallelReduction {..} | Stmt::Synchronize {..} |
        Stmt::WarpReduce {..} | Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} |
        Stmt::AllocDevice {..} | Stmt::AllocShared {..} | Stmt::FreeDevice {..} |
        Stmt::CopyMemory {..} => {
            stmt.sflatten(acc, |acc, s| {
                transform_thread_independent_memory_writes_stmt(acc, s, &vars)
            })
        },
    }
}

fn transform_thread_independent_memory_writes_stmts(
    stmts: Vec<Stmt>,
    vars: &BTreeSet<Name>
) -> Vec<Stmt> {
    stmts.sflatten(vec![], |acc, s| {
        transform_thread_independent_memory_writes_stmt(acc, s, &vars)
    })
}

fn transform_thread_independent_memory_writes_top(
    top: Top,
    vars: &BTreeSet<Name>
) -> Top {
    match top {
        Top::KernelFunDef {attrs, id, params, body, i} => {
            let body = transform_thread_independent_memory_writes_stmts(body, vars);
            Top::KernelFunDef {attrs, id, params, body, i}
        },
        Top::ExtDecl {..} | Top::FunDef {..} => top,
    }
}

fn transform_thread_independent_memory_writes(
    ast: Ast,
    vars: &BTreeSet<Name>
) -> Ast {
    ast.into_iter()
        .map(|top| {
            transform_thread_independent_memory_writes_top(top, vars)
        })
        .collect::<Ast>()
}


pub fn eliminate_block_wide_memory_writes(ast: Ast) -> CompileResult<Ast> {
    let thread_vars = find_thread_index_dependent_variables(&ast)?;
    Ok(transform_thread_independent_memory_writes(ast, &thread_vars))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;
    use crate::test::*;

    #[test]
    fn test_write_to_array() {
        let ptr_ty = pointer(scalar(ElemSize::I32), MemSpace::Device);
        let lhs = array_access(var("x", ptr_ty), int(0, Some(ElemSize::I32)), scalar(ElemSize::I32));
        let rhs = var("y", scalar(ElemSize::I32));
        let s = assign(lhs.clone(), rhs.clone());
        let r = transform_thread_independent_memory_writes_stmt(vec![], s, &BTreeSet::new());
        if let [a, b, c] = &r[..] {
            if let Stmt::Definition {ty, id, expr, i: _} = a.clone() {
                assert_eq!(ty, scalar(ElemSize::I32));
                assert!(id.has_sym());
                assert_eq!(expr, rhs);
            } else {
                panic!("Unexpected form of initialization of temporary value.");
            };
            if let Stmt::Expr {e: Expr::IfExpr {cond, thn, els, ..}, ..} = b.clone() {
                let l = Expr::ThreadIdx {dim: Dim::X, ty: scalar(ElemSize::I32), i: i()};
                let r = int(0, Some(ElemSize::I32));
                assert_eq!(*cond, binop(l, BinOp::Eq, r, scalar(ElemSize::Bool)));
                if let Expr::Convert {e, ty} = thn.as_ref() {
                    assert!(matches!(ty, Type::Void));
                    if let Expr::Assign {lhs, rhs, ..} = e.as_ref() {
                        assert!(matches!(*lhs.clone(), Expr::ArrayAccess {..}));
                        assert!(matches!(*rhs.clone(), Expr::Var {..}));
                    } else {
                        panic!("Unexpected form of assignment (no assignment)")
                    }
                } else {
                    panic!("Unexpected form of assignment (no conversion)")
                }
                // The else-branch is a conversion expression consisting of an integer literal that
                // we cast to void, so that its type is ignored.
                assert!(matches!(*els, Expr::Convert {ty: Type::Void, ..}));
            } else {
                panic!("Unexpected form of conditional statement");
            };
            assert_eq!(c.clone(), Stmt::Synchronize {scope: SyncScope::Block, i: i()});
        } else {
            panic!("Unexpected form of result");
        }
    }
}
