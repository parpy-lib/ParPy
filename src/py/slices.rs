use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use pyo3::prelude::*;

fn count_slicing_dims(elems: &Vec<Expr>) -> i64 {
    let is_slice = |e: &&Expr| match e {
        Expr::Slice {..} => true,
        _ => false
    };
    elems.iter().filter(is_slice).count() as i64
}

fn collect_slice_dims(acc: i64, e: &Expr) -> PyResult<i64> {
    match e {
        Expr::Subscript {idx, ..} => {
            let ndims = match idx.as_ref() {
                Expr::Tuple {elems, ..} => count_slicing_dims(elems),
                _ => count_slicing_dims(&vec![*idx.clone()])
            };
            if acc == 0 || ndims == acc {
                Ok(i64::max(ndims, acc))
            } else {
                py_runtime_error!(e.get_info(), "Inconsistent slicing dimensions in expression")
            }
        },
        _ => e.sfold_result(Ok(acc), collect_slice_dims)
    }
}

fn extract_shape(e: &Expr, i: &Info) -> PyResult<Vec<i64>> {
    match e {
        Expr::Var {i, ..} => {
            py_runtime_error!(i, "Slice result cannot be stored in a variable")
        },
        Expr::Subscript {ty: Type::Tensor {shape, ..}, ..} => Ok(shape.clone()),
        _ => py_runtime_error!(i, "Unexpected left-hand side of slice statement")
    }
}

fn generate_loop_ids(ndims: i64) -> Vec<Name> {
    (0..ndims).into_iter()
        .map(|i| Name::new(format!("slice_{i}")).with_new_sym())
        .collect::<Vec<Name>>()
}

fn insert_loop_ids_indices<'a>(
    ids: &'a [Name],
    e: Expr
) -> (&'a [Name], Expr) {
    match e {
        Expr::Slice {lo, ty, i, ..} => {
            let lo_ofs = match lo {
                Some(e) => match e.as_ref() {
                    Expr::Int {v, ..} => *v,
                    _ => panic!("Invalid form of slice expression in src/py/slices.rs")
                },
                None => 0
            };
            let e = Expr::Var {id: ids[0].clone(), ty, i: i.clone()};
            let e = if lo_ofs > 0 {
                let i64_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
                Expr::BinOp {
                    lhs: Box::new(e),
                    op: BinOp::Add,
                    rhs: Box::new(Expr::Int {
                        v: lo_ofs, ty: i64_ty.clone(), i: i.clone()
                    }),
                    ty: i64_ty,
                    i: i.clone()
                }
            } else {
                e
            };
            (&ids[1..], e)
        },
        _ => e.smap_accum_l(ids, insert_loop_ids_indices)
    }
}

fn insert_loop_ids<'a>(ids: &'a [Name], e: Expr) -> Expr {
    match e {
        Expr::Subscript {target, idx, ty, i} => {
            let (_, idx) = insert_loop_ids_indices(ids, *idx);
            Expr::Subscript {target, idx: Box::new(idx), ty, i}
        },
        _ => e.smap(|e| insert_loop_ids(ids, e))
    }
}

fn insert_for_loops(
    lhs: Expr,
    rhs: Expr,
    dims: Vec<(i64, Name)>,
    _labels: Vec<String>, // TODO: Properly insert these into the loops...
    i: &Info
) -> Stmt {
    let int = |v| Expr::Int {
        v,
        ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]},
        i: i.clone()
    };
    let mut stmt = Stmt::Assign {dst: lhs, expr: rhs, labels: vec![], i: i.clone()};
    for (shape, id) in dims.into_iter() {
        stmt = Stmt::For {
            var: id, lo: int(0), hi: int(shape), step: 1,
            body: vec![stmt], labels: vec![], i: i.clone()
        }
    }
    stmt
}

fn replace_slices_with_for_loops_stmt(stmt: Stmt) -> PyResult<Stmt> {
    match stmt {
        Stmt::Assign {dst, expr, labels, i} => {
            let rhs_dims = collect_slice_dims(0, &expr)?;
            if rhs_dims > 0 {
                let lhs_dims = collect_slice_dims(0, &dst)?;
                if lhs_dims == rhs_dims {
                    let result_shape = extract_shape(&dst, &i)?;
                    let ids = generate_loop_ids(lhs_dims);
                    if result_shape.len() == ids.len() {
                        let dst = insert_loop_ids(&ids[..], dst);
                        let expr = insert_loop_ids(&ids[..], expr);
                        let dims = result_shape.into_iter()
                            .zip(ids.into_iter())
                            .collect::<Vec<(i64, Name)>>();
                        Ok(insert_for_loops(dst, expr, dims, labels, &i))
                    } else {
                        let msg = format!(
                            "Internal error: Found {0} slices, but expected {1}.",
                            result_shape.len(), ids.len()
                        );
                        py_runtime_error!(i, "{}", msg)
                    }
                } else {
                    py_runtime_error!(
                        i,
                        "Slicing dimensionality mismatch between left- and right-hand sides"
                    )
                }
            } else {
                Ok(Stmt::Assign {dst, expr, labels, i})
            }
        },
        // TODO: Support slice operations reducing the dimensions (e.g., sum)
        Stmt::Definition {..} | Stmt::Label {..} | Stmt::For {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::WithGpuContext {..} |
        Stmt::Call {..} => {
            stmt.smap_result(replace_slices_with_for_loops_stmt)
        }
    }
}

pub fn replace_slices_with_for_loops(fun: FunDef) -> PyResult<FunDef> {
    let body = fun.body.smap_result(replace_slices_with_for_loops_stmt)?;
    Ok(FunDef {body, ..fun})
}
