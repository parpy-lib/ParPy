use super::ast::*;
use super::slices;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;

fn add_shape_dim_if_negative(idx: i64, dim: i64) -> i64 {
    if idx < 0 { idx + dim } else { idx }
}

fn resolve_index_entry(
    idx: Expr,
    dim: i64
) -> PyResult<Expr> {
    match idx {
        Expr::Slice {lo, hi, ty, i} => {
            let lo_idx = slices::extract_slice_index(&lo, 0).unwrap();
            let lo = Expr::Int {v: lo_idx, ty: ty.clone(), i: i.clone()};
            let hi_idx = slices::extract_slice_index(&hi, dim).unwrap();
            let hi_idx = add_shape_dim_if_negative(hi_idx, dim);
            let hi = Expr::Int {v: hi_idx, ty: ty.clone(), i: i.clone()};
            Ok(Expr::Slice {lo: Some(Box::new(lo)), hi: Some(Box::new(hi)), ty, i})
        },
        Expr::Int {v, ty, i} => {
            let idx = add_shape_dim_if_negative(v, dim);
            Ok(Expr::Int {v: idx, ty, i})
        },
        _ => Ok(idx)
    }
}

fn resolve_index_entries(
    idx: Expr,
    shape: &Vec<i64>
) -> PyResult<Expr> {
    match idx {
        Expr::Tuple {elems, ty, i} => {
            if shape.len() == elems.len() {
                let elems = elems.into_iter()
                    .zip(shape.iter())
                    .map(|(e, dim)| resolve_index_entry(e, *dim))
                    .collect::<PyResult<Vec<Expr>>>()?;
                Ok(Expr::Tuple {elems, ty, i})
            } else {
                py_runtime_error!(i, "Unexpected shape of subscript target \
                                      (expected {0} dimensions based on its \
                                      type but {1} indices were provided",
                                      shape.len(), elems.len())
            }
        },
        _ => {
            if shape.len() >= 1 {
                resolve_index_entry(idx, shape[0])
            } else {
                println!("{idx}\n{shape:?}");
                py_runtime_error!(idx.get_info(), "Invalid shape of index target")
            }
        }
    }
}

fn extract_shape(e: &Expr) -> Option<Vec<i64>> {
    match e.get_type() {
        Type::Tensor {shape, ..} => Some(shape.clone()),
        _ => None
    }
}

fn resolve_indices_expr(e: Expr) -> PyResult<Expr> {
    match e {
        Expr::Subscript {target, idx, ty, i} => {
            let target = resolve_indices_expr(*target)?;
            match extract_shape(&target) {
                Some(target_shape) => {
                    let idx = resolve_index_entries(*idx, &target_shape)?;
                    Ok(Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i})
                },
                None => {
                    Ok(Expr::Subscript {target: Box::new(target), idx, ty, i})
                }
            }
        },
        _ => e.smap_result(resolve_indices_expr)
    }
}

fn resolve_indices_stmt(s: Stmt) -> PyResult<Stmt> {
    s.smap_result(resolve_indices_stmt)?
        .smap_result(resolve_indices_expr)
}

pub fn resolve_indices(fun: FunDef) -> PyResult<FunDef> {
    let body = fun.body.smap_result(resolve_indices_stmt)?;
    Ok(FunDef {body, ..fun})
}
