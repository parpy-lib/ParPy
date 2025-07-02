use super::ast::*;
use super::slices;
use crate::py_runtime_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;

fn add_shape_dim_if_negative(idx: i128, dim: i128) -> i128 {
    if idx < 0 { idx + dim } else { idx }
}

fn resolve_index_entry(
    idx: Expr,
    dim: i128
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
    let (dims, ty, i) = match idx {
        Expr::Tuple {elems, ty, i} => (elems, ty, i),
        _ => {
            let ty = idx.get_type().clone();
            let i = idx.get_info();
            (vec![idx], ty, i)
        }
    };
    if dims.len() <= shape.len() {
        let elems = dims.into_iter()
            .zip(shape.iter())
            .map(|(e, dim)| resolve_index_entry(e, *dim as i128))
            .collect::<PyResult<Vec<Expr>>>()?;
        Ok(Expr::Tuple {elems, ty, i})
    } else {
        py_runtime_error!(i, "Received too many indices ({0} indices for a \
                              tensor of {1} dimensions).", dims.len(), shape.len())
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

fn resolve_indices_def(fun: FunDef) -> PyResult<FunDef> {
    let body = fun.body.smap_result(resolve_indices_stmt)?;
    Ok(FunDef {body, ..fun})
}

pub fn resolve_indices(ast: Ast) -> PyResult<Ast> {
    ast.smap_result(resolve_indices_def)
}
