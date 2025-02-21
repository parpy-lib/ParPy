use super::ast::*;
use crate::py_runtime_error;
use crate::par::REDUCE_PAR_LABEL;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use pyo3::prelude::*;

fn slice_err<T>(i: &Info, spec_msg: &str) -> PyResult<T> {
    let msg =
        "Slices must be of a particular form. Importantly, it is not supported \
         to use slices in a way that requires materializing them.\n\
         Two primary kinds of slice operations are supported:\n\
         1. Mapping operations where the left- and right-hand sides both \
         refer to all dimensions:\n\
           x[0:N,0:M] = y[0:N,0:M] * z[0:N,0:M]\n\
         2. Reducing operations where the left-hand side has fewer dimensions \
         than the right-hand side:\n\
           x[0:N] = sum(y[0:N,0:M], z[0:N,0:M], axis=1)";
    py_runtime_error!(i, "{spec_msg}\n{msg}")
}

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

fn is_reduction_op(op: &Builtin) -> bool {
    match op {
        Builtin::Sum | Builtin::Max | Builtin::Min => true,
        _ => false
    }
}

fn find_reduction_dim(e: &Expr) -> PyResult<Option<i64>> {
    match e {
        Expr::Builtin {func, axis, ..} if is_reduction_op(func) => Ok(*axis),
        Expr::Convert {e, ..} => find_reduction_dim(e),
        _ => {
            py_runtime_error!(
                e.get_info(),
                "Expected reduction operation in the RHS of assignment"
            )
        }
    }
}

fn extract_shape(e: &Expr) -> PyResult<Vec<i64>> {
    match e {
        Expr::Builtin {func, args, ..} if is_reduction_op(func) => {
            extract_shape(&args[0])
        },
        Expr::Convert {e, ..} => extract_shape(e),
        _ => {
            let ty = e.get_type();
            match ty {
                Type::Tensor {shape, ..} => Ok(shape.clone()),
                _ => {
                    py_runtime_error!(e.get_info(), "Invalid type of slice operation: {ty}")
                }
            }
        }
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

fn transform_reduction_to_elementwise(
    lhs: Expr,
    rhs: Expr
) -> PyResult<Expr> {
    let i = rhs.get_info();
    match rhs {
        Expr::Builtin {func, mut args, ..} => {
            if args.len() == 1 {
                let rhs = args.remove(0);
                let op = match func {
                    Builtin::Sum => BinOp::Add,
                    Builtin::Max => BinOp::Max,
                    Builtin::Min => BinOp::Min,
                    _ => py_runtime_error!(i, "Invalid reduction operation {func}")?
                };
                let ty = match lhs.get_type() {
                    Type::Tensor {sz, ..} => {
                        Type::Tensor {sz: sz.clone(), shape: vec![]}
                    },
                    _ => py_runtime_error!(i, "Invalid type of slice operation")?
                };
                Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
            } else {
                py_runtime_error!(i, "Invalid number of arguments of built-in {func}")
            }
        },
        Expr::Convert {e, ty} => {
            let e = Box::new(transform_reduction_to_elementwise(lhs, *e)?);
            Ok(Expr::Convert {e, ty})
        },
        _ => {
            let msg =
                "Reduction operation must apply to the whole right-hand side \
                 of the assignment statement.";
            slice_err(&i, msg)
        }
    }
}

fn insert_for_loops(
    inner_stmt: Stmt,
    dims: Vec<(i64, Name)>,
    mut labels: Vec<String>,
    reduce_dim: Option<(i64, Name, Option<String>)>
) -> PyResult<Stmt> {
    let i = inner_stmt.get_info();
    let int = |v| Expr::Int {
        v,
        ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]},
        i: i.clone()
    };
    let mut stmt = inner_stmt;
    match reduce_dim {
        Some((shape, id, label)) => {
            let mut l = label.map(|l| vec![l]).unwrap_or(vec![]);
            l.push(REDUCE_PAR_LABEL.to_string());
            stmt = Stmt::For {
                var: id, lo: int(0), hi: int(shape), step: 1,
                body: vec![stmt], labels: l, i: i.clone()
            };
        },
        None => ()
    };
    for (shape, id) in dims.into_iter().rev() {
        let for_label = labels.pop().map(|l| vec![l]).unwrap_or(vec![]);
        stmt = Stmt::For {
            var: id, lo: int(0), hi: int(shape), step: 1,
            body: vec![stmt], labels: for_label, i: i.clone()
        }
    }
    Ok(stmt)
}

fn replace_slices_assignment(
    reconstruct_stmt: impl Fn(Expr, Expr, Vec<String>, Info) -> Stmt,
    lhs: Expr,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info
) -> PyResult<Stmt> {
    let rhs_dims = collect_slice_dims(0, &rhs)?;
    if !labels.is_empty() && labels.len() < rhs_dims as usize {
        let msg =
            "Slice operations cannot be partially labelled; either the operation \
             should not be labelled at all, or it should be provided one label \
             corresponding to each dimension in the right-hand side, provided \
             in a left-to-right order.";
        py_runtime_error!(i, "{}", msg)?
    };
    if rhs_dims > 0 {
        if let Expr::Var {..} = &lhs {
            match &lhs.get_type() {
                Type::Tensor {shape, ..} if !shape.is_empty() => {
                    let msg =
                        "Assigning a non-scalar result of a slice operation to \
                         a fresh variable is not supported, as this would \
                         require allocating memory to materialize it.";
                    slice_err(&i, &msg)
                },
                Type::Tensor {..} => Ok(()),
                _ => py_runtime_error!(i, "Found invalid type in slice operation")
            }?
        };
        let lhs_shape = extract_shape(&lhs)?;
        let lhs_dims = lhs_shape.len() as i64;
        let ids = generate_loop_ids(lhs_dims);
        // If we do a reduction in the right-hand side of the assignment, we should have fewer
        // dimensions (depending on how we reduce), whereas we should otherwise have the same
        // number of dimensions on both sides (all dimensions must be mentioned in the slices).
        if let Ok(reduce_dim) = find_reduction_dim(&rhs) {
            let n = match reduce_dim {
                Some(n) => n as usize,
                None => 0
            };
            let reduce_id = Name::sym_str("slice_reduce_dim");
            let mut rhs_ids = ids.clone();
            rhs_ids.insert(n, reduce_id.clone());
            let lhs = insert_loop_ids(&ids[..], lhs);
            let rhs = insert_loop_ids(&rhs_ids[..], rhs);
            let reduce_label = if n < labels.len() {
                Some(labels.remove(n))
            } else {
                None
            };
            let mut reduce_shape = extract_shape(&rhs)?;
            let rhs = transform_reduction_to_elementwise(lhs.clone(), rhs)?;
            if let Some(_) = reduce_dim {
                // Reduction over one dimension.
                if lhs_dims + 1 == rhs_dims {
                    let dims = lhs_shape.into_iter()
                        .zip(ids.into_iter())
                        .collect::<Vec<(i64, Name)>>();
                    let reduce_shape = reduce_shape.remove(n);
                    let reduce_dim = (reduce_shape, reduce_id, reduce_label);
                    let inner_stmt = reconstruct_stmt(lhs, rhs, vec![], i);
                    insert_for_loops(inner_stmt, dims, labels, Some(reduce_dim))
                } else {
                    let msg = format!(
                        "Expected slice reduction over the {n}:th dimension, \
                         but found {lhs_dims} slices in LHS (expected {0}).",
                         lhs_dims+1
                    );
                    slice_err(&i, &msg)
                }
            } else {
                // Reduction over all dimensions.
                if lhs_dims == 0 {
                    let reduce_shape = reduce_shape.into_iter().product();
                    let reduce_dim = (reduce_shape, reduce_id, reduce_label);
                    let inner_stmt = reconstruct_stmt(lhs, rhs, vec![], i);
                    insert_for_loops(inner_stmt, vec![], labels, Some(reduce_dim))
                } else {
                    let msg = format!(
                        "Expected slice reduction over all dimensions, but \
                         found {lhs_dims} slices in LHS (expected zero)."
                    );
                    slice_err(&i, &msg)
                }
            }
        } else {
            if lhs_dims == rhs_dims {
                let lhs = insert_loop_ids(&ids[..], lhs);
                let rhs = insert_loop_ids(&ids[..], rhs);
                let dims = lhs_shape.into_iter()
                    .zip(ids.into_iter())
                    .collect::<Vec<(i64, Name)>>();
                let inner_stmt = reconstruct_stmt(lhs, rhs, vec![], i);
                insert_for_loops(inner_stmt, dims, labels, None)
            } else {
                let msg = format!(
                    "Expected mapping slice operation, but found {lhs_dims} \
                     slices in LHS and {rhs_dims} in RHS."
                );
                slice_err(&i, &msg)
            }
        }
    } else {
        Ok(reconstruct_stmt(lhs, rhs, labels, i))
    }
}

fn replace_slices_with_for_loops_stmt(stmt: Stmt) -> PyResult<Stmt> {
    match stmt {
        Stmt::Definition {ty, id, expr, labels, i} => {
            let reconstruct_def = |lhs, rhs, labels, i| {
                if let Expr::Var {id, ty, ..} = lhs {
                    Stmt::Definition {ty, id, expr: rhs, labels, i}
                } else {
                    unreachable!()
                }
            };
            let lhs = Expr::Var {id, ty, i: i.clone()};
            replace_slices_assignment(reconstruct_def, lhs, expr, labels, i)
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let reconstruct_assign = |lhs, rhs, labels, i| {
                Stmt::Assign {dst: lhs, expr: rhs, labels, i}
            };
            replace_slices_assignment(reconstruct_assign, dst, expr, labels, i)
        },
        Stmt::Label {..} | Stmt::For {..} | Stmt::While {..} | Stmt::If {..} |
        Stmt::WithGpuContext {..} | Stmt::Call {..} => {
            stmt.smap_result(replace_slices_with_for_loops_stmt)
        }
    }
}

fn unsupported_slice_error(i: &Info) -> PyResult<()> {
    let msg =
        "Found slice expression in unsupported position.\n\
         Slicing statements should be used in assignments, where all dimensions \
         are mentioned on the right-hand side, and at most one is omitted in \
         the left-hand side expression.";
    py_runtime_error!(i, "{}", msg)
}

fn ensure_no_remaining_slices_expr(acc: (), e: &Expr) -> PyResult<()> {
    match e {
        Expr::Slice {i, ..} => unsupported_slice_error(i),
        _ => e.sfold_result(Ok(acc), ensure_no_remaining_slices_expr)
    }
}

fn ensure_no_remaining_slices_stmt(acc: (), s: &Stmt) -> PyResult<()> {
    let _ = s.sfold_result(Ok(acc), ensure_no_remaining_slices_stmt)?;
    s.sfold_result(Ok(acc), ensure_no_remaining_slices_expr)
}

fn ensure_no_remaining_reduction_ops_expr(acc: (), e: &Expr) -> PyResult<()> {
    match e {
        Expr::Builtin {func, args, i, ..} if is_reduction_op(func) && args.len() == 1 => {
            let msg = format!(
                "Found reduction operation {func} in unsupported position.\n\
                 Reduce operations must be used on a slice expression, \
                 containing the right-hand side of an assigment."
            );
            py_runtime_error!(i, "{}", msg)
        },
        _ => e.sfold_result(Ok(acc), ensure_no_remaining_reduction_ops_expr)
    }
}

fn ensure_no_remaining_reduction_ops_stmt(acc: (), s: &Stmt) -> PyResult<()> {
    let _ = s.sfold_result(Ok(acc), ensure_no_remaining_reduction_ops_stmt)?;
    s.sfold_result(Ok(acc), ensure_no_remaining_reduction_ops_expr)
}

pub fn replace_slices_with_for_loops(fun: FunDef) -> PyResult<FunDef> {
    let body = fun.body.smap_result(replace_slices_with_for_loops_stmt)?;
    body.sfold_result(Ok(()), ensure_no_remaining_slices_stmt)?;
    body.sfold_result(Ok(()), ensure_no_remaining_reduction_ops_stmt)?;
    Ok(FunDef {body, ..fun})
}
