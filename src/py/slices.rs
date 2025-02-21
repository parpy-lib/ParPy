use super::ast::*;
use crate::py_runtime_error;
use crate::par::REDUCE_PAR_LABEL;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::reduce;
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

fn extract_reduction_data(e: &Expr) -> PyResult<(BinOp, Expr)> {
    let i = e.get_info();
    match e {
        Expr::Builtin {func, args, ..} => {
            let op = match reduce::builtin_to_reduction_op(func) {
                Some(op) => Ok(op),
                None => py_runtime_error!(i, "Invalid reduction operation {func}")
            }?;
            if args.len() == 1 {
                Ok((op, args[0].clone()))
            } else {
                py_runtime_error!(i, "Invalid number of arguments of built-in {func}")
            }
        },
        Expr::Convert {e, ..} => extract_reduction_data(e),
        _ => {
            let msg =
                "Reduction operation must be applied to the whole right-hand \
                 side of an assignment in slice operations.";
            slice_err(&i, msg)
        }
    }
}

fn transform_reduction_to_elementwise(
    lhs: Expr,
    rhs: Expr
) -> PyResult<Expr> {
    let i = rhs.get_info();
    let (op, arg) = extract_reduction_data(&rhs)?;
    let ty = match lhs.get_type() {
        Type::Tensor {sz, ..} => Type::Tensor {sz: sz.clone(), shape: vec![]},
        _ => py_runtime_error!(i, "Invalid type of slice operation")?
    };
    Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(arg), ty, i})
}

enum TargetData {
    Def(Type, Name),
    Assign(Expr)
}

struct ReduceData {
    pub niters: i64,
    pub var_id: Name,
    pub label: Option<String>,
    pub ne: Expr,
    pub lhs: TargetData
}

fn insert_for_loops(
    inner_stmt: Stmt,
    dims: Vec<(i64, Name)>,
    mut labels: Vec<String>,
    reduce_dim: Option<ReduceData>
) -> PyResult<Stmt> {
    let i = inner_stmt.get_info();
    let int = |v| Expr::Int {
        v,
        ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]},
        i: i.clone()
    };
    let mut stmt = inner_stmt;
    match reduce_dim {
        Some(r) => {
            let mut l = r.label.map(|l| vec![l]).unwrap_or(vec![]);
            l.push(REDUCE_PAR_LABEL.to_string());
            stmt = Stmt::For {
                var: r.var_id, lo: int(0), hi: int(r.niters), step: 1,
                body: vec![stmt], labels: l, i: i.clone()
            };
            let pre_stmt = match r.lhs {
                TargetData::Def(ty, id) => {
                    Stmt::Definition {ty, id, expr: r.ne, labels: vec![], i: i.clone()}
                },
                TargetData::Assign(dst) => {
                    Stmt::Assign {dst, expr: r.ne, labels: vec![], i: i.clone()}
                }
            };
            stmt = Stmt::Scope {body: vec![pre_stmt, stmt], i: i.clone()};
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
    lhs_data: TargetData,
    lhs: Expr,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info
) -> PyResult<Stmt> {
    let construct_inner_stmt = |lhs, rhs, i| {
        Stmt::Assign {dst: lhs, expr: rhs, labels: vec![], i}
    };
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
            let (op, _) = extract_reduction_data(&rhs)?;
            let sz = lhs.get_type()
                .get_scalar_elem_size()
                .unwrap()
                .clone();
            let ne = reduce::reduction_op_neutral_element(&op, sz, i.clone())
                .unwrap();
            let mut reduce_data = ReduceData {
                niters: 0, var_id: reduce_id, label: reduce_label, ne, lhs: lhs_data
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
                    reduce_data.niters = reduce_shape;
                    let inner_stmt = construct_inner_stmt(lhs, rhs, i);
                    insert_for_loops(inner_stmt, dims, labels, Some(reduce_data))
                } else {
                    let msg = format!(
                        "Expected slice reduction over the {n}:th dimension, \
                         but found {lhs_dims} slices in LHS (expected {0}).",
                         lhs_dims+1
                    );
                    slice_err(&i, &msg)
                }
            } else {
                // Reduction over all dimensions in a single loop.
                if lhs_dims == 0 {
                    reduce_data.niters = reduce_shape.into_iter().product();
                    let inner_stmt = construct_inner_stmt(lhs, rhs, i);
                    insert_for_loops(inner_stmt, vec![], labels, Some(reduce_data))
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
                let inner_stmt = construct_inner_stmt(lhs, rhs, i);
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
            let def_data = TargetData::Def(ty.clone(), id.clone());
            let lhs = Expr::Var {id, ty, i: i.clone()};
            replace_slices_assignment(reconstruct_def, def_data, lhs, expr, labels, i)
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let reconstruct_assign = |lhs, rhs, labels, i| {
                Stmt::Assign {dst: lhs, expr: rhs, labels, i}
            };
            let assign_data = TargetData::Assign(dst.clone());
            replace_slices_assignment(reconstruct_assign, assign_data, dst, expr, labels, i)
        },
        Stmt::Label {..} | Stmt::For {..} | Stmt::While {..} | Stmt::If {..} |
        Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Call {..} => {
            stmt.smap_result(replace_slices_with_for_loops_stmt)
        }
    }
}

fn eliminate_scopes_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {var, lo, hi, step, body, labels, i} => {
            let body = body.into_iter().fold(vec![], eliminate_scopes_stmt);
            acc.push(Stmt::For {var, lo, hi, step, body, labels, i});
        },
        Stmt::While {cond, body, i} => {
            let body = body.into_iter().fold(vec![], eliminate_scopes_stmt);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = thn.into_iter().fold(vec![], eliminate_scopes_stmt);
            let els = els.into_iter().fold(vec![], eliminate_scopes_stmt);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::WithGpuContext {body, i} => {
            let body = body.into_iter().fold(vec![], eliminate_scopes_stmt);
            acc.push(Stmt::WithGpuContext {body, i});
        },
        Stmt::Scope {body, ..} => {
            acc = body.into_iter().fold(acc, eliminate_scopes_stmt);
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Call {..} |
        Stmt::Label {..} => {
            acc.push(s);
        },
    };
    acc
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
    let body = body.into_iter().fold(vec![], eliminate_scopes_stmt);
    body.sfold_result(Ok(()), ensure_no_remaining_slices_stmt)?;
    body.sfold_result(Ok(()), ensure_no_remaining_reduction_ops_stmt)?;
    Ok(FunDef {body, ..fun})
}
