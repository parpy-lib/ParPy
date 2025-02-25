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

fn count_slices_subscript_index(acc: i64, idx: &Expr) -> i64 {
    match idx {
        Expr::Slice {..} => acc + 1,
        _ => idx.sfold(acc, count_slices_subscript_index)
    }
}

fn count_slices_expr(acc: i64, e: &Expr) -> i64 {
    match e {
        Expr::Subscript {target, idx, ..} => {
            let acc = count_slices_expr(acc, target);
            let idx_acc = count_slices_subscript_index(0, idx);
            i64::max(acc, idx_acc)
        },
        _ => e.sfold(acc, count_slices_expr)
    }
}

fn is_reduction_op(op: &Builtin) -> bool {
    match op {
        Builtin::Sum | Builtin::Max | Builtin::Min => true,
        _ => false
    }
}

fn find_reduce_dim(rhs: &Expr) -> Option<i64> {
    match rhs {
        Expr::Builtin {func, axis, ..} if is_reduction_op(func) => *axis,
        _ => None
    }
}

pub fn extract_slice_index(
    o: &Option<Box<Expr>>,
    default: i64
) -> Option<i64> {
    if let Some(e) = o {
        if let Expr::Int {v, ..} = e.as_ref() {
            Some(*v)
        } else {
            None
        }
    } else {
        Some(default)
    }
}

fn insert_slice_dim_ids_index<'a>(
    ids: &'a[Name],
    e: Expr
) -> (&'a[Name], Expr) {
    match e {
        Expr::Slice {lo, i, ty, ..} => {
            let lo = extract_slice_index(&lo, 0).unwrap();
            let slice_dim = Expr::BinOp {
                lhs: Box::new(Expr::Int {v: lo, ty: ty.clone(), i: i.clone()}),
                op: BinOp::Add,
                rhs: Box::new(Expr::Var {id: ids[0].clone(), ty: ty.clone(), i: i.clone()}),
                ty, i
            };
            (&ids[1..], slice_dim)
        },
        _ => e.smap_accum_l(ids, insert_slice_dim_ids_index)
    }
}

fn insert_slice_dim_ids(
    ids: &Vec<Name>,
    e: Expr
) -> PyResult<Expr> {
    match e {
        Expr::Subscript {target, idx, ty, i} => {
            let target = insert_slice_dim_ids(ids, *target)?;
            let (_, idx) = insert_slice_dim_ids_index(&ids[..], *idx);
            Ok(Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i})
        },
        _ => e.smap_result(|e| insert_slice_dim_ids(ids, e))
    }
}

fn extract_shape(e: &Expr) -> PyResult<Vec<i64>> {
    match e {
        Expr::Builtin {func, args, ..} if is_reduction_op(func) => {
            extract_shape(&args[0])
        },
        _ => match e.get_type() {
            Type::Tensor {shape, ..} => Ok(shape.clone()),
            _ => {
                py_runtime_error!(e.get_info(), "Invalid type of slice \
                                                 operation: {0}", e.get_type())
            }
        }
    }
}

fn sub_var_expr(e: Expr, var_id: &Name, sub: &Expr) -> Expr {
    match e {
        Expr::Var {id, ..} if id == *var_id => {
            sub.clone()
        },
        _ => e.smap(|e| sub_var_expr(e, var_id, sub))
    }
}

fn sub_var_stmt_rhs(s: Stmt, id: &Name, sub: &Expr) -> PyResult<Expr> {
    match s {
        Stmt::Definition {expr: Expr::BinOp {rhs, ..}, ..} |
        Stmt::Assign {expr: Expr::BinOp {rhs, ..}, ..} => {
            Ok(sub_var_expr(*rhs, id, sub))
        },
        _ => py_runtime_error!(s.get_info(), "Unexpected form of statement \
                                              (internal error)")
    }
}

enum TargetData {
    Def(Name),
    Assign(Expr)
}

struct ReduceData {
    pub niters: i64,
    pub var_id: Name,
    pub label: Option<String>,
    pub target_data: TargetData
}

fn generate_for_loops(
    inner_stmt: Stmt,
    dims: Vec<(Name, i64)>,
    mut labels: Vec<String>,
    reduce_dim: Option<ReduceData>
) -> PyResult<Stmt> {
    let i = inner_stmt.get_info();
    let int = |v| Expr::Int {v, ty: Type::Unknown, i: i.clone()};
    let mut stmt = inner_stmt;
    match reduce_dim {
        Some(r) => {
            let mut l = r.label.map(|l| vec![l]).unwrap_or(vec![]);
            l.push(REDUCE_PAR_LABEL.to_string());
            if r.niters > 0 {
                // We compute the result of the first iteration outside the loop, and use this as
                // our initial value. This avoids issues with typing, as we would need to infer the
                // type of the neutral element before looking at the loop.
                let expr = sub_var_stmt_rhs(stmt.clone(), &r.var_id, &int(0))?;
                stmt = Stmt::For {
                    var: r.var_id, lo: int(1), hi: int(r.niters), step: 1,
                    body: vec![stmt], labels: l, i: i.clone()
                };
                let pre_stmt = match r.target_data {
                    TargetData::Def(id) => {
                        let ty = Type::Unknown;
                        Stmt::Definition {ty, id, expr, labels: vec![], i: i.clone()}
                    },
                    TargetData::Assign(dst) => {
                        Stmt::Assign {dst, expr, labels: vec![], i: i.clone()}
                    }
                };
                stmt = Stmt::Scope {body: vec![pre_stmt, stmt], i: i.clone()};
            } else {
                py_runtime_error!(i, "Cannot apply reduction on empty dimension")?
            }
        },
        None => ()
    };
    for (id, shape) in dims.into_iter().rev() {
        let for_label = labels.pop().map(|l| vec![l]).unwrap_or(vec![]);
        stmt = Stmt::For {
            var: id, lo: int(0), hi: int(shape), step: 1,
            body: vec![stmt], labels: for_label, i: i.clone()
        }
    }
    Ok(stmt)
}

fn extract_reduction_data(e: Expr) -> PyResult<(BinOp, Expr)> {
    let i = e.get_info();
    match e {
        Expr::Builtin {func, args, ..} => {
            let op = match reduce::builtin_to_reduction_op(&func) {
                Some(op) => Ok(op),
                None => py_runtime_error!(i, "Invalid reduction operation {func}")
            }?;
            if args.len() == 1 {
                Ok((op, args[0].clone()))
            } else {
                py_runtime_error!(i, "Invalid number of arguments of builtin")
            }
        },
        _ => {
            let msg = "Reduction operation must be applied to the whole \
                       right-hand side of an assignment in slice operations.";
            slice_err(&i, msg)
        }
    }
}

fn replace_slices_assignment(
    reconstruct_stmt: impl Fn(Expr, Expr, Vec<String>, Info) -> Stmt,
    def_id: Option<Name>,
    lhs: Expr,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info
) -> PyResult<Stmt> {
    // If either side of the assignment contains at least one slice expression, we consider the
    // corresponding statement to be a slice statement.
    let lslices = count_slices_expr(0, &lhs);
    let rslices = count_slices_expr(0, &rhs);
    let nslices = i64::max(lslices, rslices);
    if nslices > 0 {
        // Ensure that the number of labels are either zero (no labels are specified) or exactly one
        // label is provided for each dimension.
        if !(labels.is_empty() || labels.len() == nslices as usize) {
            let msg = format!(
                "Expected {0} labels but found {1}\nSlice statements must \
                 either be given no labels or exactly one label per \
                 dimension of the slice operation.",
                 nslices, labels.len()
            );
            py_runtime_error!(i, "{}", msg)?
        }

        let ids = (0..nslices).into_iter()
            .map(|_| Name::sym_str("slice_dim"))
            .collect::<Vec<Name>>();
        let shapes = if rslices == nslices {
            extract_shape(&rhs)
        } else {
            extract_shape(&lhs)
        }?;
        let lhs = insert_slice_dim_ids(&ids, lhs)?;
        let rhs = insert_slice_dim_ids(&ids, rhs)?;
        let mut dims = ids.into_iter()
            .zip(shapes.into_iter())
            .collect::<Vec<(Name, i64)>>();
        match find_reduce_dim(&rhs) {
            Some(n) => {
                let idx = n.rem_euclid(nslices) as usize;
                let (reduce_id, reduce_dim) = dims.remove(idx);
                let label = if idx < labels.len() {
                    Some(labels.remove(idx))
                } else {
                    None
                };
                let target_data = match def_id {
                    Some(id) => TargetData::Def(id),
                    None => TargetData::Assign(lhs.clone())
                };
                let (op, rhs) = extract_reduction_data(rhs)?;
                let rhs = Expr::BinOp {
                    lhs: Box::new(lhs.clone()), op, rhs: Box::new(rhs),
                    ty: Type::Unknown, i: i.clone()
                };
                let inner_stmt = Stmt::Assign {
                    dst: lhs, expr: rhs, labels: vec![], i
                };
                let reduce_data = ReduceData {
                    niters: reduce_dim,
                    var_id: reduce_id,
                    label, target_data
                };
                generate_for_loops(inner_stmt, dims, labels, Some(reduce_data))
            },
            None => {
                match def_id {
                    None => {
                        let inner_stmt = Stmt::Assign {
                            dst: lhs, expr: rhs, labels: vec![], i
                        };
                        generate_for_loops(inner_stmt, dims, labels, None)
                    },
                    Some(id) => {
                        let msg = format!(
                            "Slice expression cannot be assigned to fresh \
                             variable {id}."
                        );
                        slice_err(&i, &msg)
                    }
                }
            }
        }
    } else {
        Ok(reconstruct_stmt(lhs, rhs, labels, i))
    }
}

fn replace_slices_with_for_loops_stmt(s: Stmt) -> PyResult<Stmt> {
    match s {
        Stmt::Definition {ty, id, expr, labels, i} => {
            let reconstruct_def = |lhs, rhs, labels, i| {
                if let Expr::Var {id, ty, ..} = lhs {
                    Stmt::Definition {ty, id, expr: rhs, labels, i}
                } else {
                    unreachable!()
                }
            };
            let def_data = Some(id.clone());
            let lhs = Expr::Var {id, ty, i: i.clone()};
            replace_slices_assignment(reconstruct_def, def_data, lhs, expr, labels, i)
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let reconstruct_assign = |lhs, rhs, labels, i| {
                Stmt::Assign {dst: lhs, expr: rhs, labels, i}
            };
            let def_data = None;
            replace_slices_assignment(reconstruct_assign, def_data, dst, expr, labels, i)
        },
        Stmt::Label {..} | Stmt::For {..} | Stmt::While {..} | Stmt::If {..} |
        Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Call {..} => {
            s.smap_result(replace_slices_with_for_loops_stmt)
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
