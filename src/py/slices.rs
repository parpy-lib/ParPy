use super::ast::*;
use crate::py_runtime_error;
use crate::par::REDUCE_PAR_LABEL;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::reduce;
use crate::utils::smap::{SFold, SMapAccum};

use pyo3::prelude::*;

use std::collections::BTreeMap;

fn slice_err<T>(i: &Info, spec_msg: &str) -> PyResult<T> {
    let msg =
        "Slices must be of a particular form. Importantly, it is not supported \
         to use slices in a way that requires materializing them.\n\
         Two primary kinds of slice operations are supported:\n\
         1. Mapping operations where the left- and right-hand sides both \
         refer to all dimensions:\n\
           x[:,:] = y[:,:] * z[:,:]\n\
         2. Reducing operations where the left-hand side has fewer dimensions \
         than the right-hand side:\n\
           x[:] = sum(y[:,:], z[:,:], axis=1)";
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
    reduce::builtin_to_reduction_op(op).is_some()
}

fn assert_target_is_variable(target: &Expr, i: &Info) -> PyResult<()> {
    match target {
        Expr::Var {..} => Ok(()),
        Expr::Subscript {target, ..} => assert_target_is_variable(target, i),
        _ => py_runtime_error!(i, "Target of slice must be a variable.")
    }
}

fn assert_all_dimensions_addressed(target_ty: &Type, idx: &Expr, i: &Info) -> PyResult<()> {
    if let Type::Tensor {shape, ..} = target_ty {
        let dims = match idx {
            Expr::Tuple {elems, ..} => elems,
            _ => &vec![idx.clone()]
        };
        if dims.len() == shape.len() {
            Ok(())
        } else {
            py_runtime_error!(i, "Subscript operations containing slices must \
                                  refer to all dimensions of the target. \
                                  Target has {0} dimensions, but only {1} \
                                  dimensions are addressed by the index.",
                                  shape.len(), dims.len())
        }
    } else {
        py_runtime_error!(i, "Target of slice subscript operation has invalid type.")
    }
}

fn validate_slices_expr(acc: (), e: &Expr) -> PyResult<()> {
    match e {
        Expr::Subscript {target, idx, i, ..} if count_slices_subscript_index(0, idx) > 0 => {
            assert_target_is_variable(target, i)?;
            assert_all_dimensions_addressed(target.get_type(), idx, i)?;
            Ok(acc)
        },
        _ => e.sfold_result(Ok(acc), validate_slices_expr)
    }
}

fn contains_no_slices_expr(acc: (), e: &Expr) -> PyResult<()> {
    if count_slices_expr(0, e) > 0 {
        py_runtime_error!(e.get_info(), "Slice expressions are only allowed in \
                                         assignment statements.")
    } else {
        Ok(acc)
    }
}

/// Validates all slices found within the provided body. This validation ensures slices are:
/// - Only used in a definition or assignment.
/// - The target of a subscript containing a slice must be a variable.
/// - Slices refer to all dimensions of the target.
fn validate_slices_stmt(acc: (), s: &Stmt) -> PyResult<()> {
    match s {
        Stmt::Definition {expr, ..} => validate_slices_expr(acc, expr),
        Stmt::Assign {dst, expr, ..} => {
            validate_slices_expr(acc, dst)?;
            validate_slices_expr((), expr)
        },
        _ => {
            let acc = s.sfold_result(Ok(acc), validate_slices_stmt);
            s.sfold_result(acc, contains_no_slices_expr)
        }
    }
}

fn validate_slices(body: &Vec<Stmt>) -> PyResult<()> {
    body.sfold_result(Ok(()), validate_slices_stmt)
}

enum ReduceDim {
    One(i64),
    All,
    None
}

fn find_reduce_dim(rhs: &Expr) -> ReduceDim {
    match rhs {
        Expr::Builtin {func, args, axis, ..} if is_reduction_op(func) && args.len() == 1 => {
            match axis {
                Some(n) => ReduceDim::One(*n),
                None => ReduceDim::All
            }
        },
        _ => ReduceDim::None
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
            let (_, idx) = insert_slice_dim_ids_index(&ids[..], *idx);
            Ok(Expr::Subscript {target, idx: Box::new(idx), ty, i})
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

fn sub_vars_expr(e: Expr, vars: &BTreeMap<Name, Expr>) -> Expr {
    match e {
        Expr::Var {id, ..} if vars.contains_key(&id) => {
            vars.get(&id).unwrap().clone()
        },
        _ => e.smap(|e| sub_vars_expr(e, vars))
    }
}

fn sub_var_stmt_rhs(s: Stmt, id: &Name, sub: &Expr) -> PyResult<Expr> {
    match s {
        Stmt::Definition {expr: Expr::BinOp {rhs, ..}, ..} |
        Stmt::Assign {expr: Expr::BinOp {rhs, ..}, ..} => {
            let mut vars = BTreeMap::new();
            vars.insert(id.clone(), sub.clone());
            Ok(sub_vars_expr(*rhs, &vars))
        },
        _ => py_runtime_error!(s.get_info(), "Unexpected form of statement \
                                              (internal error)")
    }
}

fn replace_ids_with_shape_expr(
    e: Expr,
    reduce_id: &Name,
    dims: &Vec<(Name, i64)>
) -> Expr {
    let i = e.get_info();
    let mut reduce_expr = Expr::Var {
        id: reduce_id.clone(), ty: Type::Unknown, i: i.clone()
    };
    let mut sub_map = BTreeMap::new();
    for (id, sh) in dims[1..].iter().rev() {
        let sub_expr = Expr::BinOp {
            lhs: Box::new(reduce_expr.clone()),
            op: BinOp::Rem,
            rhs: Box::new(Expr::Int {v: *sh, ty: Type::Unknown, i: i.clone()}),
            ty: Type::Unknown,
            i: i.clone()
        };
        sub_map.insert(id.clone(), sub_expr);
        reduce_expr = Expr::BinOp {
            lhs: Box::new(reduce_expr),
            op: BinOp::Div,
            rhs: Box::new(Expr::Int {v: *sh, ty: Type::Unknown, i: i.clone()}),
            ty: Type::Unknown,
            i: i.clone()
        };
    }
    let (last_id, _) = &dims[0];
    sub_map.insert(last_id.clone(), reduce_expr);
    sub_vars_expr(e, &sub_map)
}

enum TargetData {
    Def(Name),
    Assign(Expr)
}

struct ReduceData {
    pub niters: i64,
    pub var_id: Name,
    pub label: Option<String>,
    pub op: BinOp,
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
                // We use the expression produced in the first iteration in the type-checker to
                // determine the type of the neutral element.
                let expr = sub_var_stmt_rhs(stmt.clone(), &r.var_id, &int(0))?;
                let ne = Expr::NeutralElement {
                    op: r.op.clone(), tyof: Box::new(expr), i: i.clone()
                };
                stmt = Stmt::For {
                    var: r.var_id, lo: int(0), hi: int(r.niters), step: 1,
                    body: vec![stmt], labels: l, i: i.clone()
                };
                let pre_stmt = match r.target_data {
                    TargetData::Def(id) => {
                        let ty = Type::Unknown;
                        Stmt::Definition {ty, id, expr: ne, labels: vec![], i: i.clone()}
                    },
                    TargetData::Assign(dst) => {
                        Stmt::Assign {dst, expr: ne, labels: vec![], i: i.clone()}
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

fn validate_nlabels(
    nlabels: usize,
    expected: i64,
    i: &Info,
    msg: &str
) -> PyResult<()> {
    if nlabels == 0 || nlabels as i64 == expected {
        Ok(())
    } else {
        py_runtime_error!(i, "{}", msg)
    }
}

fn assert_slice_count(ok: bool, i: &Info, msg: &str) -> PyResult<()> {
    if !ok {
        py_runtime_error!(i, "{}", msg)
    } else {
        Ok(())
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
        let target_data = match def_id {
            Some(ref id) => TargetData::Def(id.clone()),
            None => TargetData::Assign(lhs.clone())
        };
        match find_reduce_dim(&rhs) {
            ReduceDim::One(n) => {
                let msg = "When reducing along one dimension, the number of \
                           slice dimensions of the left-hand side expression \
                           must be one less than in the right-hand side.";
                assert_slice_count(lslices == rslices - 1, &i, msg)?;
                let msg = format!(
                    "Expected zero or {nslices} labels, found {0}",
                    labels.len()
                );
                validate_nlabels(labels.len(), nslices, &i, &msg)?;
                let idx = n.rem_euclid(nslices) as usize;
                let (reduce_id, reduce_dim) = dims.remove(idx);
                let label = if idx < labels.len() {
                    Some(labels.remove(idx))
                } else {
                    None
                };
                let (op, rhs) = extract_reduction_data(rhs)?;
                let rhs = Expr::BinOp {
                    lhs: Box::new(lhs.clone()), op: op.clone(),
                    rhs: Box::new(rhs), ty: Type::Unknown, i: i.clone()
                };
                let inner_stmt = Stmt::Assign {
                    dst: lhs, expr: rhs, labels: vec![], i
                };
                let reduce_data = ReduceData {
                    niters: reduce_dim,
                    var_id: reduce_id,
                    label, op, target_data
                };
                generate_for_loops(inner_stmt, dims, labels, Some(reduce_data))
            },
            ReduceDim::All => {
                let msg = "When reducing along all dimensions, the left-hand \
                           side expression must be a scalar.";
                assert_slice_count(lslices == 0, &i, msg)?;
                let msg = format!(
                    "Expected zero or one labels for full reduction \
                     operation, found {0}", labels.len()
                );
                validate_nlabels(labels.len(), 1, &i, &msg)?;
                let (op, rhs) = extract_reduction_data(rhs)?;
                let reduce_id = Name::sym_str("reduce_dim");
                let rhs = replace_ids_with_shape_expr(rhs, &reduce_id, &dims);
                let rhs = Expr::BinOp {
                    lhs: Box::new(lhs.clone()), op: op.clone(),
                    rhs: Box::new(rhs), ty: Type::Unknown, i: i.clone()
                };
                let inner_stmt = Stmt::Assign {
                    dst: lhs, expr: rhs, labels: vec![], i
                };
                let reduce_dim = dims.iter().map(|(_, sh)| sh).product();
                let label = if labels.len() == 1 {
                    Some(labels.remove(0))
                } else {
                    None
                };
                let reduce_data = ReduceData {
                    niters: reduce_dim,
                    var_id: reduce_id,
                    label, op, target_data
                };
                generate_for_loops(inner_stmt, vec![], vec![], Some(reduce_data))
            },
            ReduceDim::None => {
                let msg = "Slice statements cannot have more slice dimensions \
                           in the right-hand side expression than in the \
                           left-hand side expression.";
                assert_slice_count(lslices >= rslices, &i, msg)?;
                let msg = format!(
                    "Expected zero or {nslices} labels, found {0}", labels.len()
                );
                validate_nlabels(labels.len(), nslices, &i, &msg)?;
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
    validate_slices(&fun.body)?;
    let body = fun.body.smap_result(replace_slices_with_for_loops_stmt)?;
    let body = body.into_iter().fold(vec![], eliminate_scopes_stmt);
    body.sfold_result(Ok(()), ensure_no_remaining_reduction_ops_stmt)?;
    Ok(FunDef {body, ..fun})
}
