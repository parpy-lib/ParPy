use super::ast::*;
use super::constant_fold;
use crate::py_internal_error;
use crate::py_runtime_error;
use crate::py_type_error;
use crate::par;
use crate::utils::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;
use crate::utils::reduce;
use crate::utils::smap::*;
use crate::utils::substitute::SubVars;

use itertools::Itertools;
use pyo3::prelude::*;
use std::collections::BTreeMap;

fn count_slice_dims_index(acc: usize, idx: &Expr) -> usize {
    match idx {
        Expr::Slice {..} => acc + 1,
        _ => idx.sfold(acc, count_slice_dims_index)
    }
}

fn count_slice_dims_expr(acc: usize, e: &Expr) -> usize {
    match e {
        Expr::Subscript {target, idx, ..} => {
            let acc = count_slice_dims_expr(acc, target);
            let idx_acc = count_slice_dims_index(0, idx);
            usize::max(acc, idx_acc)
        },
        _ => e.sfold(acc, count_slice_dims_expr)
    }
}

fn is_reduction(rhs: &Expr) -> bool {
    match rhs {
        Expr::ReduceOp {..} => true,
        Expr::Convert {e, ..} => is_reduction(e),
        _ => false
    }
}

fn unify_shapes_helper<'a>(
    mut acc: Vec<i64>,
    mut l: impl Iterator<Item=&'a i64>,
    mut r: impl Iterator<Item=&'a i64>
) -> Option<Vec<i64>> {
    match (l.next(), r.next()) {
        (Some(ls), Some(rs)) if ls == rs || *ls == 1 || *rs == 1 => {
            acc.push(i64::max(*ls, *rs));
            unify_shapes_helper(acc, l, r)
        },
        (Some(_), Some(_)) => None,
        (Some(s), None) | (None, Some(s)) => {
            acc.push(*s);
            unify_shapes_helper(acc, l, r)
        },
        (None, None) => Some(acc)
    }
}

fn unify_shapes(
    lshape: Vec<i64>,
    rshape: Vec<i64>,
    i: &Info
) -> PyResult<Vec<i64>> {
    let lit = lshape.iter().rev();
    let rit = rshape.iter().rev();
    match unify_shapes_helper(vec![], lit, rit) {
        Some(mut acc) => {
            acc.reverse();
            Ok(acc)
        },
        None => {
            let lsh = lshape.iter().join(", ");
            let rsh = rshape.iter().join(", ");
            py_type_error!(i, "Found incompatible shapes [{lsh}] and [{rsh}].")
        }
    }
}

fn extract_tensor_shape(shape: &Vec<TensorShape>, i: &Info) -> PyResult<Vec<i64>> {
    shape.iter()
        .map(|sh| match sh {
            TensorShape::Num {n} => Ok(*n),
            TensorShape::Symbol {..} => {
                py_internal_error!(i, "Found shape symbol in slice transformation.")
            }
        })
        .collect::<PyResult<Vec<i64>>>()
}

fn extract_target_shape(target: &Expr) -> PyResult<Vec<i64>> {
    let i = target.get_info();
    let ty = target.get_type();
    match ty {
        Type::Tensor {shape, ..} => extract_tensor_shape(shape, &i),
        _ => py_runtime_error!(i, "Invalid type of slice target {ty}")
    }
}

fn extract_slice_lower_bound(lo: &Option<Box<Expr>>, i: &Info) -> PyResult<i128> {
    match lo {
        Some(l) => match l.as_ref() {
            Expr::Int {v, ..} => Ok(*v),
            _ => py_runtime_error!(i, "Failed to statically determine lower-bound of slice."),
        },
        None => Ok(0)
    }
}

fn extract_slice_upper_bound(hi: &Option<Box<Expr>>, dim: i64, i: &Info) -> PyResult<i128> {
    let n = match hi {
        Some(u) => match u.as_ref() {
            Expr::Int {v, ..} => Ok(*v),
            _ => py_runtime_error!(i, "Failed to statically determine upper-bound of slice."),
        },
        None => Ok(dim as i128)
    }?;
    if n < 0 { Ok(n + dim as i128) } else { Ok(n) }
}

fn extract_index_dim(dim: i64, idx: &Expr) -> PyResult<i64> {
    match idx {
        Expr::Slice {lo, hi, i, ..} => {
            let l = extract_slice_lower_bound(lo, &i)?;
            let u = extract_slice_upper_bound(hi, dim, &i)?;
            if l < u {
                Ok((u - l) as i64)
            } else {
                py_runtime_error!(i, "Slice lower-bound ({l}) must be less than \
                                      its upper-bound ({u}).")
            }
        },
        _ => Ok(0)
    }
}

fn extract_index_dims(shape: Vec<i64>, idx: &Expr) -> PyResult<Vec<i64>> {
    match idx {
        Expr::Tuple {elems, i, ..} => {
            if shape.len() == elems.len() {
                elems.iter()
                    .zip(shape.iter())
                    .map(|(e, dim)| extract_index_dim(*dim, &e))
                    .collect::<PyResult<Vec<i64>>>()
            } else {
                py_runtime_error!(i, "Tuple index does not address all dimensions of target.")
            }
        },
        _ => Ok(vec![extract_index_dim(shape[0], idx)?])
    }
}

fn extract_shape_expr(acc: Vec<i64>, e: &Expr) -> PyResult<Vec<i64>> {
    match e {
        Expr::Subscript {target, idx, i, ..} => {
            let shape = extract_target_shape(&target)?;
            let dims = extract_index_dims(shape, &idx)?;
            let result_shape = dims.into_iter()
                .filter(|dim| *dim > 0)
                .collect::<Vec<i64>>();
            unify_shapes(acc, result_shape, &i)
        },
        _ => e.sfold_result(Ok(acc), extract_shape_expr)
    }
}

fn extract_shape(
    lhs: &Expr,
    rhs: &Expr,
    i: &Info
) -> PyResult<Vec<i64>> {
    let lsh = extract_shape_expr(vec![], &lhs)?;
    let rsh = extract_shape_expr(vec![], &rhs)?;
    unify_shapes(lsh, rsh, &i)
}

fn insert_slice_dim_ids_index<'a>(ids: &'a[Name], e: Expr) -> (&'a[Name], Expr) {
    match e {
        Expr::Slice {lo, i, ty, ..} => {
            let l = extract_slice_lower_bound(&lo, &i).unwrap();
            let slice_dim = Expr::BinOp {
                lhs: Box::new(Expr::Int {v: l, ty: ty.clone(), i: i.clone()}),
                op: BinOp::Add,
                rhs: Box::new(Expr::Var {id: ids[0].clone(), ty: ty.clone(), i: i.clone()}),
                ty, i
            };
            (&ids[1..], slice_dim)
        },
        _ => e.smap_accum_l(ids, insert_slice_dim_ids_index)
    }
}

fn insert_slice_dim_ids(ids: &[Name], e: Expr) -> Expr {
    match e {
        Expr::Subscript {target, idx, ty, i} => {
            let (_, idx) = insert_slice_dim_ids_index(ids, *idx);
            Expr::Subscript {target, idx: Box::new(idx), ty, i}
        },
        _ => e.smap(|e| insert_slice_dim_ids(ids, e))
    }
}

fn mk_int(v: i128, scalar_sizes: &ScalarSizes, i: &Info) -> Expr {
    let ty = Type::fixed_scalar(scalar_sizes.int.clone());
    Expr::Int {v, ty, i: i.clone()}
}

fn derive_dims_from_reduce_id_helper(
    mut sub_map: BTreeMap<Name, Expr>,
    dims: &[(i64, Name)],
    reduce_var: &Expr,
    scalar_sizes: &ScalarSizes
) -> BTreeMap<Name, Expr> {
    if dims.is_empty() {
        sub_map
    } else {
        let i = reduce_var.get_info();
        let (dim, id) = &dims[0];
        let right_dims = &dims[1..];
        let divn = right_dims.iter().map(|(n, _)| *n as i128).product();
        let remn = *dim as i128;
        let sub_expr = Expr::BinOp {
            lhs: Box::new(Expr::BinOp {
                lhs: Box::new(reduce_var.clone()),
                op: BinOp::FloorDiv,
                rhs: Box::new(mk_int(divn, &scalar_sizes, &i)),
                ty: reduce_var.get_type().clone(),
                i: i.clone()
            }),
            op: BinOp::Rem,
            rhs: Box::new(mk_int(remn, &scalar_sizes, &i)),
            ty: reduce_var.get_type().clone(),
            i: i.clone()
        };
        sub_map.insert(id.clone(), sub_expr);
        derive_dims_from_reduce_id_helper(sub_map, &dims[1..], reduce_var, scalar_sizes)
    }
}


fn derive_dims_from_reduce_id(
    e: Expr,
    reduce_id: &Name,
    dims: &Vec<(i64, Name)>,
    scalar_sizes: &ScalarSizes
) -> Expr {
    let reduce_var = Expr::Var {
        id: reduce_id.clone(), ty: e.get_type().clone(), i: e.get_info()
    };
    if dims.len() == 1 {
        let (_, id) = &dims[0];
        let mut sub_map = BTreeMap::new();
        sub_map.insert(id.clone(), reduce_var);
        e.sub_vars(&sub_map)
    } else {
        let sub_map = derive_dims_from_reduce_id_helper(
            BTreeMap::new(), &dims[..], &reduce_var, &scalar_sizes
        );
        e.sub_vars(&sub_map)
    }
}

enum ReduceTargetType {
    Definition {e: Expr},
    Assign {e: Expr},
}

impl ReduceTargetType {
    fn get_expr<'a>(&'a self) -> &'a Expr {
        match self {
            ReduceTargetType::Definition {e} => e,
            ReduceTargetType::Assign {e} => e,
        }
    }
}

fn extract_reduction(rhs: Expr, i: &Info) -> PyResult<(BinOp, Expr)> {
    match rhs {
        Expr::ReduceOp {op, arg, ..} => Ok((op.to_bin_op(), *arg)),
        Expr::Convert {e, ty, i: conv_i} => {
            let (op, arg) = extract_reduction(*e, &i)?;
            Ok((op, Expr::Convert {e: Box::new(arg), ty, i: conv_i}))
        },
        _ => py_internal_error!(i, "Invalid form of reduction operation")
    }
}

fn find_neutral_element(op: &BinOp, ty: &Type, i: &Info) -> PyResult<Expr> {
    match ty.get_scalar_elem_size() {
        Some(sz) => match reduce::neutral_element(&op, &sz, &i) {
            Some(e) => Ok(e),
            None => {
                let op = op.pprint_default();
                py_runtime_error!(i, "Failed to find neutral element for \
                                      reduction operation {op}.")
            }
        }
        None => py_type_error!(i, "Invalid type {ty} of reduction")
    }
}

fn generate_reduction_loop(
    lhs: ReduceTargetType,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info,
    dims: Vec<(i64, Name)>,
    scalar_sizes: &ScalarSizes
) -> PyResult<Stmt> {
    let reduce_id = Name::sym_str("reduce_dim");
    let rhs = derive_dims_from_reduce_id(rhs, &reduce_id, &dims, &scalar_sizes);

    // Extract the labels associated with the reduction (must be either zero or one), and add a
    // special label to inform later stages of the compiler that this is a parallel reduction.
    let l = match labels.len() {
        0 => Ok(vec![par::REDUCE_PAR_LABEL.to_string()]),
        1 => Ok(vec![labels.pop().unwrap(), par::REDUCE_PAR_LABEL.to_string()]),
        n => {
            py_runtime_error!(
                i,
                "Found {n} labels on reduction statement.\n\
                 A reduction operation should be annotated with either zero \
                 or one labels (the label refers to all dimensions we reduce \
                 over)."
            )
        }
    }?;

    // Extract the relevant values from the left- and right-hand side arguments of the reduction
    // operation. The exact form of a reduction is checked before this function is called, so we
    // report an internal error if these assumptions do not hold.
    let ty = lhs.get_expr().get_type().clone();
    let (op, rhs) = extract_reduction(rhs, &i)?;

    // Assign the intermediate results to a temporary variable, which we eventually assign to the
    // original left-hand side value. This ensures all operations are performed in registers,
    // rather than operating on data stored in global memory.
    let temp_id = Name::sym_str("t");
    let temp_var = Expr::Var {id: temp_id.clone(), ty: ty.clone(), i: i.clone()};
    let ne = find_neutral_element(&op, &ty, &i)?;
    let rhs = Expr::BinOp {
        lhs: Box::new(temp_var.clone()),
        op,
        rhs: Box::new(rhs),
        ty: ty.clone(),
        i: i.clone()
    };
    let inner_stmt = Stmt::Assign {
        dst: temp_var.clone(),
        expr: rhs, labels: vec![], i: i.clone()
    };
    let niters = dims.into_iter().map(|(n, _)| n as i128).product();
    let stmt = Stmt::For {
        var: reduce_id,
        lo: mk_int(0, scalar_sizes, &i),
        hi: mk_int(niters, scalar_sizes, &i),
        step: mk_int(1, scalar_sizes, &i),
        body: vec![inner_stmt],
        labels: l,
        i: i.clone()
    };
    let pre_stmt = Stmt::Definition {
        ty: ty.clone(), id: temp_id, expr: ne, labels: vec![], i: i.clone()
    };
    let post_stmt = match lhs {
        ReduceTargetType::Definition {e: Expr::Var {id, ty, ..}} => {
            Ok(Stmt::Definition {ty, id, expr: temp_var, labels: vec![], i: i.clone()})
        },
        ReduceTargetType::Assign {e} => {
            Ok(Stmt::Assign {dst: e, expr: temp_var, labels: vec![], i: i.clone()})
        },
        _ => py_internal_error!(i, "Invalid form of reduce target type")
    }?;
    Ok(Stmt::WithGpuContext {body: vec![pre_stmt, stmt, post_stmt], i: i.clone()})
}

fn generate_mapping_loops(
    lhs: Expr,
    rhs: Expr,
    mut labels: Vec<String>,
    i: Info,
    dims: Vec<(i64, Name)>,
    scalar_sizes: &ScalarSizes
) -> PyResult<Stmt> {
    let mut stmt = Stmt::Assign {dst: lhs, expr: rhs, labels: vec![], i: i.clone()};
    for (shape, id) in dims.into_iter().rev() {
        let for_label = labels.pop().map(|l| vec![l]).unwrap_or(vec![]);
        stmt = Stmt::For {
            var: id,
            lo: mk_int(0, scalar_sizes, &i),
            hi: mk_int(shape as i128, scalar_sizes, &i),
            step: mk_int(1, scalar_sizes, &i),
            body: vec![stmt],
            labels: for_label,
            i: i.clone()
        };
    }
    Ok(stmt)
}

fn replace_slices_assignment(
    reconstruct_stmt: impl Fn(Expr, Expr, Vec<String>, Info) -> Stmt,
    lhs: Expr,
    rhs: Expr,
    labels: Vec<String>,
    i: Info,
    scalar_sizes: &ScalarSizes,
    is_definition: bool
) -> PyResult<Stmt> {
    let lhs = constant_fold::fold_expr(lhs);
    let rhs = constant_fold::fold_expr(rhs);

    // If neither side of the assignment contains a slice, we reconstruct the original statement.
    // Otherwise, it is a slice statement, which is processed differently depending on whether it
    // performs a reduction or not.
    let lslices = count_slice_dims_expr(0, &lhs);
    let rslices = count_slice_dims_expr(0, &rhs);
    let ndims = usize::max(lslices, rslices);
    if ndims == 0 {
        Ok(reconstruct_stmt(lhs, rhs, labels, i))
    } else {
        // Extract a single shape containing each dimension of the slices involved in the slice
        // statement. Each slice is replaced by an identifier used to distinguish a particular
        // dimension it refers to.
        let shape = extract_shape(&lhs, &rhs, &i)?;
        let ids = shape.iter()
            .map(|_| Name::sym_str("slice_dim"))
            .collect::<Vec<Name>>();
        let lhs = insert_slice_dim_ids(&ids[..], lhs);
        let rhs = insert_slice_dim_ids(&ids[..], rhs);
        let dims = shape.into_iter()
            .zip(ids.into_iter())
            .collect::<Vec<(i64, Name)>>();
        if is_reduction(&rhs) {
            if lslices == 0 {
                // Reduction over all dimensions. The left-hand side must be a variable, and the right-hand
                // side uses one of the built-in reduction operations.
                let lhs = if is_definition {
                    ReduceTargetType::Definition {e: lhs}
                } else {
                    ReduceTargetType::Assign {e: lhs}
                };
                generate_reduction_loop(lhs, rhs, labels, i, dims, scalar_sizes)
            } else {
                py_runtime_error!(
                    i,
                    "A slice reduction produces a scalar result, while the \
                     left-hand side of the assignment expects a slice of arguments."
                )
            }
        } else if lslices >= rslices {
            // A mapping slice operation, which is repeated over all values included in the mentioned
            // slices. This also has to ensure all dimensions of slices add up relative to each other.
            generate_mapping_loops(lhs, rhs, labels, i, dims, scalar_sizes)
        } else {
            // The left-hand side contains fewer slice dimensions than the right hand side. This
            // form of slice statements are not supported as it corresponds to writing multiple
            // values to a single memory location (i.e., a form of reduction).
            py_runtime_error!(
                i,
                "Slice statements cannot have more slice dimensions in the \
                 right-hand side expression than in the left-hand side \
                 expression (found {rslices} in RHS and {lslices} in LHS)."
            )
        }
    }
}

fn replace_slices_stmt(s: Stmt, scalar_sizes: &ScalarSizes) -> PyResult<Stmt> {
    match s {
        Stmt::Definition {ty, id, expr, labels, i} => {
            let reconstruct_def = |lhs, rhs, labels, i| {
                if let Expr::Var {id, ty, ..} = lhs {
                    Stmt::Definition {ty, id, expr: rhs, labels, i}
                } else {
                    unreachable!()
                }
            };
            let lhs = Expr::Var {id, ty, i: i.clone()};
            replace_slices_assignment(reconstruct_def, lhs, expr, labels, i, scalar_sizes, true)
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let reconstruct_assign = |lhs, rhs, labels, i| {
                Stmt::Assign {dst: lhs, expr: rhs, labels, i}
            };
            replace_slices_assignment(reconstruct_assign, dst, expr, labels, i, scalar_sizes, false)
        },
        Stmt::For {..} | Stmt::While {..} | Stmt::If {..} | Stmt::Return {..} |
        Stmt::WithGpuContext {..} | Stmt::Expr {..} => {
            s.smap_result(|s| replace_slices_stmt(s, scalar_sizes))
        }
    }
}

fn replace_slices_fun_def(def: FunDef, scalar_sizes: &ScalarSizes) -> PyResult<FunDef> {
    let body = def.body.smap_result(|s| replace_slices_stmt(s, scalar_sizes))?;
    Ok(FunDef {body, ..def})
}

fn replace_slices_top(t: Top, scalar_sizes: &ScalarSizes) -> PyResult<Top> {
    match t {
        Top::CallbackDecl {..} | Top::ExtDecl {..} => Ok(t),
        Top::FunDef {v} => Ok(Top::FunDef {v: replace_slices_fun_def(v, scalar_sizes)?})
    }
}

fn ensure_no_remaining_slices_expr(e: Expr) -> PyResult<Expr> {
    match e {
        Expr::Slice {i, ..} => {
            py_runtime_error!(i, "Slices can only be used in an assignment statement.")
        },
        _ => e.smap_result(ensure_no_remaining_slices_expr)
    }
}

fn ensure_no_remaining_slices_stmt(s: Stmt) -> PyResult<Stmt> {
    s.smap_result(ensure_no_remaining_slices_stmt)?
        .smap_result(ensure_no_remaining_slices_expr)
}

fn ensure_no_remaining_slices_fun_def(def: FunDef) -> PyResult<FunDef> {
    let body = def.body.smap_result(ensure_no_remaining_slices_stmt)?;
    Ok(FunDef {body, ..def})
}

fn ensure_no_remaining_slices_top(t: Top) -> PyResult<Top> {
    match t {
        Top::CallbackDecl {..} | Top::ExtDecl {..} => Ok(t),
        Top::FunDef {v} => Ok(Top::FunDef {v: ensure_no_remaining_slices_fun_def(v)?})
    }
}

fn ensure_no_remaining_slices(ast: Ast) -> PyResult<Ast> {
    Ok(Ast {
        main: ensure_no_remaining_slices_fun_def(ast.main)?,
        tops: ast.tops.smap_result(ensure_no_remaining_slices_top)?
    })
}

pub fn apply(ast: Ast, scalar_sizes: &ScalarSizes) -> PyResult<Ast> {
    let tops = ast.tops.smap_result(|t| replace_slices_top(t, scalar_sizes))?;
    let main = replace_slices_fun_def(ast.main, scalar_sizes)?;
    ensure_no_remaining_slices(Ast {tops, main})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    #[test]
    fn count_slice_dims_scalar_index() {
        let target = var("x", tyuk());
        let index = int(0, Some(ElemSize::I64));
        let e = subscript(target, index, tyuk());
        assert_eq!(count_slice_dims_expr(0, &e), 0);
    }

    #[test]
    fn count_slice_dims_singleton_slice() {
        let target = var("x", tyuk());
        let index = slice(None, None);
        let e = subscript(target, index, tyuk());
        assert_eq!(count_slice_dims_expr(0, &e), 1);
    }

    #[test]
    fn count_slice_dims_add_slice_exprs() {
        let lhs = subscript(
            var("x", tyuk()),
            slice(None, None),
            tyuk()
        );
        let rhs = subscript(
            var("y", tyuk()),
            tuple(vec![slice(None, None), slice(None, None)]),
            tyuk()
        );
        let e = binop(lhs, BinOp::Add, rhs, tyuk());
        assert_eq!(count_slice_dims_expr(0, &e), 2);
    }

    #[test]
    fn unify_equal_shapes() {
        let sh = vec![10, 20];
        assert_eq!(unify_shapes(sh.clone(), sh.clone(), &i()).unwrap(), sh);
    }

    #[test]
    fn unify_shapes_broadcasting() {
        let lsh = vec![];
        let rsh = vec![10];
        assert_eq!(unify_shapes(lsh, rsh, &i()).unwrap(), vec![10]);
    }

    #[test]
    fn unify_shapes_broadcast_multidims() {
        let lsh = vec![10, 1, 30, 1];
        let rsh = vec![1, 20, 30, 40];
        assert_eq!(unify_shapes(lsh, rsh, &i()).unwrap(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn extract_tensor_shape_num() {
        let sh = vec![TensorShape::Num {n: 10}];
        assert_eq!(extract_tensor_shape(&sh, &i()).unwrap(), vec![10]);
    }

    #[test]
    fn extract_tensor_shape_symbol_fails() {
        let sh = vec![TensorShape::Symbol {id: id("x")}];
        assert_py_error_matches(extract_tensor_shape(&sh, &i()), "Found shape symbol");
    }

    #[test]
    fn extract_index_dims_scalar_index() {
        let idx = int(1, None);
        assert_eq!(extract_index_dims(vec![10], &idx).unwrap(), vec![0]);
    }

    #[test]
    fn extract_index_dims_scalar_and_slice_indices() {
        let idx = tuple(vec![slice(None, None), int(1, None)]);
        assert_eq!(extract_index_dims(vec![10, 20], &idx).unwrap(), vec![10, 0]);
    }

    #[test]
    fn extract_index_dims_partial_slice() {
        let idx = slice(Some(int(2, None)), None);
        assert_eq!(extract_index_dims(vec![10], &idx).unwrap(), vec![8]);
    }
}
