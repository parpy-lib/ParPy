use super::ast::*;
use super::constant_fold;
use super::inline_const;
use super::specialize;
use crate::py_internal_error;
use crate::py_type_error;
use crate::ext::buffer::DataType;
use crate::option::CompileOptions;
use crate::utils::ast::*;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;
use crate::utils::smap::*;

use itertools::Itertools;
use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct UnifyEnv {
    pub id: Name,
    pub shape_vars: BTreeMap<Name, i64>,
    pub type_vars: BTreeMap<Name, ElemSize>,
}

impl UnifyEnv {
    fn new(id: Option<&Name>) -> UnifyEnv {
        UnifyEnv {
            id: id.cloned().unwrap_or(Name::sym_str("")),
            shape_vars: BTreeMap::new(),
            type_vars: BTreeMap::new()
        }
    }

    pub fn lookup_shape_symbol(&self, id: &Name) -> Option<i64> {
        self.shape_vars.get(id).copied()
    }

    pub fn insert_shape_symbol(mut self, id: Name, n: i64) -> Self {
        self.shape_vars.insert(id, n);
        self
    }

    pub fn lookup_type_variable(&self, id: &Name) -> Option<ElemSize> {
        self.type_vars.get(id).cloned()
    }

    pub fn insert_type_variable(mut self, id: Name, sz: ElemSize) -> Self {
        self.type_vars.insert(id, sz);
        self
    }
}

fn unify_elem_size(l: ElemSize, r: ElemSize, i: &Info) -> PyResult<ElemSize> {
    match (&l, &r) {
        (ElemSize::Bool, ElemSize::Bool) => Ok(r),
        (ElemSize::I8, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I16, ElemSize::I8) => Ok(ElemSize::I16),
        (ElemSize::I16, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I32, ElemSize::I8 | ElemSize::I16) => Ok(ElemSize::I32),
        (ElemSize::I32, _) if r.is_signed_integer() => Ok(r),
        (ElemSize::I64, _) if r.is_signed_integer() => Ok(l),
        (ElemSize::U8, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U16, ElemSize::U8) => Ok(ElemSize::U16),
        (ElemSize::U16, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U32, ElemSize::U8 | ElemSize::U16) => Ok(ElemSize::U32),
        (ElemSize::U32, _) if r.is_unsigned_integer() => Ok(r),
        (ElemSize::U64, _) if r.is_unsigned_integer() => Ok(l),
        (ElemSize::F16, _) if r.is_floating_point() => Ok(r),
        (ElemSize::F32, ElemSize::F16) => Ok(ElemSize::F32),
        (ElemSize::F32, _) if r.is_floating_point() => Ok(r),
        (ElemSize::F64, _) if r.is_floating_point() => Ok(l),
        _ => py_type_error!(i, "Incompatible element types {l} and {r}")
    }
}

fn unify_tensor_elem_size(
    env: UnifyEnv,
    l: TensorElemSize,
    r: TensorElemSize,
    i: &Info,
    allow_coercion: bool
) -> PyResult<(UnifyEnv, TensorElemSize)> {
    match (l, r) {
        (TensorElemSize::Fixed {sz: lsz}, TensorElemSize::Fixed {sz: rsz}) => {
            let sz = if allow_coercion {
                unify_elem_size(lsz.clone(), rsz.clone(), &i)
            } else if lsz == rsz {
                Ok(lsz)
            } else {
                py_type_error!(i, "Failed to unify element types {lsz} and {rsz} \
                                   (no coercion).")
            }?;
            Ok((env, TensorElemSize::Fixed {sz}))
        },
        (TensorElemSize::Fixed {sz}, TensorElemSize::Variable {id}) |
        (TensorElemSize::Variable {id}, TensorElemSize::Fixed {sz}) => {
            match env.lookup_type_variable(&id) {
                Some(var_sz) => {
                    if allow_coercion {
                        match unify_elem_size(sz.clone(), var_sz.clone(), &i) {
                            Ok(res_sz) => Ok((env, TensorElemSize::Fixed {sz: res_sz})),
                            Err(_) => py_type_error!(i, "Failed to unify type variable \
                                                         {id} = {var_sz} with {sz}.")
                        }
                    } else if sz == var_sz {
                        Ok((env, TensorElemSize::Fixed {sz}))
                    } else {
                        py_type_error!(i, "Failed to unify type variable \
                                           {id} = {var_sz} with {sz} (no coercion).")
                    }
                }
                None => {
                    let env = env.insert_type_variable(id, sz.clone());
                    Ok((env, TensorElemSize::Fixed {sz}))
                }
            }
        },
        (TensorElemSize::Variable {id: lid}, TensorElemSize::Variable {id: rid}) => {
            match (env.lookup_type_variable(&lid), env.lookup_type_variable(&rid)) {
                (Some(l), Some(r)) => {
                    if allow_coercion {
                        match unify_elem_size(l.clone(), r.clone(), &i) {
                            Ok(sz) => Ok((env, TensorElemSize::Fixed {sz})),
                            Err(_) => py_type_error!(i, "Failed to unify type variables \
                                                         {lid} = {l} and {rid} = {r}.")
                        }
                    } else if l == r {
                        Ok((env, TensorElemSize::Fixed {sz: l}))
                    } else {
                        py_type_error!(i, "Failed to unify type variables \
                                           {lid} = {l} and {rid} = {r} (no coercion).")
                    }
                },
                (None, Some(sz)) => {
                    let env = env.insert_type_variable(lid, sz.clone());
                    Ok((env, TensorElemSize::Fixed {sz}))
                },
                (Some(sz), None) => {
                    let env = env.insert_type_variable(rid, sz.clone());
                    Ok((env, TensorElemSize::Fixed {sz}))
                },
                (None, None) => {
                    py_internal_error!(i, "Failed to unify unbound type variables \
                                           {lid} and {rid}.")
                }
            }
        }
    }
}

pub fn eq_tensor_elem_size(
    env: UnifyEnv,
    l: &TensorElemSize,
    r: &TensorElemSize
) -> Option<(UnifyEnv, TensorElemSize)> {
    if let Ok(res) = unify_tensor_elem_size(env, l.clone(), r.clone(), &Info::default(), false) {
        Some(res)
    } else {
        None
    }
}

fn unify_shape(
    env: UnifyEnv,
    lsh: TensorShape,
    rsh: TensorShape,
    i: &Info
) -> PyResult<(UnifyEnv, TensorShape)> {
    match (lsh, rsh) {
        (TensorShape::Num {n: ln}, TensorShape::Num {n: rn}) => {
            if ln == rn {
                Ok((env, TensorShape::Num {n: ln}))
            } else {
                py_type_error!(i, "Failed to unify distinct dimensions {ln} and {rn}.")
            }
        },
        (TensorShape::Num {n}, TensorShape::Symbol {id}) |
        (TensorShape::Symbol {id}, TensorShape::Num {n}) => {
            match env.lookup_shape_symbol(&id) {
                Some(m) if n == m => Ok((env, TensorShape::Num {n})),
                Some(m) => py_type_error!(i, "Failed to unify shape variable \
                                              {id} = {m} with {n}."),
                None => {
                    let env = env.insert_shape_symbol(id, n);
                    Ok((env, TensorShape::Num {n}))
                }
            }
        },
        (TensorShape::Symbol {id: lid}, TensorShape::Symbol {id: rid}) => {
            match (env.lookup_shape_symbol(&lid), env.lookup_shape_symbol(&rid)) {
                (Some(l), Some(r)) if l == r => Ok((env, TensorShape::Num {n: l})),
                (Some(l), Some(r)) => {
                    py_type_error!(i, "Failed to unify shape variables \
                                       {lid} = {l} and {rid} = {r}.")
                },
                (None, Some(n)) => {
                    let env = env.insert_shape_symbol(lid, n);
                    Ok((env, TensorShape::Num {n}))
                },
                (Some(n), None) => {
                    let env = env.insert_shape_symbol(rid, n);
                    Ok((env, TensorShape::Num {n}))
                },
                (None, None) => {
                    py_internal_error!(i, "Failed to unify unresolved shape \
                                           symbols {lid} and {rid}.")
                }
            }
        }
    }
}

fn unify_shapes(
    env: UnifyEnv,
    lshapes: Vec<TensorShape>,
    rshapes: Vec<TensorShape>,
    i: &Info
) -> PyResult<(UnifyEnv, Vec<TensorShape>)> {
    if lshapes.len() == rshapes.len() {
        lshapes.into_iter()
            .zip(rshapes.into_iter())
            .fold(Ok((env, vec![])), |acc, (lsh, rsh)| {
                let (env, mut shape) = acc?;
                let (env, sh) = unify_shape(env, lsh, rsh, i)?;
                shape.push(sh);
                Ok((env, shape))
            })
    } else {
        let ls = lshapes.iter().join(", ");
        let rs = rshapes.iter().join(", ");
        py_type_error!(i, "Found incompatible tensor shapes [{ls}] and [{rs}]")
    }
}

fn unify_types(
    env: UnifyEnv,
    lty: Type,
    rty: Type,
    i: &Info
) -> PyResult<(UnifyEnv, Type)> {
    match (lty, rty) {
        (Type::Unknown, Type::Unknown) => {
            py_internal_error!(i, "Unknown types cannot be unified")
        },
        (Type::Unknown, ty) | (ty, Type::Unknown) => Ok((env, ty)),
        (Type::String, Type::String) => Ok((env, Type::String)),
        ( Type::Tensor {sz: lsz, shape: lshape}
        , Type::Tensor {sz: rsz, shape: rshape} ) => {
            if lshape.is_empty() && rshape.is_empty() {
                let (env, sz) = unify_tensor_elem_size(env, lsz, rsz, i, true)?;
                Ok((env, Type::Tensor {sz, shape: vec![]}))
            } else {
                match eq_tensor_elem_size(env, &lsz, &rsz) {
                    Some((env, sz)) => {
                        let (env, shape) = unify_shapes(env, lshape, rshape, i)?;
                        Ok((env, Type::Tensor {sz, shape}))
                    },
                    None => {
                        py_type_error!(i, "Failed to unify non-scalar tensors \
                                           containing distinct element sizes \
                                           {lsz} and {rsz}.")
                    }
                }
            }
        },
        (Type::Tuple {elems: lelems}, Type::Tuple {elems: relems}) => {
            if lelems.len() == relems.len() {
                let (env, elem_types) = lelems.into_iter()
                    .zip(relems.into_iter())
                    .fold(Ok((env, vec![])), |acc: PyResult<_>, (lty, rty)| {
                        let (env, mut types) = acc?;
                        let (env, ty) = unify_types(env, lty, rty, i)?;
                        types.push(ty);
                        Ok((env, types))
                    })?;
                Ok((env, Type::Tuple {elems: elem_types}))
            } else {
                py_type_error!(i, "Failed to unify tuple types of different lengths")
            }
        },
        (Type::Dict {fields: lfields}, Type::Dict {fields: rfields}) => {
            if lfields.len() == rfields.len() {
                let (env, field_types) = lfields.into_iter()
                    .zip(rfields.into_iter())
                    .fold(Ok((env, BTreeMap::new())), |acc, (lfield, rfield)| {
                        let (env, mut fields) = acc?;
                        let (lk, lv) = lfield;
                        let (rk, rv) = rfield;
                        if lk == rk {
                            let (env, ty) = unify_types(env, lv, rv, i)?;
                            fields.insert(lk, ty);
                            Ok((env, fields))
                        } else {
                            py_type_error!(i, "Failed to unify dictionary types with distinct keys")
                        }
                    })?;
                Ok((env, Type::Dict {fields: field_types}))
            } else {
                py_type_error!(i, "Failed to unify dictionary types of \
                                   different number of entries")
            }
        },
        (Type::Void, Type::Void) => Ok((env, Type::Void)),
        (l, r) => py_type_error!(i, "Failed to unify incompatible types {l} and {r}")
    }
}

fn unify_parameter_type(
    acc: PyResult<(UnifyEnv, Vec<Param>)>,
    param: Param,
    arg_type: Type
) -> PyResult<(UnifyEnv, Vec<Param>)> {
    let (env, mut params) = acc?;
    let Param {id, ty, i} = param;
    let (env, ty) = match unify_types(env.clone(), ty.clone(), arg_type.clone(), &i) {
        Ok((env, ty)) => Ok((env, ty)),
        Err(_) => {
            let where_str = if env.shape_vars.is_empty() && env.type_vars.is_empty() {
                "".to_string()
            } else {
                let shapes = env.shape_vars.into_iter()
                    .map(|(id, n)| {
                        format!("  {} = {n}", TensorShape::Symbol {id}.pprint_default())
                    });
                let types = env.type_vars.into_iter()
                    .map(|(id, sz)| {
                        let id = TensorElemSize::Variable {id}.pprint_default();
                        format!("  {id} = {sz}")
                    });
                let vars = shapes.chain(types).join("\n");
                format!("\nWhere:\n{vars}")
            };
            py_type_error!(i, "Parameter {id} was annotated with type {ty} \
                               which is incompatible with argument type {arg_type}.\
                               {where_str}")
        }
    }?;
    params.push(Param {id, ty, i});
    Ok((env, params))
}

fn unify_parameter_types(
    params: Vec<Param>,
    args: &Vec<Expr>,
    id: &Name,
    i: &Info
) -> PyResult<(UnifyEnv, Vec<Param>)> {
    let pn = params.len();
    let an = args.len();
    if pn == an {
        let arg_types = args.iter()
            .map(|arg| arg.get_type().clone())
            .collect::<Vec<Type>>();
        params.into_iter()
            .zip(arg_types.into_iter())
            .fold(Ok((UnifyEnv::new(Some(id)), vec![])), |acc, (param, arg_type)| {
                unify_parameter_type(acc, param, arg_type)
            })
    } else {
        py_type_error!(i, "Function {id} expects {pn} parameters, but was \
                           called with {an} arguments.")
    }
}

#[derive(Debug)]
struct TypeCheckState {
    // Maps the name of a variable to its type.
    vars: BTreeMap<Name, Type>,
}

impl TypeCheckState {
    fn new(params: Vec<Param>) -> Self {
        let vars = params.into_iter()
            .map(|Param {id, ty, ..}| (id, ty))
            .collect::<BTreeMap<Name, Type>>();
        TypeCheckState {vars}
    }
}

#[derive(Debug)]
struct TypeCheckEnv<'py> {
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,

    // The options passed to the compiler.
    opts: CompileOptions,

    // The default sizes to use for int and float literals.
    scalar_sizes: ScalarSizes,

    // Maps a unification environment to the specialized function.
    specs: BTreeMap<UnifyEnv, FunDef>,

    // Contains the names of the specialized functions in the order in which they should be
    // inserted, where the first entry should be placed at the top.
    spec_list: Vec<Top>,

    // A stack of states of type-checking. Each state in the stack corresponds to the intermediate
    // results of a (specialized) function. When a function call node is encountered, we invoke it
    // recursively and insert a new entry to this stack, so that the state of the current function
    // is kept.
    state_stack: Vec<TypeCheckState>
}

impl<'py> TypeCheckEnv<'py> {
    fn new(tops: BTreeMap<String, Bound<'py, PyCapsule>>, opts: CompileOptions) -> Self {
        let scalar_sizes = ScalarSizes::from_opts(&opts);
        TypeCheckEnv {
            tops,
            opts,
            scalar_sizes,
            specs: BTreeMap::new(),
            spec_list: vec![],
            state_stack: vec![]
        }
    }

    fn enter_function(mut self, params: Vec<Param>) -> Self {
        let new_state = TypeCheckState::new(params);
        self.state_stack.push(new_state);
        self
    }

    fn exit_function(mut self, unify_env: UnifyEnv, def: &FunDef) -> PyResult<Self> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment attempted to \
                                                 pop with empty stack stack.")
        } else {
            self.state_stack.pop().unwrap();
            self.specs.insert(unify_env, def.clone());
            Ok(self)
        }
    }

    fn lookup_top(&self, id: &Name) -> Option<Top> {
        self.tops.get(id.get_str())
            .map(|cap| unsafe { cap.reference::<Top>() }.clone())
    }

    fn lookup_var(&self, id: &Name) -> PyResult<Option<Type>> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment variable lookup \
                                                 with empty state stack.")
        } else {
            let curr_state = self.state_stack.last().unwrap();
            Ok(curr_state.vars.get(&id).cloned())
        }
    }

    fn insert_var(mut self, id: &Name, ty: &Type) -> PyResult<Self> {
        if self.state_stack.is_empty() {
            py_internal_error!(Info::default(), "Type environment variable insertion \
                                                 with empty state stack.")
        } else {
            let curr_state = self.state_stack.last_mut().unwrap();
            curr_state.vars.insert(id.clone(), ty.clone());
            Ok(self)
        }
    }
}

type TypeCheckResult<'py, T> = PyResult<(TypeCheckEnv<'py>, T)>;

fn extract_type<'py>(
    arg: &Bound<'py, PyAny>,
    scalar_sizes: &ScalarSizes,
    i: &Info
) -> PyResult<Type> {
    let py = arg.py();
    let parpy = py.import("parpy")?;
    let buffer = parpy.getattr("buffer")?;
    let ty = arg.get_type();
    if arg.is_instance(&buffer.getattr("Buffer")?)? {
        let dtype = arg.getattr("dtype")?.extract::<DataType>()?;
        let sz = TensorElemSize::Fixed {sz: dtype.sz};
        let shape = arg.getattr("shape")?
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|n| TensorShape::Num {n})
            .collect::<Vec<TensorShape>>();
        Ok(Type::Tensor {sz, shape})
    } else if arg.is_instance(&PyBool::type_object(py))? {
        Ok(Type::fixed_scalar(ElemSize::Bool))
    } else if arg.is_instance(&PyInt::type_object(py))? {
        Ok(Type::fixed_scalar(scalar_sizes.int.clone()))
    } else if arg.is_instance(&PyFloat::type_object(py))? {
        Ok(Type::fixed_scalar(scalar_sizes.float.clone()))
    } else if arg.is_instance(&PyDict::type_object(py))? {
        let fields = arg.downcast::<PyDict>()?
            .items()
            .into_iter()
            .map(|kv| {
                let (k, v) = kv.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
                let k = k.extract::<String>()?;
                let ty = extract_type(&v, scalar_sizes, i)?;
                Ok((k, ty))
            })
            .collect::<PyResult<BTreeMap<String, Type>>>()?;
        Ok(Type::Dict {fields})
    } else {
        py_type_error!(i, "Argument has unsupported type {ty}")
    }
}

fn extract_argument_types<'py>(
    args: &Vec<Bound<'py, PyAny>>,
    scalar_sizes: &ScalarSizes,
    i: &Info
) -> PyResult<Vec<Type>> {
    args.iter()
        .map(|arg| extract_type(arg, scalar_sizes, i))
        .collect::<PyResult<Vec<Type>>>()
}

fn coerce_type(e: Expr, expected_ty: &Type) -> PyResult<Expr> {
    if e.get_type().eq(expected_ty) {
        Ok(e)
    } else {
        let i = e.get_info();
        let actual = e.get_type();
        match (actual, expected_ty) {
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                if lsh.is_empty() && rsh.is_empty() {
                    // If the two element sizes are considered equal, a coercion is not needed.
                    // Otherwise, if the types can be unified by coercing either element size, then
                    // we insert an explicit type conversion.
                    let unify_env = UnifyEnv::new(None);
                    if let Some(_) = eq_tensor_elem_size(unify_env.clone(), &lsz, &rsz) {
                        Ok(e)
                    } else {
                        match unify_tensor_elem_size(unify_env, lsz.clone(), rsz.clone(), &i, true) {
                            Ok(_) => {
                                let ty = Type::Tensor {sz: rsz.clone(), shape: vec![]};
                                Ok(Expr::Convert {e: Box::new(e), ty, i})
                            },
                            Err(_) => py_type_error!(i, "Cannot coerce element size {lsz} to {rsz}.")
                        }
                    }
                } else {
                    let unify_env = UnifyEnv::new(None);
                    if let Some((unify_env, _)) = eq_tensor_elem_size(unify_env, &lsz, &rsz) {
                        // If the shapes of the two sides can be unified, that means each shape symbol
                        // can be coerced to a unique value.
                        let _ = unify_shapes(unify_env, lsh.clone(), rsh.clone(), &i)?;
                        Ok(e)
                    } else {
                        py_type_error!(i, "Cannot coerce non-empty tensor of element size {lsz} to {rsz}.")
                    }
                }
            },
            (Type::Tuple {..}, Type::Tuple {elems: r}) => {
                if let Expr::Tuple {elems, i, ..} = e {
                    let elems = elems.into_iter()
                        .zip(r.iter())
                        .map(|(e, ty)| coerce_type(e, ty))
                        .collect::<PyResult<Vec<Expr>>>()?;
                    let elem_tys = elems.iter()
                        .map(|e| e.get_type().clone())
                        .collect::<Vec<Type>>();
                    let ty = Type::Tuple {elems: elem_tys};
                    Ok(Expr::Tuple {elems, ty, i})
                } else {
                    py_type_error!(i, "Cannot coerce non-literal tuple value {e}")
                }
            },
            _ => py_type_error!(i, "Coercion of expression {e} to type {expected_ty} failed")
        }
    }
}

fn extract_return_type_stmt(
    ret_ty: Type,
    s: &Stmt
) -> PyResult<Type> {
    match s {
        Stmt::Return {value, i} => {
            let ty = value.get_type();
            match ret_ty {
                Type::Void => Ok(ty.clone()),
                actual_ty => {
                    let env = UnifyEnv::new(None);
                    match unify_types(env, actual_ty.clone(), ty.clone(), &i) {
                        Ok((_, r)) => Ok(r),
                        Err(_) => py_type_error!(i, "Found incompatible return types \
                                                     {ty} and {actual_ty}.")
                    }
                }
            }
        },
        _ => s.sfold_result(Ok(ret_ty), |ty, s| extract_return_type_stmt(ty, s))
    }
}

fn extract_return_type(body: &Vec<Stmt>) -> PyResult<Type> {
    body.sfold_result(Ok(Type::Void), extract_return_type_stmt)
}

trait TypeCheck {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>,
    ) -> TypeCheckResult<'py, Self> where Self: Sized;
}

impl<T: Clone + TypeCheck + SMapAccum<T>> TypeCheck for Vec<T> {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Self> {
        self.smap_accum_l_result(Ok(env), |env, t| t.type_check(env))
    }
}

// Ensure that the provided expression is a valid condition. For an integer conditions, we
// automatically insert an inequality comparison with zero, to convert them to a proper boolean
// condition. Other types of conditions result in an error.
fn ensure_conditional_type(cond: Expr) -> PyResult<Expr> {
    let i = cond.get_info();
    let cond_ty = cond.get_type().clone();
    if cond_ty.is_bool_scalar() {
        Ok(cond)
    } else if cond_ty.is_int_scalar() {
        Ok(Expr::BinOp {
            lhs: Box::new(cond),
            op: BinOp::Neq,
            rhs: Box::new(Expr::Int {v: 0, ty: cond_ty.clone(), i: i.clone()}),
            ty: Type::fixed_scalar(ElemSize::Bool),
            i
        })
    } else {
        py_type_error!(i, "Conditions must be boolean or integer values, found {cond_ty}")
    }
}

fn type_check_unop(
    op: &UnOp,
    arg: &Expr,
    i: &Info
) -> PyResult<Type> {
    let ty = arg.get_type();
    match op {
        UnOp::Sub if ty.is_int_scalar() || ty.is_float_scalar() => Ok(ty.clone()),
        UnOp::Not if ty.is_bool_scalar() => Ok(ty.clone()),
        UnOp::BitNeg if ty.is_int_scalar() => Ok(ty.clone()),
        _ => py_type_error!(i, "Unsupported argument type {ty} of unary operator {op:?}")
    }
}

fn type_check_binop(
    lhs: Expr,
    op: &BinOp,
    rhs: Expr,
    i: &Info
) -> PyResult<(Box<Expr>, Type, Box<Expr>)> {
    let lty = lhs.get_type().clone();
    let rty = rhs.get_type().clone();
    let (_, ty) = unify_types(UnifyEnv::new(None), lty, rty, i)?;
    let lhs = coerce_type(lhs, &ty)?;
    let rhs = coerce_type(rhs, &ty)?;
    let ty = match op {
        BinOp::Add | BinOp::Sub |
        BinOp::Mul if ty.is_int_scalar() || ty.is_float_scalar() => {
            Ok(ty)
        },
        BinOp::Div if ty.is_float_scalar() => Ok(ty),
        BinOp::FloorDiv | BinOp::Rem if ty.is_int_scalar() => Ok(ty),
        BinOp::Pow if ty.is_float_scalar() => Ok(ty),
        BinOp::And | BinOp::Or if ty.is_bool_scalar() => Ok(ty),
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor |
        BinOp::BitShl | BinOp::BitShr if ty.is_int_scalar() => Ok(ty),
        BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq |
        BinOp::Lt | BinOp::Gt if ty.is_scalar() => {
            Ok(Type::fixed_scalar(ElemSize::Bool))
        },
        BinOp::Max | BinOp::Min if ty.is_int_scalar() || ty.is_float_scalar() => {
            Ok(ty)
        },
        _ => py_type_error!(i, "Unsupported type {ty} of binary operator {op:?}")
    }?;
    Ok((Box::new(lhs), ty, Box::new(rhs)))
}

fn type_check_dict_indexing(
    target: Expr,
    key: String,
    str_info: Info,
    target_info: Info
) -> PyResult<Expr> {
    if let Type::Dict {fields} = target.get_type() {
        if let Some(ty) = fields.get(&key) {
            let result_ty = ty.clone();
            Ok(Expr::Subscript {
                target: Box::new(target),
                idx: Box::new(Expr::String {v: key, ty: Type::String, i: str_info}),
                ty: result_ty,
                i: target_info
            })
        } else {
            py_type_error!(
                target_info, 
                "Field {key} not found in type {0}",
                target.get_type()
            )
        }
    } else {
        py_type_error!(target_info, "Strings can only be used as keys on \
                                     non-dictionary targets")
    }
}

fn count_dimensions(idx: &Expr) -> usize {
    match idx {
        Expr::Tuple {elems, ..} => elems.len(),
        _ => 1
    }
}

fn contains_slices(acc: bool, e: &Expr) -> bool {
    match e {
        Expr::Slice {..} => true,
        _ => e.sfold(acc, contains_slices)
    }
}

fn type_check_tensor_indexing<'py>(
    env: TypeCheckEnv<'py>,
    target: Expr,
    idx: Expr,
    i: Info
) -> TypeCheckResult<'py, Expr> {
    // Add coercion of each component index of the index expression.
    let int_ty = Type::fixed_scalar(env.scalar_sizes.int.clone());
    let expected_index_type = match idx.get_type() {
        Type::Tensor {shape, ..} if shape.len() == 0 => Ok(int_ty),
        Type::Tuple {elems} => {
            let expected_elem_types = elems.into_iter()
                .map(|_| int_ty.clone())
                .collect::<Vec<Type>>();
            Ok(Type::Tuple {elems: expected_elem_types})
        },
        ty => py_type_error!(i, "Found index of unsupported type {ty}")
    }?;
    let idx = coerce_type(idx, &expected_index_type)?;

    // Determine the result type based on the shape of the index and the target.
    let target_ty = target.get_type();
    let ty = match target_ty {
        Type::Tensor {sz, shape} => {
            let ndims = count_dimensions(&idx);
            if contains_slices(false, &idx) {
                if ndims == shape.len() {
                    Ok(Type::Tensor {sz: sz.clone(), shape: vec![]})
                } else {
                    let sh = shape.iter().join(", ");
                    py_type_error!(i, "Slices must address all dimensions of \
                                       the target.\n Index refers to {ndims} \
                                       dimensions, while the target has shape [{sh}].")
                }
            } else if ndims <= shape.len() {
                let shape = shape.clone()
                    .into_iter()
                    .skip(ndims)
                    .collect::<Vec<TensorShape>>();
                Ok(Type::Tensor {sz: sz.clone(), shape})
            } else {
                py_type_error!(i, "Indexing with {ndims} dimensions on tensor \
                                   of type {target_ty}.")
            }
        },
        _ => py_type_error!(i, "Indexing into unsupported target of type {target_ty}")
    }?;
    Ok((env, Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i}))
}

impl TypeCheck for Expr {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Expr> {
        match self {
            Expr::Var {id, ty: _, i} => {
                let ty = match env.lookup_var(&id)? {
                    Some(ty) => Ok(ty.clone()),
                    None => py_type_error!(i, "Variable {id} has unknown type")
                }?;
                Ok((env, Expr::Var {id, ty, i}))
            },
            Expr::String {v, ty: _, i} => {
                Ok((env, Expr::String {v, ty: Type::String, i}))
            },
            Expr::Bool {v, ty: _, i} => {
                let ty = Type::fixed_scalar(ElemSize::Bool);
                Ok((env, Expr::Bool {v, ty, i}))
            },
            Expr::Int {v, ty: _, i} => {
                let ty = Type::fixed_scalar(env.scalar_sizes.int.clone());
                Ok((env, Expr::Int {v, ty, i}))
            },
            Expr::Float {v, ty: _, i} => {
                let ty = Type::fixed_scalar(env.scalar_sizes.float.clone());
                Ok((env, Expr::Float {v, ty, i}))
            },
            Expr::UnOp {op, arg, ty: _, i} => {
                let (env, arg) = arg.type_check(env)?;
                let ty = type_check_unop(&op, &arg, &i)?;
                Ok((env, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::BinOp {lhs, op, rhs, ty: _, i} => {
                let (env, lhs) = lhs.type_check(env)?;
                let (env, rhs) = rhs.type_check(env)?;
                let (lhs, ty, rhs) = type_check_binop(lhs, &op, rhs, &i)?;
                Ok((env, Expr::BinOp {lhs, op, rhs, ty, i}))
            },
            Expr::ReduceOp {op, arg, ty: _, i} => {
                let (env, arg) = arg.type_check(env)?;
                let arg_ty = arg.get_type();
                if arg_ty.is_int_scalar() || arg_ty.is_float_scalar() {
                    let ty = arg_ty.clone();
                    Ok((env, Expr::ReduceOp {op, arg: Box::new(arg), ty, i}))
                } else {
                    py_type_error!(i, "Reduction operations are only supported \
                                       on integer and float types.")
                }
            },
            Expr::IfExpr {cond, thn, els, ty: _, i} => {
                let (env, cond) = cond.type_check(env)?;
                let cond = ensure_conditional_type(cond)?;
                let (env, thn) = thn.type_check(env)?;
                let (env, els) = els.type_check(env)?;
                let thn_type = thn.get_type().clone();
                let els_type = els.get_type().clone();
                let (_, ty) = unify_types(UnifyEnv::new(None), thn_type, els_type, &i)?;
                let thn = Box::new(coerce_type(thn, &ty)?);
                let els = Box::new(coerce_type(els, &ty)?);
                Ok((env, Expr::IfExpr {cond: Box::new(cond), thn, els, ty, i}))
            },
            Expr::Subscript {target, idx, ty: _, i} => {
                let (env, target) = target.type_check(env)?;
                let (env, idx) = idx.type_check(env)?;
                match idx {
                    Expr::String {v, ty: _, i: str_info} => {
                        Ok((env, type_check_dict_indexing(target, v, str_info, i)?))
                    },
                    idx => type_check_tensor_indexing(env, target, idx, i)
                }
            },
            Expr::Slice {lo, hi, ty: _, i} => {
                let type_check_boxed = |env, o: Option<Box<Expr>>| match o {
                    Some(e) => {
                        let (env, e) = e.type_check(env)?;
                        Ok::<_, PyErr>((env, Some(Box::new(e))))
                    },
                    None => Ok((env, None))
                };
                let (env, lo) = type_check_boxed(env, lo)?;
                let (env, hi) = type_check_boxed(env, hi)?;
                // NOTE(larshum, 2025-09-12): Slices are considered as integers. Assuming they are
                // never used outside of indexing (we cannot parse such Python code), this is fine.
                let ty = Type::fixed_scalar(env.scalar_sizes.int.clone());
                Ok((env, Expr::Slice {lo, hi, ty, i}))
            },
            Expr::Tuple {elems, ty: _, i} => {
                let (env, elems) = elems.type_check(env)?;
                let elem_types = elems.iter()
                    .map(|e| e.get_type().clone())
                    .collect::<Vec<Type>>();
                let ty = Type::Tuple {elems: elem_types};
                Ok((env, Expr::Tuple {elems, ty, i}))
            },
            Expr::List {i, ..} => {
                py_internal_error!(i, "Found list node in the type-checker")
            },
            Expr::Call {id, args, ty: _, i} => {
                let (env, args) = args.type_check(env)?;
                let (env, new_id, ty, args) = type_check_call(env, &id, args, &i)?;
                Ok((env, Expr::Call {id: new_id, args, ty, i}))
            },
            Expr::Callback {i, ..} => {
                py_internal_error!(i, "Found callback node in the type-checker")
            },
            Expr::Convert {e, ty, i} => {
                let (env, e) = e.type_check(env)?;
                Ok((env, Expr::Convert {e: Box::new(e), ty, i}))
            },
            Expr::GpuContext {..} | Expr::Label {..} => Ok((env, self)),
            Expr::Inline {e, ty: _, i} => {
                let (env, e) = e.type_check(env)?;
                let ty = e.get_type().clone();
                Ok((env, Expr::Inline {e: Box::new(e), ty, i}))
            },
            Expr::StaticBackendEq {backend, ty: _, i} => {
                let ty = Type::fixed_scalar(ElemSize::Bool);
                Ok((env, Expr::StaticBackendEq {backend, ty, i}))
            },
            Expr::StaticTypesEq {lhs, rhs, ty: _, i} => {
                let ty = Type::fixed_scalar(ElemSize::Bool);
                Ok((env, Expr::StaticTypesEq {lhs, rhs, ty, i}))
            },
            Expr::StaticFail {msg, ty: _, i} => {
                py_internal_error!(i, "Type-checker reached fail node: {msg}")
            },
            Expr::AllocShared {shape, sz, ty: _, i} => {
                let (env, shape) = shape.type_check(env)?;
                let sh = shape.clone()
                    .into_iter()
                    .map(|e| {
                        // NOTE(larshum, 2025-12-02): We apply constant folding to allow using
                        // constant offsets in the size (for avoiding bank conflicts).
                        let e = constant_fold::fold_expr(e);
                        match e {
                            Expr::Int {v, ..} => Ok(TensorShape::Num {n: v as i64}),
                            _ => py_type_error!(i, "Found statically unknown shape in shared memory allocation: {e}"),
                        }
                    })
                    .collect::<PyResult<Vec<TensorShape>>>()?;
                let sz = match sz {
                    TensorElemSize::Fixed {..} => Ok(sz),
                    TensorElemSize::Variable {..} => {
                        py_internal_error!(i, "Found unresolved type variable")
                    }
                }?;
                let ty = Type::Tensor {shape: sh, sz: sz.clone()};
                Ok((env, Expr::AllocShared {shape, sz, ty, i}))
            },
        }
    }
}

impl TypeCheck for Stmt {
    fn type_check<'py>(
        self,
        env: TypeCheckEnv<'py>
    ) -> TypeCheckResult<'py, Stmt> {
        match self {
            Stmt::Definition {ty: _, id, expr, labels, i} => {
                let (env, expr) = expr.type_check(env)?;
                let ty = expr.get_type().clone();
                let env = env.insert_var(&id, &ty)?;
                Ok((env, Stmt::Definition {ty, id, expr, labels, i}))
            },
            Stmt::Assign {dst, expr, labels, i} => {
                let (env, dst) = dst.type_check(env)?;
                let (env, expr) = expr.type_check(env)?;
                let expr = coerce_type(expr, dst.get_type())?;
                Ok((env, Stmt::Assign {dst, expr, labels, i}))
            },
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let (env, lo) = lo.type_check(env)?;
                let (env, hi) = hi.type_check(env)?;
                let (env, step) = step.type_check(env)?;
                let int_ty = Type::fixed_scalar(env.scalar_sizes.int.clone());
                let lo = coerce_type(lo, &int_ty)?;
                let hi = coerce_type(hi, &int_ty)?;
                let step = coerce_type(step, &int_ty)?;
                let env = env.insert_var(&var, &int_ty)?;
                let (env, body) = body.type_check(env)?;
                Ok((env, Stmt::For {var, lo, hi, step, body, labels, i}))
            },
            Stmt::While {cond, body, i} => {
                let (env, cond) = cond.type_check(env)?;
                let cond = ensure_conditional_type(cond)?;
                let (env, body) = body.type_check(env)?;
                Ok((env, Stmt::While {cond, body, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (env, cond) = cond.type_check(env)?;
                let cond = ensure_conditional_type(cond)?;
                let (env, thn) = thn.type_check(env)?;
                let (env, els) = els.type_check(env)?;
                Ok((env, Stmt::If {cond, thn, els, i}))
            },
            Stmt::Return {value, i} => {
                let (env, value) = value.type_check(env)?;
                Ok((env, Stmt::Return {value, i}))
            },
            Stmt::WithGpuContext {body, i} => {
                let (env, body) = body.type_check(env)?;
                Ok((env, Stmt::WithGpuContext {body, i}))
            },
            Stmt::Expr {e, i} => {
                let (env, e) = e.type_check(env)?;
                match e.get_type() {
                    Type::Void => Ok((env, Stmt::Expr {e, i})),
                    ty => py_type_error!(i, "Expression statement cannot produce \
                                             a result (found result type {ty})")
                }
            },
        }
    }
}

fn any_parameter_is_annotated(params: &Vec<Param>) -> bool {
    params.iter().any(|Param {ty, ..}| *ty != Type::Unknown)
}

fn extract_annotated_params(t: &Top) -> Option<Vec<Param>> {
    let params = match t {
        Top::CallbackDecl {params, ..} => params,
        Top::ExtDecl {params, ..} => params,
        Top::FunDef {v: FunDef {params, ..}} => params,
    };
    if any_parameter_is_annotated(&params) {
        Some(params.clone())
    } else {
        None
    }
}

fn type_check_argument(
    arg: Expr,
    id: Name,
    ty: Type,
    i: Info
) -> PyResult<Expr> {
    let arg_type = arg.get_type().clone();
    if let Type::Unknown = ty {
        Ok(arg)
    } else {
        match coerce_type(arg, &ty) {
            Ok(e) => Ok(e),
            Err(_) => {
                py_type_error!(i, "Parameter {id} was annotated with type {ty} \
                                   which is incompatible with argument type {arg_type}.")
            }
        }
    }
}

fn type_check_arguments(args: Vec<Expr>, annot: Vec<Param>) -> PyResult<Vec<Expr>> {
    args.into_iter()
        .zip(annot.into_iter())
        .map(|(arg, Param {id, ty, i})| type_check_argument(arg, id, ty, i))
        .collect::<PyResult<Vec<Expr>>>()
}

fn type_check_call<'py>(
    env: TypeCheckEnv<'py>,
    id: &Name,
    args: Vec<Expr>,
    i: &Info
) -> PyResult<(TypeCheckEnv<'py>, Name, Type, Vec<Expr>)> {
    match env.lookup_top(id) {
        Some(t) => {
            // If the parameters of the called function have been annotated with types, we
            // type-check the provided arguments by attempting to coerce them to the annotated
            // types.
            let args = match extract_annotated_params(&t) {
                Some(annot) => type_check_arguments(args, annot),
                None => Ok(args)
            }?;
            let (env, t) = type_check_top(env, t, args.clone())?;
            let (new_id, ret_ty) = match t {
                Top::CallbackDecl {id, ..} => (id.clone(), Type::Void),
                Top::ExtDecl {id, res_ty, ..} |
                Top::FunDef {v: FunDef {id, res_ty, ..}} => {
                    (id.clone(), res_ty.clone())
                }
            };
            Ok((env, new_id, ret_ty, args))
        },
        None => py_type_error!(i, "Call to unknown function {id}")
    }
}

fn type_check_fun_def<'py>(
    env: TypeCheckEnv<'py>,
    def: FunDef,
    args: Vec<Expr>
) -> PyResult<(TypeCheckEnv<'py>, FunDef, bool)> {
    // Produce environment for the function by unifying the parameter declarations with the
    // types of the provided arguments.
    let (unify_env, params) = unify_parameter_types(def.params.clone(), &args, &def.id, &def.i)?;

    // If this particular function has already been specialized, we immediately return.
    match env.specs.get(&unify_env).cloned() {
        Some(mono_def) => Ok((env, mono_def, false)),
        None => {
            let env = env.enter_function(params.clone());

            // Inline the values of any scalar arguments, as determined by inspecting the AST
            // nodes of the provided arguments.
            let def = inline_const::inline_scalar_values_ast_args(def, &args)?;

            // Specialize the function body by replacing expressions according to the
            // environment.
            let body = specialize::apply(&unify_env, &env.opts, def.body)?;

            // Apply constant folding to the function body, to ensure that slice bounds can be
            // determined given the shapes.
            let body = constant_fold::fold(body);

            // Perform an extended type-check of the body. In addition to normal type-checking,
            // this will:
            // - Insert the bounds of slices based on derived type information.
            // - Recursively run these steps on any called function, or retrieve it from the cache
            //   immediately.
            let (env, body) = body.type_check(env)?;

            let id = def.id.with_new_sym();
            let res_ty = extract_return_type(&body)?;
            let def = FunDef {id, params, body, res_ty, ..def};
            let env = env.exit_function(unify_env, &def)?;

            Ok((env, def, true))
        }
    }
}

fn type_check_top<'py>(
    mut env: TypeCheckEnv<'py>,
    t: Top,
    args: Vec<Expr>
) -> TypeCheckResult<'py, Top> {
    match t {
        Top::CallbackDecl {id, params, i} => {
            let (_, params) = unify_parameter_types(params, &args, &id, &i)?;
            let t = Top::CallbackDecl {id, params, i};
            env.spec_list.push(t.clone());
            Ok((env, t))
        },
        Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i} => {
            let (_, params) = unify_parameter_types(params, &args, &id, &i)?;
            let t = Top::ExtDecl {id, ext_id, params, res_ty, target, header, par, i};
            env.spec_list.push(t.clone());
            Ok((env, t))
        },
        Top::FunDef {v} => {
            let (mut env, v, new_spec) = type_check_fun_def(env, v, args)?;
            let t = Top::FunDef {v};
            if new_spec {
                env.spec_list.push(t.clone());
            }
            Ok((env, t))
        },
    }
}

fn type_check_main<'py>(
    env: TypeCheckEnv<'py>,
    main: FunDef,
    args: Vec<Expr>
) -> PyResult<Ast> {
    let (env, main, _) = type_check_fun_def(env, main, args)?;
    let FunDef {ref res_ty, ref id, ref i, ..} = main;
    match res_ty {
        Type::Void => Ok(Ast {tops: env.spec_list, main}),
        _ => py_type_error!(i, "Main function {id} cannot return a value.")
    }
}

pub fn apply<'py>(
    main: FunDef,
    args: &Vec<Bound<'py, PyAny>>,
    tops: BTreeMap<String, Bound<'py, PyCapsule>>,
    opts: &CompileOptions
) -> PyResult<Ast> {
    let i = main.i.clone();
    let env = TypeCheckEnv::new(tops, opts.clone());
    // NOTE(larshum, 2025-12-17): To represent the arguments provided to the entry point function,
    // we extract the types of the actual Python values and put these in variable nodes to prevent
    // any attempt to inline their contents further.
    let arg_types = extract_argument_types(args, &env.scalar_sizes, &i)?;
    let args = arg_types.into_iter()
        .map(|ty| Expr::Var {id: Name::sym_str(""), ty, i: Info::default()})
        .collect::<Vec<Expr>>();
    type_check_main(env, main, args)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    use std::ffi::CString;
    use strum::IntoEnumIterator;

    #[test]
    fn unify_bool_size() {
        assert_eq!(
            unify_elem_size(ElemSize::Bool, ElemSize::Bool, &i()).unwrap(),
            ElemSize::Bool
        );
    }

    #[test]
    fn unify_i16_i32_size() {
        assert_eq!(
            unify_elem_size(ElemSize::I16, ElemSize::I32, &i()).unwrap(),
            ElemSize::I32
        );
    }

    #[test]
    fn unify_int_float_sizes_fails() {
        assert!(unify_elem_size(ElemSize::I32, ElemSize::F32, &i()).is_err());
    }

    #[test]
    fn unify_signed_unsigned_int_fails() {
        assert!(unify_elem_size(ElemSize::I32, ElemSize::U32, &i()).is_err());
    }

    fn mk_unify_env() -> UnifyEnv {
        UnifyEnv::new(None)
    }

    fn ts_num(n: i64) -> TensorShape {
        TensorShape::Num {n}
    }

    fn ts_sym(id: Name) -> TensorShape {
        TensorShape::Symbol {id}
    }

    #[test]
    fn unify_numerical_tensor_shapes() {
        let env = mk_unify_env();
        let sh = ts_num(10);
        let (env, r) = unify_shape(env, sh.clone(), sh.clone(), &i()).unwrap();
        assert!(env.shape_vars.is_empty());
        assert_eq!(r, sh);
    }

    #[test]
    fn unify_num_symb_tensor_shapes() {
        let env = mk_unify_env();
        let lsh = ts_num(10);
        let rsh = ts_sym(id("x"));
        let (env, r) = unify_shape(env, lsh, rsh, &i()).unwrap();
        assert_eq!(env.shape_vars.len(), 1);
        assert!(env.shape_vars.contains_key(&id("x")));
        assert_eq!(r, ts_num(10));
    }

    #[test]
    fn unify_num_symb_inconsistent_tensor_shapes_fails() {
        let env = mk_unify_env()
            .insert_shape_symbol(id("x"), 20);
        let lsh = ts_num(10);
        let rsh = ts_sym(id("x"));
        let r = unify_shape(env, lsh, rsh, &i());
        assert_py_error_matches(r, "Failed to unify shape variable x = 20 with 10.");
    }

    #[test]
    fn unify_symb_consistent_tensor_shapes() {
        let env = mk_unify_env()
            .insert_shape_symbol(id("x"), 10)
            .insert_shape_symbol(id("y"), 10);
        let lsh = ts_sym(id("x"));
        let rsh = ts_sym(id("y"));
        let (env, r) = unify_shape(env, lsh, rsh, &i()).unwrap();
        assert_eq!(env.shape_vars.len(), 2);
        assert!(env.shape_vars.contains_key(&id("x")));
        assert!(env.shape_vars.contains_key(&id("y")));
        assert_eq!(r, ts_num(10));
    }

    #[test]
    fn unify_symb_inconsistent_tensor_shapes() {
        let env = mk_unify_env()
            .insert_shape_symbol(id("x"), 10)
            .insert_shape_symbol(id("y"), 20);
        let lsh = ts_sym(id("x"));
        let rsh = ts_sym(id("y"));
        let r = unify_shape(env, lsh, rsh, &i());
        assert_py_error_matches(r, "Failed to unify shape variables x = 10 and y = 20.");
    }

    #[test]
    fn unify_symb_lhs_known_tensor_shapes() {
        let env = mk_unify_env().insert_shape_symbol(id("x"), 10);
        let lsh = ts_sym(id("x"));
        let rsh = ts_sym(id("y"));
        let (env, r) = unify_shape(env, lsh, rsh, &i()).unwrap();
        assert_eq!(env.shape_vars.len(), 2);
        assert_eq!(env.shape_vars.get(&id("y")).cloned(), Some(10));
        assert_eq!(r, ts_num(10));
    }

    #[test]
    fn unify_unknown_symbols_tensor_shapes() {
        let env = mk_unify_env();
        let lsh = ts_sym(id("x"));
        let rsh = ts_sym(id("y"));
        let r = unify_shape(env, lsh, rsh, &i());
        assert_py_error_matches(r, "Failed to unify unresolved shape symbols x and y.");
    }

    #[test]
    fn unify_shapes_same_length() {
        let env = mk_unify_env();
        let lsh = vec![ts_num(10)];
        let rsh = vec![ts_num(10)];
        let (env, r) = unify_shapes(env, lsh, rsh, &i()).unwrap();
        assert!(env.shape_vars.is_empty());
        assert_eq!(r, vec![ts_num(10)]);
    }

    #[test]
    fn unify_shapes_with_vars() {
        let env = mk_unify_env();
        let lsh = vec![ts_sym(id("x")), ts_num(10)];
        let rsh = vec![ts_num(20), ts_sym(id("y"))];
        let (env, r) = unify_shapes(env, lsh, rsh, &i()).unwrap();
        assert_eq!(env.shape_vars.len(), 2);
        assert_eq!(env.shape_vars.get(&id("x")).cloned(), Some(20));
        assert_eq!(env.shape_vars.get(&id("y")).cloned(), Some(10));
        assert_eq!(r, vec![ts_num(20), ts_num(10)]);
    }

    #[test]
    fn unify_incompatible_shapes_fails() {
        let env = mk_unify_env();
        let lsh = vec![ts_num(10), ts_num(20), ts_num(30)];
        let rsh = vec![ts_sym(id("x")), ts_sym(id("y")), ts_sym(id("x"))];
        let r = unify_shapes(env, lsh, rsh, &i());
        assert_py_error_matches(r, r"Failed to unify shape variable x = 10 with 30.");
    }

    #[test]
    fn unify_shapes_with_distinct_lengths_fails() {
        let env = mk_unify_env();
        let lsh = vec![ts_num(10), ts_num(20)];
        let rsh = vec![ts_sym(id("x"))];
        let r = unify_shapes(env, lsh, rsh, &i());
        assert_py_error_matches(r, "Found incompatible tensor shapes .*");
    }

    #[test]
    fn unify_unknown_types_fails() {
        let env = mk_unify_env();
        let r = unify_types(env, Type::Unknown, Type::Unknown, &i());
        assert_py_error_matches(r, "Unknown types cannot be unified");
    }

    #[test]
    fn unify_string_types() {
        let env = mk_unify_env();
        assert!(unify_types(env, Type::String, Type::String, &i()).is_ok());
    }

    #[test]
    fn unify_equivalent_scalar_tensor_types() {
        let env = mk_unify_env();
        let ty = scalar(ElemSize::F32);
        let (_, r) = unify_types(env, ty.clone(), ty.clone(), &i()).unwrap();
        assert_eq!(r, ty);
    }

    #[test]
    fn unify_scalar_int_types() {
        let env = mk_unify_env();
        let lty = scalar(ElemSize::I32);
        let rty = scalar(ElemSize::I64);
        let (_, r) = unify_types(env, lty, rty, &i()).unwrap();
        assert_eq!(r, scalar(ElemSize::I64));
    }

    #[test]
    fn unify_non_empty_tensor_types() {
        let env = mk_unify_env();
        let lty = Type::Tensor {sz: fixed_elem_sz(ElemSize::I32), shape: vec![ts_num(10)]};
        let rty = Type::Tensor {sz: fixed_elem_sz(ElemSize::I32), shape: vec![ts_sym(id("x"))]};
        let (env, r) = unify_types(env, lty, rty, &i()).unwrap();
        assert_eq!(env.shape_vars.get(&id("x")).cloned(), Some(10));
        assert_eq!(r, Type::Tensor {sz: fixed_elem_sz(ElemSize::I32), shape: vec![ts_num(10)]});
    }

    #[test]
    fn unify_non_empty_distinct_tensor_types_fails() {
        let env = mk_unify_env();
        let lty = Type::Tensor {sz: fixed_elem_sz(ElemSize::I32), shape: vec![ts_num(10)]};
        let rty = Type::Tensor {sz: fixed_elem_sz(ElemSize::I64), shape: vec![ts_num(10)]};
        let r = unify_types(env, lty, rty, &i());
        assert_py_error_matches(r, "Failed to unify non-scalar tensors .*");
    }

    #[test]
    fn unify_tuples_same_length() {
        let env = mk_unify_env();
        let lty = Type::Tuple {elems: vec![scalar(ElemSize::I32)]};
        let rty = Type::Tuple {elems: vec![scalar(ElemSize::I64)]};
        let (_, r) = unify_types(env, lty, rty, &i()).unwrap();
        assert_eq!(r, Type::Tuple {elems: vec![scalar(ElemSize::I64)]});
    }

    #[test]
    fn unify_tuples_distinct_length() {
        let env = mk_unify_env();
        let lty = Type::Tuple {elems: vec![]};
        let rty = Type::Tuple {elems: vec![scalar(ElemSize::I32)]};
        let r = unify_types(env, lty, rty, &i());
        assert_py_error_matches(r, "Failed to unify tuple types of different lengths");
    }

    #[test]
    fn unify_dictionaries_equal_keys() {
        let env = mk_unify_env();
        let lty = dict_ty(vec![
            ("x", scalar(ElemSize::F32)),
            ("y", scalar(ElemSize::F64))
        ]);
        let rty = dict_ty(vec![
            ("x", scalar(ElemSize::F64)),
            ("y", scalar(ElemSize::F16))
        ]);
        let (_, r) = unify_types(env, lty, rty, &i()).unwrap();
        let expected = dict_ty(vec![
            ("x", scalar(ElemSize::F64)),
            ("y", scalar(ElemSize::F64))
        ]);
        assert_eq!(r, expected);
    }

    #[test]
    fn unify_dictionaries_distinct_lengths_fails() {
        let env = mk_unify_env();
        let lty = dict_ty(vec![]);
        let rty = dict_ty(vec![("x", tyuk())]);
        let r = unify_types(env, lty, rty, &i());
        assert_py_error_matches(r, "Failed to unify dictionary types of different .*");
    }

    #[test]
    fn unify_dictionaries_distinct_keys_fails() {
        let env = mk_unify_env();
        let lty = dict_ty(vec![("x", tyuk())]);
        let rty = dict_ty(vec![("y", tyuk())]);
        let r = unify_types(env, lty, rty, &i());
        assert_py_error_matches(r, "Failed to unify dictionary types with distinct keys");
    }

    #[test]
    fn unify_void_types() {
        let env = mk_unify_env();
        let (_, r) = unify_types(env, Type::Void, Type::Void, &i()).unwrap();
        assert_eq!(r, Type::Void);
    }

    #[test]
    fn unify_incompatible_types_fails() {
        let env = mk_unify_env();
        let r = unify_types(env, Type::Void, Type::String, &i());
        assert_py_error_matches(r, "Failed to unify incompatible types .*")
    }

    #[test]
    fn unify_type_variable_with_known() {
        let env = mk_unify_env();
        let id = Name::sym_str("");
        let lty = scalar(ElemSize::F32);
        let rty = Type::Tensor {
            sz: TensorElemSize::Variable {id: id.clone()},
            shape: vec![]
        };
        let (env, ty) = unify_types(env, lty, rty, &i()).unwrap();
        assert_eq!(env.type_vars.len(), 1);
        assert!(env.type_vars.contains_key(&id));
        assert_eq!(ty, scalar(ElemSize::F32));
    }

    #[test]
    fn unify_unknown_type_variables_fails() {
        let env = mk_unify_env();
        let lty = Type::Tensor {
            sz: TensorElemSize::Variable {id: id("a")},
            shape: vec![]
        };
        let rty = Type::Tensor {
            sz: TensorElemSize::Variable {id: id("b")},
            shape: vec![]
        };
        let r = unify_types(env, lty, rty, &i());
        assert_py_error_matches(r, "Failed to unify unbound type variables a and b.");
    }

    #[test]
    fn unify_parameters_no_annot() {
        let params = vec![
            Param {id: id("x"), ty: Type::Unknown, i: i()},
            Param {id: id("y"), ty: Type::Unknown, i: i()}
        ];
        let args = vec![
            var("", scalar(ElemSize::F32)),
            var("", Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_num(10)]})
        ];
        let (_, r) = unify_parameter_types(params, &args, &id("f"), &i()).unwrap();
        let expected = vec![
            Param {id: id("x"), ty: scalar(ElemSize::F32), i: i()},
            Param {
                id: id("y"),
                ty: Type::Tensor {
                    sz: fixed_elem_sz(ElemSize::F32),
                    shape: vec![ts_num(10)]
                },
                i: i()
            }
        ];
        assert_eq!(r, expected);
    }

    #[test]
    fn unify_parameters_incompatible_annot_fails() {
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I64),
            shape: vec![ts_sym(id("x"))]
        };
        let params = vec![
            Param {id: id("a"), ty: ty.clone(), i: i()},
            Param {id: id("b"), ty: ty.clone(), i: i()},
        ];
        let args = vec![
            var("", Type::Tensor {sz: fixed_elem_sz(ElemSize::I64), shape: vec![ts_num(10)]}),
            var("", Type::Tensor {sz: fixed_elem_sz(ElemSize::I64), shape: vec![ts_num(20)]}),
        ];
        let r = unify_parameter_types(params, &args, &id("f"), &i());
        assert_py_error_matches(r, "Parameter b was annotated with type .* \
                                    which is incompatible with argument type .*");
    }

    #[test]
    fn unify_parameters_invalid_number_of_arguments_fails() {
        let params = vec![];
        let args = vec![var("", Type::Unknown)];
        let r = unify_parameter_types(params, &args, &id("f"), &i());
        assert_py_error_matches(r, r"Function f expects 0 parameters.*called with 1 argument.*");
    }

    fn ssz_i32_f32() -> ScalarSizes {
        ScalarSizes {int: ElemSize::I32, float: ElemSize::F32}
    }

    fn try_extract_type(s: &str, scalar_sizes: &ScalarSizes, i: &Info, expected: Type) {
        let s = CString::new(s).unwrap();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let arg = py.eval(&s, None, None).unwrap();
            let ty = extract_type(&arg, scalar_sizes, i).unwrap();
            assert_eq!(ty, expected);
        })
    }

    #[test]
    fn extract_bool_literal_type() {
        try_extract_type("True", &ssz_i32_f32(), &i(), scalar(ElemSize::Bool));
    }

    #[test]
    fn extract_int_literal_types() {
        let mut ss = ssz_i32_f32();
        for int_sz in vec![ElemSize::I8, ElemSize::I16, ElemSize::I32, ElemSize::I64] {
            ss.int = int_sz.clone();
            try_extract_type("0", &ss, &i(), scalar(int_sz));
        }
    }

    #[test]
    fn extract_float_literal_types() {
        let mut ss = ssz_i32_f32();
        for float_sz in vec![ElemSize::F16, ElemSize::F32, ElemSize::F64] {
            ss.float = float_sz.clone();
            try_extract_type("0.0", &ss, &i(), scalar(float_sz));
        }
    }

    #[test]
    fn extract_return_type_empty() {
        assert_eq!(extract_return_type(&vec![]).unwrap(), Type::Void);
    }

    #[test]
    fn extract_return_type_returns_value() {
        let body = vec![return_stmt(int(1, Some(ElemSize::I32)))];
        assert_eq!(extract_return_type(&body).unwrap(), scalar(ElemSize::I32));
    }

    #[test]
    fn extract_incompatible_return_types_fails() {
        let body = vec![
            if_stmt(
                bool_expr(true, Some(ElemSize::Bool)),
                vec![return_stmt(int(1, Some(ElemSize::I32)))],
                vec![return_stmt(float(1.0, Some(ElemSize::F32)))]
            )
        ];
        let r = extract_return_type(&body);
        assert_py_error_matches(r, "Found incompatible return types .* and .*");
    }

    #[test]
    fn ensure_bool_conditional() {
        let cond = bool_expr(true, Some(ElemSize::Bool));
        assert_eq!(ensure_conditional_type(cond.clone()).unwrap(), cond);
    }

    #[test]
    fn ensure_int_conditional() {
        for int_sz in ElemSize::iter().filter(|sz| sz.is_integer()) {
            let cond = int(1, Some(int_sz.clone()));
            let expected = binop(
                cond.clone(),
                BinOp::Neq,
                int(0, Some(int_sz)),
                scalar(ElemSize::Bool)
            );
            assert_eq!(ensure_conditional_type(cond).unwrap(), expected);
        }
    }

    #[test]
    fn coerce_equal_types_identity() {
        let ty = scalar(ElemSize::I32);
        let e = int(2, Some(ElemSize::I32));
        assert_eq!(coerce_type(e.clone(), &ty).unwrap(), e);
    }

    #[test]
    fn coerce_type_i32_i64() {
        let ty = scalar(ElemSize::I64);
        let e = int(2, Some(ElemSize::I32));
        let expected = convert(e.clone(), ty.clone());
        assert_eq!(coerce_type(e, &ty).unwrap(), expected);
    }

    #[test]
    fn coerce_type_i64_i32() {
        let ty = scalar(ElemSize::I32);
        let e = int(2, Some(ElemSize::I64));
        let expected = convert(e.clone(), ty.clone());
        assert_eq!(coerce_type(e, &ty).unwrap(), expected);
    }

    #[test]
    fn coerce_type_float_to_int_fails() {
        let ty = scalar(ElemSize::I32);
        let e = float(2.5, Some(ElemSize::F32));
        assert_py_error_matches(coerce_type(e, &ty), "Cannot coerce element size .* to .*");
    }

    #[test]
    fn coerce_type_non_empty_tensor_distinct_types_fails() {
        let ty = Type::Tensor {sz: fixed_elem_sz(ElemSize::F64), shape: vec![ts_num(10)]};
        let e = var("x", Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_num(10)]});
        assert_py_error_matches(coerce_type(e, &ty), "Cannot coerce non-empty tensor of element size .* to .*");
    }

    #[test]
    fn coerce_type_non_empty_tensor_distinct_shapes_fails() {
        let ty = Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_num(12)]};
        let e = var("x", Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_num(10)]});
        assert_py_error_matches(coerce_type(e, &ty), "Failed to unify distinct dimensions 10 and 12.");
    }

    #[test]
    fn coerce_type_non_empty_tensor_variable() {
        let ty = Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_num(10)]};
        let e = var("x", Type::Tensor {sz: fixed_elem_sz(ElemSize::F32), shape: vec![ts_sym(id("N"))]});
        let expected = var("x", ty.clone());
        assert_eq!(coerce_type(e, &ty).unwrap(), expected);
    }

    #[test]
    fn coerce_type_non_empty_tensor() {
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::F32),
            shape: vec![ts_num(10), ts_num(12)]
        };
        let var_ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::F32),
            shape: vec![ts_sym(id("N")), ts_sym(id("N"))]
        };
        let e = var("x", var_ty);
        assert_py_error_matches(coerce_type(e, &ty), "Failed to unify shape variable N.*with");
    }

    #[test]
    fn coerce_type_type_variable_no_coercion() {
        let ty = Type::Tensor {
            sz: TensorElemSize::Variable {id: id("sz")},
            shape: vec![]
        };
        let e = var("x", Type::Tensor {
            sz: fixed_elem_sz(ElemSize::F32),
            shape: vec![]
        });
        assert_eq!(coerce_type(e.clone(), &ty).unwrap(), e);
    }

    #[test]
    fn type_check_unary_subtraction_i64() {
        let r = type_check_unop(&UnOp::Sub, &int(2, Some(ElemSize::I64)), &i());
        assert_eq!(r.unwrap(), scalar(ElemSize::I64));
    }

    #[test]
    fn type_check_unary_subtraction_f16() {
        let r = type_check_unop(&UnOp::Sub, &float(2.5, Some(ElemSize::F16)), &i());
        assert_eq!(r.unwrap(), scalar(ElemSize::F16));
    }

    #[test]
    fn type_check_unary_subtraction_bool_fails() {
        let r = type_check_unop(&UnOp::Sub, &bool_expr(true, Some(ElemSize::Bool)), &i());
        assert_py_error_matches(r, "Unsupported argument type .* of unary operator");
    }

    #[test]
    fn type_check_subtraction_distinct_int_types() {
        let l = int(2, Some(ElemSize::I32));
        let r = int(3, Some(ElemSize::I16));
        let (l, ty, r) = type_check_binop(l, &BinOp::Sub, r, &i()).unwrap();
        assert_eq!(*l, int(2, Some(ElemSize::I32)));
        assert_eq!(*r, convert(int(3, Some(ElemSize::I16)), scalar(ElemSize::I32)));
        assert_eq!(ty, scalar(ElemSize::I32));
    }

    #[test]
    fn type_check_subtraction_distinct_float_types() {
        let l = float(2.5, Some(ElemSize::F16));
        let r = float(3.5, Some(ElemSize::F64));
        let (l, ty, r) = type_check_binop(l, &BinOp::Sub, r, &i()).unwrap();
        assert_eq!(*l, convert(float(2.5, Some(ElemSize::F16)), scalar(ElemSize::F64)));
        assert_eq!(*r, float(3.5, Some(ElemSize::F64)));
        assert_eq!(ty, scalar(ElemSize::F64));
    }

    fn mk_tcenv<'py>(params: Vec<Param>) -> TypeCheckEnv<'py> {
        let opts = CompileOptions::default();
        let env = TypeCheckEnv::new(BTreeMap::new(), opts);
        env.enter_function(params)
    }

    #[test]
    fn type_check_dict_access_expression() {
        let target = var("x", Type::Unknown);
        let e = subscript(target, string("a"), Type::Unknown);
        let params = vec![
            Param {id: id("x"), ty: dict_ty(vec![("a", scalar(ElemSize::F32))]), i: i()}
        ];
        let env = mk_tcenv(params);
        let (_, e) = e.type_check(env).unwrap();
        assert_eq!(e.get_type().clone(), scalar(ElemSize::F32));
    }

    #[test]
    fn type_check_slice_expression() {
        let target_shape = vec![
            TensorShape::Num {n: 10},
            TensorShape::Num {n: 30},
            TensorShape::Num {n: 20},
        ];
        let target = var("x", Type::Unknown);
        let index = tuple(vec![slice(None, None), int(4, None), slice(None, None)]);
        let e = subscript(target, index, Type::Unknown);
        let ty = Type::Tensor {
            sz: TensorElemSize::Fixed {sz: ElemSize::F32},
            shape: target_shape
        };
        let params = vec![Param {id: id("x"), ty, i: i()}];
        let env = mk_tcenv(params);
        let (_, e) = e.type_check(env).unwrap();
        assert_eq!(e.get_type().clone(), scalar(ElemSize::F32));
    }
}
