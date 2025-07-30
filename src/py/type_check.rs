use super::ast::*;
use super::constant_fold;
use super::slices;
use crate::py_type_error;
use crate::buffer::DataType;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;
use crate::utils::reduce;
use crate::utils::smap::{SFold, SMapAccum};

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;

use std::collections::BTreeMap;

use itertools::Itertools;

#[derive(Clone, Debug)]
pub struct TypeCheckEnv {
    pub vars: BTreeMap<Name, Type>,
    pub arg_types: BTreeMap<String, Vec<Type>>,
    pub res_types: BTreeMap<String, Type>,
    pub res_ty: Type,
    pub shapes_only: bool,
    pub float_size: ElemSize
}

impl Default for TypeCheckEnv {
    fn default() -> TypeCheckEnv {
        TypeCheckEnv {
            vars: BTreeMap::new(),
            arg_types: BTreeMap::new(),
            res_types: BTreeMap::new(),
            res_ty: Type::Unknown,
            shapes_only: false,
            float_size: ElemSize::F64
        }
    }
}

impl TypeCheckEnv {
    fn is_scalar_h(&self, ty: &Type, predicate: impl Fn(&ElemSize) -> bool) -> bool {
        match ty {
            Type::Tensor {sz, shape} => {
                self.shapes_only || (shape.is_empty() && predicate(sz))
            },
            _ => false
        }
    }

    pub fn is_scalar(&self, ty: &Type) -> bool {
        self.is_scalar_h(ty, |_| true)
    }

    pub fn is_bool_scalar(&self, ty: &Type) -> bool {
        self.is_scalar_h(ty, |sz| sz.is_boolean())
    }

    pub fn is_signed_int_scalar(&self, ty: &Type) -> bool {
        self.is_scalar_h(ty, |sz| sz.is_signed_integer())
    }

    pub fn is_unsigned_int_scalar(&self, ty: &Type) -> bool {
        self.is_scalar_h(ty, |sz| sz.is_unsigned_integer())
    }

    pub fn is_int_scalar(&self, ty: &Type) -> bool {
        self.is_signed_int_scalar(ty) || self.is_unsigned_int_scalar(ty)
    }

    pub fn is_float_scalar(&self, ty: &Type) -> bool {
        self.is_scalar_h(ty, |sz| sz.is_floating_point())
    }

    fn is_tensor(&self, ty: &Type, predicate: impl Fn(&ElemSize) -> bool) -> bool {
        match ty {
            Type::Tensor {sz, shape} if !shape.is_empty() => {
                self.shapes_only || predicate(sz)
            },
            _ => false
        }
    }

    pub fn is_arith_tensor(&self, ty: &Type) -> bool {
        self.is_tensor(ty, |sz| sz.is_integer() || sz.is_floating_point())
    }
}

fn assert_known_param_type(acc: Result<(), String>, id: &Name, ty: &Type) -> Result<(), String> {
    match ty {
        Type::Tuple {elems} => {
            elems.iter()
                .fold(Ok(acc?), |acc, ty| assert_known_param_type(acc, id, ty))
        },
        Type::Dict {fields} => {
            fields.values()
                .fold(Ok(acc?), |acc, ty| assert_known_param_type(acc, id, ty))
        },
        Type::Unknown => Err(format!("{id}")),
        Type::String | Type::Tensor {..} | Type::Void => Ok(())
    }
}

fn assert_known_params(id: &Name, params: &Vec<Param>) -> PyResult<()> {
    let errs = params.iter()
        .map(|Param {id, ty, ..}| assert_known_param_type(Ok(()), id, ty))
        .filter(|r| r.is_err())
        .map(|r| r.unwrap_err())
        .collect::<Vec<String>>();
    if errs.is_empty() {
        Ok(())
    } else {
        let id = id.get_str();
        let e = errs.into_iter().join(", ");
        py_type_error!(
            Info::default(),
            "Parameters {e} of function {id} have unknown type.\n\
             Make sure to annotate the types of parameters in functions called \
             within the main function.")
    }
}

fn get_buffer_shape<'py>(
    t: &Bound<'py, PyAny>
) -> PyResult<Vec<i64>> {
    t.getattr("shape")?.extract::<Vec<i64>>()
}

fn convert_type<'py>(arg: &Bound<'py, PyAny>, float_size: &ElemSize) -> PyResult<Type> {
    let py = arg.py();
    let buffer = py.import("prickle.buffer")?;
    let ty = arg.get_type();
    if ty.eq(buffer.getattr("Buffer")?)? {
        let dtype = arg.getattr("dtype")?.extract::<DataType>()?;
        let sz = dtype.sz;
        let shape = get_buffer_shape(&arg)?;
        Ok(Type::Tensor {sz, shape})
    } else if arg.is_instance(&PyInt::type_object(arg.py()))? {
        Ok(Type::Tensor {sz: ElemSize::I64, shape: vec![]})
    } else if arg.is_instance(&PyFloat::type_object(arg.py()))? {
        Ok(Type::Tensor {sz: float_size.clone(), shape: vec![]})
    } else if arg.is_instance(&PyDict::type_object(arg.py()))? {
        let fields = arg.call_method0("items")?
            .try_iter()?
            .map(|f| {
                let f = f?;
                let id = f.get_item(0)?.extract::<String>()?;
                let ty = f.get_item(1)?;
                Ok((id, convert_type(&ty, float_size)?))
            })
            .collect::<PyResult<BTreeMap<String, Type>>>()?;
        Ok(Type::Dict {fields})
    } else {
        py_type_error!(Info::default(), "Argument {0:?} has unsupported type {1:?}", arg, ty)
    }
}

fn lub_elem_size(
    lhs: &ElemSize,
    rhs: &ElemSize,
    i: &Info
) -> PyResult<ElemSize> {
    match (lhs, rhs) {
        (ElemSize::Bool, ElemSize::Bool) => Ok(rhs.clone()),
        (ElemSize::I8, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I16, ElemSize::I8) => Ok(ElemSize::I16),
        (ElemSize::I16, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I32, ElemSize::I8 | ElemSize::I16) => Ok(ElemSize::I32),
        (ElemSize::I32, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I64, _) if rhs.is_signed_integer() => Ok(lhs.clone()),
        (ElemSize::U8, _) if rhs.is_unsigned_integer() => Ok(rhs.clone()),
        (ElemSize::U16, ElemSize::U8) => Ok(ElemSize::U16),
        (ElemSize::U16, _) if rhs.is_unsigned_integer() => Ok(rhs.clone()),
        (ElemSize::U32, ElemSize::U8 | ElemSize::U16) => Ok(ElemSize::U32),
        (ElemSize::U32, _) if rhs.is_unsigned_integer() => Ok(rhs.clone()),
        (ElemSize::U64, _) if rhs.is_unsigned_integer() => Ok(lhs.clone()),
        (ElemSize::F16, _) if rhs.is_floating_point() => Ok(rhs.clone()),
        (ElemSize::F32, ElemSize::F16) => Ok(ElemSize::F32),
        (ElemSize::F32, _) if rhs.is_floating_point() => Ok(rhs.clone()),
        (ElemSize::F64, _) if rhs.is_floating_point() => Ok(lhs.clone()),
        _ => py_type_error!(i, "Incompatible element types")
    }
}

fn eq_elem_size(
    env: &TypeCheckEnv,
    lsz: &ElemSize,
    rsz: &ElemSize
) -> bool {
    env.shapes_only || lsz.eq(&rsz)
}

fn compatible_elem_types(
    env: &TypeCheckEnv,
    lhs: &ElemSize,
    rhs: &ElemSize
) -> bool {
    // If we are only interested in the shape, we treat all element sizes as equal.
    env.shapes_only || lub_elem_size(lhs, rhs, &Info::default()).is_ok()
}

fn ensure_scalar_type(
    env: &TypeCheckEnv,
    e: Expr,
    expected: ElemSize
) -> PyResult<Expr> {
    let i = e.get_info();
    let ty = e.get_type();
    match ty {
        Type::Tensor {sz, shape} if shape.is_empty() => {
            // We allow it if the two element size types are compatible. If the types are not
            // equivalent, we insert a conversion to the expected type, unless we are only
            // interested in checking the shapes.
            if compatible_elem_types(env, &sz, &expected) {
                if eq_elem_size(env, &sz, &expected) {
                    Ok(e)
                } else {
                    let ty = Type::Tensor {sz: expected, shape: vec![]};
                    Ok(Expr::Convert {e: Box::new(e), ty})
                }
            } else {
                py_type_error!(i, "Expected element of type {expected}, found \
                                   incompatible element type {sz}.")
            }
        },
        _ => py_type_error!(i, "Expected element of scalar type {expected}, \
                                but found type {ty}.")
    }
}

fn unify_shapes_helper<'a>(
    mut acc: Vec<i64>,
    mut l: impl Iterator<Item=&'a i64>,
    mut r: impl Iterator<Item=&'a i64>
) -> Option<Vec<i64>> {
    match (l.next(), r.next()) {
        (Some(ls), Some(rs)) if ls == rs || *ls == 1 || *rs == 1 => {
            acc.push(*ls);
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
            let ls = lshape.iter().join(", ");
            let rs = rshape.iter().join(", ");
            py_type_error!(i, "Found incompatible shapes {ls} and {rs}")
        }
    }
}

fn coerce_type(env: &TypeCheckEnv, e: Expr, expected: &Type) -> PyResult<Expr> {
    if e.get_type().eq(expected) {
        Ok(e)
    } else {
        let i = e.get_info();
        let actual = e.get_type();
        match (actual, expected) {
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                if env.shapes_only {
                    let _ = unify_shapes(lsh.clone(), rsh.clone(), &i)?;
                    Ok(e)
                } else {
                    if lsh.len() == 0 && rsh.len() == 0 {
                        ensure_scalar_type(env, e, rsz.clone())
                    } else if lsz == rsz && lsh == rsh {
                        Ok(e)
                    } else {
                        py_type_error!(i, "Cannot coerce incompatible tensor types ({actual} != {expected})")
                    }
                }
            }
            (Type::Tuple {..}, Type::Tuple {elems: r}) => {
                if let Expr::Tuple {elems, i, ..} = e {
                    let elems = elems.into_iter()
                        .zip(r.iter())
                        .map(|(e, ty)| coerce_type(env, e, ty))
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
            _ => py_type_error!(
                i,
                "Cannot coerce expression {0} of type {1} to type {2}",
                e, actual, expected
            )
        }
    }
}

/// Finds the least upper bound of two types, with respect to sizes of types. The least upper bound
/// of two integer or floating-point types is the integer or floating-point type with the smallest
/// size that is larger than or equal to that of both arguments. For instance, the least upper
/// bound of an int16 and an int32 is int32.
fn lub_type(env: &TypeCheckEnv, l: Type, r: Type, i: &Info) -> PyResult<Type> {
    match (&l, &r) {
        (_, Type::Unknown) => Ok(l),
        (Type::Unknown, _) => Ok(r),
        (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
            if env.shapes_only {
                // When we only care about the shapes, we do not care about the element size type
                // of the tensor, so we pick the left-hand side value.
                let shape = unify_shapes(lsh.clone(), rsh.clone(), i)?;
                Ok(Type::Tensor {sz: lsz.clone(), shape})
            } else {
                if lsh.is_empty() && rsh.is_empty() {
                    let sz = lub_elem_size(lsz, rsz, i)?;
                    Ok(Type::Tensor {sz, shape: vec![]})
                } else {
                    if l.eq(&r) {
                        Ok(l)
                    } else {
                        py_type_error!(i, "Cannot unify incompatible tensor \
                                           types {l} and {r}.")
                    }
                }
            }
        },
        _ if l.eq(&r) => Ok(l),
        _ => py_type_error!(i, "Cannot unify incompatible types {l} and {r}")
    }
}

fn is_well_typed_arith_tensor_builtin(
    env: &TypeCheckEnv,
    op: &Builtin,
    ty: &Type
) -> bool {
    match op {
        Builtin::Max | Builtin::Min | Builtin::Sum | Builtin::Prod => {
            env.is_arith_tensor(ty)
        },
        _ => false
    }
}

fn type_check_builtin(
    env: &TypeCheckEnv,
    func: Builtin,
    mut args: Vec<Expr>,
    axis: Option<i64>,
    i: Info
) -> PyResult<Expr> {
    // Type-check the built-in functions
    match &func {
        // Literals
        Builtin::Inf if args.is_empty() => {
            let ty = Type::Tensor {sz: ElemSize::F64, shape: vec![]};
            Ok(Expr::Builtin {func, args, axis, ty, i})
        },
        // Unary tensor reductions, with an optional keyword argument
        Builtin::Max | Builtin::Min | Builtin::Sum | Builtin::Prod if args.len() == 1 => {
            let arg_ty = args[0].get_type();
            if is_well_typed_arith_tensor_builtin(env, &func, &arg_ty) {
                let (sz, mut shape) = match arg_ty.clone() {
                    Type::Tensor {sz, shape} => (sz, shape),
                    _ => unreachable!()
                };
                match axis {
                    Some(n) => {
                        let dim = shape.len() as i64;
                        if n < dim && n >= -dim {
                            shape.remove(n.rem_euclid(dim) as usize);
                        } else {
                            py_type_error!(i, "Invalid axis value {n} for \
                                               tensor of {dim} dimensions.")?
                        }
                    },
                    None => {
                        shape.clear();
                    }
                };
                let ty = Type::Tensor {sz, shape};
                Ok(Expr::Builtin {func, args, axis, ty, i})
            } else {
                py_type_error!(i, "Unexpected argument of type {arg_ty} passed \
                                   to tensor reduction builtin")
            }
        },
        // Unary operations on (floating-point) scalar values
        Builtin::Exp | Builtin::Log | Builtin::Cos | Builtin::Sin |
        Builtin::Sqrt if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if env.is_float_scalar(&ty) {
                Ok(Expr::Builtin {func, args, axis, ty, i})
            } else {
                py_type_error!(i, "Unexpected type {ty} of unary builtin (expected float)")
            }
        },
        Builtin::Tanh if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if env.shapes_only {
                if env.is_scalar(&ty) {
                    Ok(Expr::Builtin {func, args, axis, ty, i})
                } else {
                    py_type_error!(i, "Expected scalar but found non-scalar tensor")
                }
            } else {
                if env.is_float_scalar(&ty) {
                    Ok(Expr::Builtin {func, args, axis, ty, i})
                } else {
                    py_type_error!(i, "Unexpected type {ty} of tanh builtin (expected float)")
                }
            }
        },
        Builtin::Abs if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if env.is_signed_int_scalar(&ty) || env.is_float_scalar(&ty) {
                Ok(Expr::Builtin {func, args, axis, ty, i})
            } else if env.is_unsigned_int_scalar(&ty) {
                // NOTE: The absolute function of an unsigned integer is the identity function, so
                // we can simply return the value without wrapping it in a call to the absolute
                // function.
                let arg = args.remove(0);
                Ok(arg)
            } else {
                py_type_error!(i, "Unexpected type {ty} of abs builtin")
            }
        },
        // Unary cast operation on scalar values
        Builtin::Convert {sz} if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if env.shapes_only {
                Ok(Expr::Builtin {func, args, axis, ty, i})
            } else if env.is_scalar(&ty) {
                let arg = args.remove(0);
                Ok(Expr::Convert {
                    e: Box::new(arg),
                    ty: Type::Tensor {sz: sz.clone(), shape: vec![]}
                })
            } else {
                py_type_error!(i, "Expected scalar type in type conversion, found {ty}")
            }
        },
        // Binary operations on scalar values
        Builtin::Max | Builtin::Min | Builtin::Atan2 if args.len() == 2 => {
            let mk_builtin = |func, fst, snd, ty, i| {
                let fst = coerce_type(env, fst, &ty)?;
                let snd = coerce_type(env, snd, &ty)?;
                let args = vec![fst, snd];
                Ok(Expr::Builtin {func, args, axis, ty, i})
            };
            let snd = args.pop().unwrap();
            let fst = args.pop().unwrap();
            let ty = lub_type(env, fst.get_type().clone(), snd.get_type().clone(), &i)?;
            if env.shapes_only {
                mk_builtin(func, fst, snd, ty, i)
            } else {
                match ty.get_scalar_elem_size() {
                    Some(_) if func != Builtin::Atan2 => {
                        mk_builtin(func, fst, snd, ty, i)
                    },
                    Some(sz) if func == Builtin::Atan2 => {
                        if sz.is_floating_point() {
                            mk_builtin(func, fst, snd, ty, i)
                        } else {
                            py_type_error!(i, "Operation atan2 is only supported for floats.")
                        }
                    },
                    _ => {
                        py_type_error!(i, "Unexpected type {ty} of builtin")
                    }
                }
            }
        },
        _ => py_type_error!(i, "Unsupported use of builtin {func} with {0} \
                                args.", args.len())
    }
}

fn type_check_unop(
    env: &TypeCheckEnv,
    op: &UnOp,
    arg: &Expr,
    i: &Info
) -> PyResult<Type> {
    let ty = arg.get_type();
    match op {
        UnOp::Sub => {
            if env.is_int_scalar(&ty) || env.is_float_scalar(&ty) {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of unary minus")
            }
        },
        UnOp::Not => {
            if env.is_bool_scalar(&ty) {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of boolean negation")
            }
        }
        UnOp::BitNeg => {
            if env.is_int_scalar(&ty) {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of bitwise negation")
            }
        },
        UnOp::Exp | UnOp::Log | UnOp::Cos | UnOp::Sin | UnOp::Sqrt |
        UnOp::Tanh | UnOp::Abs | UnOp::Addressof => {
            py_type_error!(i, "Type-checking not implemented for this operator")
        }
    }
}

fn type_check_binop(
    env: &TypeCheckEnv,
    lhs: Expr,
    op: &BinOp,
    rhs: Expr,
    i: &Info
) -> PyResult<(Box<Expr>, Type, Box<Expr>)> {
    let lty = lhs.get_type().clone();
    let rty = rhs.get_type().clone();
    let ty = lub_type(env, lty, rty, i)?;
    let lhs = coerce_type(&env, lhs, &ty)?;
    let rhs = coerce_type(&env, rhs, &ty)?;
    let ty = if env.shapes_only {
        // If the shapes_only argument is set, we accept arithmetic operations on non-scalar
        // tensors to support slicing operations.
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => Ok(ty),
            _ if env.is_scalar(&ty) => Ok(ty),
            _ => py_type_error!(i, "Binary operator not supported for \
                                    non-scalar tensors")
        }
    } else {
        match op {
            // Arithmetic operations supporting either integers or floating point numbers
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                if env.is_int_scalar(&ty) || env.is_float_scalar(&ty) {
                    Ok(ty)
                } else {
                    py_type_error!(i, "Invalid type {ty} of arithmetic operation")
                }
            },
            // Arithmetic operations only supported for integers
            BinOp::FloorDiv | BinOp::Rem => {
                if env.is_int_scalar(&ty) {
                    Ok(ty)
                } else {
                    py_type_error!(i, "Invalid type {ty} of integer arithmetic operation")
                }
            },
            // Arithmetic operations only supported for floating-point numbers
            BinOp::Pow => {
                match ty.get_scalar_elem_size() {
                    Some(ElemSize::F32 | ElemSize::F64) => Ok(ty),
                    _ => py_type_error!(i, "Invalid type {ty} of floating-point \
                                            arithmetic operation")
                }
            },
            // Boolean operations
            BinOp::And | BinOp::Or => {
                if env.is_bool_scalar(&ty) {
                    Ok(ty)
                } else {
                    py_type_error!(i, "Invalid type {ty} of boolean operation")
                }
            },
            // Bitwise operations
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::BitShl | BinOp::BitShr => {
                if env.is_int_scalar(&ty) {
                    Ok(ty)
                } else {
                    py_type_error!(i, "Invalid type {ty} of bitwise operation")
                }
            },
            // Boolean comparison operations, allowing comparison between elementary types
            BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => {
                if env.is_scalar(&ty) {
                    Ok(Type::Tensor {sz: ElemSize::Bool, shape: vec![]})
                } else {
                    py_type_error!(i, "Invalid type {ty} of boolean comparison operation")
                }
            },
            // Comparison operations on arithmetic types
            BinOp::Max | BinOp::Min => {
                if env.is_int_scalar(&ty) || env.is_float_scalar(&ty) {
                    Ok(ty)
                } else {
                    py_type_error!(i, "Invalid type {ty} of comparison operation")
                }
            }
            BinOp::Atan2 => {
                py_type_error!(i, "Type-checking not implemented for this operator")
            }
        }
    }?;
    Ok((Box::new(lhs), ty, Box::new(rhs)))
}

fn num_index_dimensions(idx: &Expr) -> usize {
    match idx {
        Expr::Tuple {elems, ..} => elems.len(),
        _ => 1
    }
}

fn extract_slice_bound(
    o: &Option<Box<Expr>>,
    default: i128,
    msg_prefix: &str,
    i: &Info
) -> PyResult<i64> {
    match slices::extract_slice_index(o, default) {
        Some(idx) => Ok(idx as i64),
        None => py_type_error!(i, "{msg_prefix} of slice must be fixed.")
    }
}

fn extract_slice_dim<'a>(
    acc: (&'a [i64], Vec<i64>),
    e: &Expr
) -> PyResult<(&'a [i64], Vec<i64>)> {
    let (shape, mut slice_dims) = acc;
    let i = e.get_info();
    match e {
        Expr::Slice {lo, hi, i, ..} => {
            let lo_idx = extract_slice_bound(lo, 0, "Lower-bound", &i)?;
            let hi_idx = extract_slice_bound(hi, shape[0] as i128, "Upper-bound", &i)?;
            let hi_idx = if hi_idx < 0 {
                hi_idx + shape[0]
            } else {
                hi_idx
            };
            if lo_idx < hi_idx {
                if lo_idx >= 0 && hi_idx <= shape[0] {
                    slice_dims.push(hi_idx - lo_idx);
                } else {
                    py_type_error!(
                        i,
                        "Slice dimensions {0} and {1} out of range for shape {2}",
                        lo_idx, hi_idx, shape[0]
                    )?
                }
            } else {
                py_type_error!(
                    i,
                    "Lower-bound {0} of slice must be less than its upper-bound {1}.",
                    lo_idx, hi_idx
                )?
            }
        },
        Expr::Int {v, ..} => {
            let vi = *v as i64;
            let idx = if vi < 0 {
                vi + shape[0]
            } else {
                vi
            };
            if idx >= 0 && idx < shape[0] {
                slice_dims.push(0);
            } else {
                py_type_error!(i, "Index {0} out of range for shape {1}", v, shape[0])?
            }
        },
        // For non-literal valued indices, we cannot do bounds-checking. We allow them anyway for
        // the sake of flexibility.
        _ => slice_dims.push(0)
    };
    Ok((&shape[1..], slice_dims))
}

fn extract_slice_dims(idx: &Expr, shape: &[i64]) -> PyResult<Vec<i64>> {
    match idx {
        Expr::Tuple {elems, ..} => {
            let (_, dims) = elems.sfold_result(Ok((shape, vec![])), extract_slice_dim)?;
            Ok(dims)
        },
        Expr::Slice {..} => {
            let (_, dims) = extract_slice_dim((shape, vec![]), idx)?;
            Ok(dims)
        },
        _ => Ok(vec![0])
    }
}

fn type_check_indexing(
    env: TypeCheckEnv,
    target: &Expr,
    idx: Expr,
) -> PyResult<(TypeCheckEnv, Type, Expr)> {
    let (env, idx) = type_check_expr(env, idx)?;
    // NOTE: We immediately run constant folding on the index expression to eliminate any simple
    // arithmetic on the bounds of a slice (e.g., 10-1), as we require them to be literal values.
    let idx = constant_fold::fold_expr(idx);
    let i = idx.get_info();
    let elem_ty = if let Type::Tensor {sz, shape} = target.get_type() {
        let ndims = num_index_dimensions(&idx);
        if ndims <= shape.len() {
            let idx_dims = extract_slice_dims(&idx, &shape[..])?;
            let res_shape = idx_dims.into_iter()
                .chain(shape.clone().into_iter().skip(ndims))
                .filter(|dim| *dim > 0)
                .collect::<Vec<i64>>();
            Ok(Type::Tensor {sz: sz.clone(), shape: res_shape})
        } else {
            let sh = shape.iter().map(|i| i.to_string()).join(",");
            py_type_error!(i, "Indexing with {ndims} dimensions on tensor of shape [{sh}]")
        }
    } else {
        py_type_error!(i, "Subscript operation on unsupported target {target}")
    }?;
    let expected_ty = match idx.get_type() {
        Type::Tensor {shape, ..} if shape.len() == 0 => {
            Ok(Type::Tensor {sz: ElemSize::I64, shape: vec![]})
        },
        Type::Tuple {elems} => {
            let expected_types = elems.iter()
                .map(|_| Type::Tensor {sz: ElemSize::I64, shape: vec![]})
                .collect::<Vec<Type>>();
            Ok(Type::Tuple {elems: expected_types})
        },
        ty => py_type_error!(i, "Unsupported index of type {ty} in subscript operation")
    }?;
    let idx = coerce_type(&env, idx, &expected_ty)?;
    Ok((env, elem_ty, idx))
}

pub fn type_check_expr(
    env: TypeCheckEnv,
    e: Expr
) -> PyResult<(TypeCheckEnv, Expr)> {
    match e {
        Expr::Var {id, i, ..} => {
            let ty = match env.vars.get(&id) {
                Some(ty) if ty != &Type::Unknown => Ok(ty.clone()),
                _ => py_type_error!(i, "Variable {id} has unknown type")
            }?;
            Ok((env, Expr::Var {id, ty, i}))
        },
        Expr::String {v, i, ..} => Ok((env, Expr::String {v, ty: Type::String, i})),
        Expr::Bool {v, i, ..} => {
            let ty = Type::Tensor {sz: ElemSize::Bool, shape: vec![]};
            Ok((env, Expr::Bool {v, ty, i}))
        },
        Expr::Int {v, i, ..} =>
            Ok((env, Expr::Int {v, ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]}, i})),
        Expr::Float {v, i, ..} => {
            let ty = Type::Tensor {sz: env.float_size.clone(), shape: vec![]};
            Ok((env, Expr::Float {v, ty, i}))
        },
        Expr::UnOp {op, arg, i, ..} => {
            let (env, arg) = type_check_expr(env, *arg)?;
            let ty = type_check_unop(&env, &op, &arg, &i)?;
            Ok((env, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
        },
        Expr::BinOp {lhs, op, rhs, i, ..} => {
            let (env, lhs) = type_check_expr(env, *lhs)?;
            let (env, rhs) = type_check_expr(env, *rhs)?;
            let (lhs, ty, rhs) = type_check_binop(&env, lhs, &op, rhs, &i)?;
            Ok((env, Expr::BinOp {lhs, op, rhs, ty, i}))
        },
        Expr::IfExpr {cond, thn, els, i, ..} => {
            let (env, cond) = type_check_expr(env, *cond)?;
            let ty = cond.get_type();
            if env.is_bool_scalar(ty) {
                let (env, thn) = type_check_expr(env, *thn)?;
                let (env, els) = type_check_expr(env, *els)?;
                let thn_ty = thn.get_type().clone();
                let els_ty = els.get_type().clone();
                let ty = lub_type(&env, thn_ty, els_ty, &i)?;
                let thn = Box::new(coerce_type(&env, thn, &ty)?);
                let els = Box::new(coerce_type(&env, els, &ty)?);
                Ok((env, Expr::IfExpr {cond: Box::new(cond), thn, els, ty, i}))
            } else {
                py_type_error!(i, "If expression has condition of invalid type {ty}")
            }
        },
        Expr::Subscript {target, idx, i, ..} => {
            let (env, target) = type_check_expr(env, *target)?;
            let (env, ty, idx) = match *idx {
                Expr::String {v, i, ..} => {
                    if let Type::Dict {fields} = target.get_type() {
                        if let Some(ty) = fields.get(&v) {
                            let idx_ty = Type::String;
                            Ok((env, ty.clone(), Expr::String {v, ty: idx_ty, i}))
                        } else {
                            py_type_error!(i, "Field {v} not present in {0}", target.get_type())
                        }
                    } else {
                        py_type_error!(i, "Cannot index using a string on non-dict expression")
                    }
                },
                idx => type_check_indexing(env, &target, idx)
            }?;
            Ok((env, Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i}))
        },
        Expr::Slice {lo, hi, i, ..} => {
            let type_check_boxed = |env, o: Option<Box<Expr>>| match o {
                Some(e) => {
                    let (env, e) = type_check_expr(env, *e)?;
                    Ok::<_, PyErr>((env, Some(Box::new(e))))
                },
                None => Ok((env, None))
            };
            let (env, lo) = type_check_boxed(env, lo)?;
            let (env, hi) = type_check_boxed(env, hi)?;
            // NOTE: The type of a slice expression is pointless since it can never be used outside
            // of a subscript operation, so we consider it an integer type to be consistent with
            // regular indices.
            let ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
            Ok((env, Expr::Slice {lo, hi, ty, i}))
        },
        Expr::Tuple {elems, i, ..} => {
            let (env, elems) = elems.smap_accum_l_result(Ok(env), type_check_expr)?;
            let elem_types = elems.iter()
                .map(|e| e.get_type().clone())
                .collect::<Vec<Type>>();
            let ty = Type::Tuple {elems: elem_types};
            Ok((env, Expr::Tuple {elems, ty, i}))
        },
        Expr::Call {id, args, i, ..} => {
            let (mut env, args) = args.smap_accum_l_result(Ok(env), type_check_expr)?;
            let arg_types = args.iter()
                .map(|e| e.get_type().clone())
                .collect::<Vec<Type>>();
            let arg_types = match env.arg_types.get(&id) {
                Some(tys) => {
                    let lub_types = arg_types.into_iter()
                        .zip(tys.clone().into_iter())
                        .map(|(lty, rty)| lub_type(&env, lty, rty, &i))
                        .collect::<PyResult<Vec<Type>>>();
                    match lub_types {
                        Ok(tys) => Ok(tys),
                        Err(e) => py_type_error!(i, "Found incompatible types in function call:\n{e}")
                    }
                },
                None => Ok(arg_types)
            }?;
            env.arg_types.insert(id.clone(), arg_types);
            let ty = match env.res_types.get(&id) {
                Some(ty) => ty.clone(),
                None => py_type_error!(i, "Unknown return type of function {id}")?
            };
            Ok((env, Expr::Call {id, args, ty, i}))
        },
        Expr::NeutralElement {op, tyof, i} => {
            let (env, tyof) = type_check_expr(env, *tyof)?;
            match tyof.get_type() {
                Type::Tensor {sz, ..} => {
                    match reduce::neutral_element(&op, &sz, &i) {
                        Some(ne) => Ok((env, ne)),
                        None => {
                            let op_str = op.pprint_default();
                            py_type_error!(i, "Failed to find neutral element \
                                               for operation {op_str}")
                        }
                    }
                },
                _ => {
                    py_type_error!(i, "Failed to find neutral element of \
                                       non-tensor value")
                }
            }
        },
        Expr::Builtin {func, args, axis, i, ..} => {
            let (env, args) = type_check_exprs(env, args)?;
            let builtin = type_check_builtin(&env, func, args, axis, i)?;
            Ok((env, builtin))
        },
        Expr::Convert {e, ty} => {
            let (env, e) = type_check_expr(env, *e)?;
            Ok((env, Expr::Convert {e: Box::new(e), ty}))
        },
    }
}

fn type_check_exprs(
    env: TypeCheckEnv,
    exprs: Vec<Expr>
) -> PyResult<(TypeCheckEnv, Vec<Expr>)> {
    exprs.smap_accum_l_result(Ok(env), type_check_expr)
}

fn validate_condition_type(cond: Expr, i: &Info) -> PyResult<Expr> {
    let ty = cond.get_type();
    match ty {
        Type::Tensor {..} => Ok(cond),
        _ => py_type_error!(i, "Unsupported type {ty} of conditional expression")
    }
}

fn type_check_stmt(
    env: TypeCheckEnv,
    stmt: Stmt
) -> PyResult<(TypeCheckEnv, Stmt)> {
    match stmt {
        Stmt::Definition {id, expr, labels, i, ..} => {
            let (mut env, expr) = type_check_expr(env, expr)?;
            let ty = expr.get_type().clone();
            env.vars.insert(id.clone(), ty.clone());
            Ok((env, Stmt::Definition {ty, id, expr, labels, i}))
        },
        Stmt::Assign {dst, expr, labels, i} => {
            let (env, dst) = type_check_expr(env, dst)?;
            let (env, expr) = type_check_expr(env, expr)?;
            let expr = coerce_type(&env, expr, dst.get_type())?;
            Ok((env, Stmt::Assign {dst, expr, labels, i}))
        },
        Stmt::For {var, lo, hi, step, body, labels, i} => {
            let (env, lo) = type_check_expr(env, lo)?;
            let lo = ensure_scalar_type(&env, lo, ElemSize::I64)?;
            let (mut env, hi) = type_check_expr(env, hi)?;
            let hi = ensure_scalar_type(&env, hi, ElemSize::I64)?;
            env.vars.insert(var.clone(), Type::Tensor {sz: ElemSize::I64, shape: vec![]});
            let (mut env, body) = type_check_stmts(env, body)?;
            env.vars.remove(&var);
            Ok((env, Stmt::For {var, lo, hi, step, body, labels, i}))
        },
        Stmt::While {cond, body, i} => {
            let (env, cond) = type_check_expr(env, cond)?;
            let cond = validate_condition_type(cond, &i)?;
            let (env, body) = type_check_stmts(env, body)?;
            Ok((env, Stmt::While {cond, body, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let (env, cond) = type_check_expr(env, cond)?;
            let cond = validate_condition_type(cond, &i)?;
            let (thn_env, thn) = type_check_stmts(env, thn)?;
            let (els_env, els) = type_check_stmts(thn_env, els)?;
            Ok((els_env, Stmt::If {cond, thn, els, i}))
        },
        Stmt::Return {value, i} => {
            let (mut env, value) = type_check_expr(env, value)?;
            env.res_ty = match &env.res_ty {
                Type::Void => Ok(value.get_type().clone()),
                ty => {
                    let r = lub_type(&env, value.get_type().clone(), ty.clone(), &i);
                    if r.is_err() {
                        py_type_error!(i, "Found conflicting types of return statements")
                    } else {
                        r
                    }
                },
            }?;
            Ok((env, Stmt::Return {value, i}))
        },
        Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Label {..} => {
            stmt.smap_accum_l_result(Ok(env), type_check_stmt)
        },
        Stmt::Call {func, i, ..} => {
            py_type_error!(i, "Call to function {func} should have been inlined (internal error)")
        },
    }
}

fn type_check_stmts(
    env: TypeCheckEnv,
    stmts: Vec<Stmt>
) -> PyResult<(TypeCheckEnv, Vec<Stmt>)> {
    stmts.smap_accum_l_result(Ok(env), type_check_stmt)
}

fn add_param_types<'py>(
    id: &Name,
    params: Vec<Param>,
    args: &Vec<Bound<'py, PyAny>>,
    float_size: &ElemSize
) -> PyResult<Vec<Param>> {
    if args.len() == params.len() {
        args.iter()
            .zip(params.into_iter())
            .map(|(arg, Param {id, i, ..})| {
                Ok(Param {id, ty: convert_type(&arg, float_size)?, i})
            })
            .collect::<PyResult<Vec<Param>>>()
    } else {
        py_type_error!(
            Info::default(),
            "Function {id} expected {0} arguments but received {1}",
            params.len(), args.len()
        )
    }
}

pub fn type_check_params<'py>(
    def: FunDef,
    args: &Vec<Bound<'py, PyAny>>,
    float_size: &ElemSize
) -> PyResult<FunDef> {
    let params = add_param_types(&def.id, def.params, args, float_size)?;
    Ok(FunDef {params, ..def})
}

fn type_check_def(
    mut env: TypeCheckEnv,
    def: FunDef
) -> PyResult<(TypeCheckEnv, FunDef)> {
    let params = match env.arg_types.get(def.id.get_str()) {
        Some(arg_tys) => {
            def.params.into_iter()
                .zip(arg_tys.iter())
                .map(|(Param {id, ty, i}, arg_ty)| {
                    let ty = lub_type(&env, ty, arg_ty.clone(), &i)?;
                    Ok(Param {id, ty, i})
                })
                .collect::<PyResult<Vec<Param>>>()
        },
        None => Ok(def.params)
    }?;
    // Ensure the types of all parameters are known at this point. If we find any parameter whose
    // type is unknown, the user is recommended to use explicit type annotations (not needed for
    // the "main" function).
    assert_known_params(&def.id, &params)?;

    // Before we type-check the body, we reset the return type to unknown and set the function
    // parameters as our known variables.
    env.res_ty = Type::Void;
    env.vars = params.iter()
        .map(|Param {id, ty, ..}| (id.clone(), ty.clone()))
        .collect::<BTreeMap<Name, Type>>();
    let (mut env, body) = type_check_stmts(env, def.body)?;

    let res_ty = env.res_ty.clone();
    env.res_types.insert(def.id.get_str().clone(), res_ty.clone());
    let def = FunDef {params, body, res_ty, ..def};
    Ok((env, def))
}

fn type_check_body_shapes(
    ast: Ast,
    shapes_only: bool,
    float_size: Option<ElemSize>
) -> PyResult<(TypeCheckEnv, Ast)> {
    let mut env = TypeCheckEnv::default();
    env.shapes_only = shapes_only;
    if let Some(sz) = float_size {
        env.float_size = sz;
    }

    // Collect the argument types of all functions before running.
    env.arg_types = ast.iter()
        .map(|FunDef {id, params, ..}| {
            let params = params.clone()
                .into_iter()
                .map(|Param {ty, ..}| ty.clone())
                .collect::<Vec<Type>>();
            (id.get_str().clone(), params)
        })
        .collect::<BTreeMap<String, Vec<Type>>>();

    // Type-check the AST nodes
    ast.smap_accum_l_result(Ok(env), type_check_def)
}

pub fn type_check_body(ast: Ast, float_size: ElemSize) -> PyResult<(TypeCheckEnv, Ast)> {
    type_check_body_shapes(ast, false, Some(float_size))
}

pub fn check_body_shape(ast: Ast) -> PyResult<(TypeCheckEnv, Ast)> {
    type_check_body_shapes(ast, true, None)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;
    use crate::utils::pprint::PrettyPrint;

    use std::ffi::CString;
    use strum::IntoEnumIterator;

    fn test_lub_elem_size_ok(lhs: &ElemSize, rhs: &ElemSize, expected: ElemSize) {
        let result = lub_elem_size(lhs, rhs, &Info::default());
        assert_eq!(expected, result.unwrap());
    }

    fn test_lub_elem_size_fail(lhs: &ElemSize, rhs: &ElemSize) {
        let result = lub_elem_size(lhs, rhs, &Info::default());
        assert!(result.is_err());
    }

    fn bool_type() -> Type {
        scalar(ElemSize::Bool)
    }

    #[test]
    fn is_bool_scalar() {
        assert!(TypeCheckEnv::default().is_bool_scalar(&bool_type()));
    }

    #[test]
    fn is_signed_int_scalar() {
        assert!(TypeCheckEnv::default().is_signed_int_scalar(&scalar(ElemSize::I16)));
    }

    #[test]
    fn is_unsigned_int_scalar() {
        assert!(TypeCheckEnv::default().is_unsigned_int_scalar(&scalar(ElemSize::U32)));
    }

    #[test]
    fn is_float_scalar() {
        assert!(TypeCheckEnv::default().is_float_scalar(&scalar(ElemSize::F16)));
    }

    #[test]
    fn is_scalar_tensor_fails() {
        assert!(!TypeCheckEnv::default().is_scalar(&shape(vec![10])));
    }

    #[test]
    fn is_arith_tensor_scalar_fails() {
        assert!(!TypeCheckEnv::default().is_arith_tensor(&scalar(ElemSize::Bool)));
    }

    #[test]
    fn is_arith_tensor_tensor() {
        assert!(TypeCheckEnv::default().is_arith_tensor(&shape(vec![10])));
    }

    #[test]
    fn is_arith_tensor_bool_tensor_fails() {
        let ty = Type::Tensor {sz: ElemSize::Bool, shape: vec![10]};
        assert!(!TypeCheckEnv::default().is_arith_tensor(&ty));
    }

    #[test]
    fn assert_known_type_unknown_fails() {
        assert!(assert_known_param_type(Ok(()), &id("x"), &tyuk()).is_err());
    }

    #[test]
    fn assert_known_type_nested_unknown_fails() {
        let tuple_ty = Type::Tuple {elems: vec![Type::String, Type::Unknown]};
        let mut fields = BTreeMap::new();
        fields.insert("x".to_string(), tuple_ty);
        let ty = Type::Dict { fields };
        assert!(assert_known_param_type(Ok(()), &id("y"), &ty).is_err());
    }

    #[test]
    fn assert_known_type_scalar() {
        assert!(assert_known_param_type(Ok(()), &id("x"), &scalar(ElemSize::I32)).is_ok());
    }

    #[test]
    fn known_params() {
        let params = vec![
            Param {id: id("x"), ty: scalar(ElemSize::I32), i: Info::default()},
            Param {id: id("y"), ty: scalar(ElemSize::F32), i: Info::default()},
        ];
        assert!(assert_known_params(&id("f"), &params).is_ok());
    }

    #[test]
    fn unknown_params() {
        let params = vec![
            Param {id: id("x"), ty: tyuk(), i: Info::default()},
            Param {id: id("y"), ty: scalar(ElemSize::F32), i: Info::default()},
            Param {id: id("z"), ty: tyuk(), i: Info::default()}
        ];
        assert_py_error_matches(
            assert_known_params(&id("f"), &params),
            r"Parameters x, z of function f have unknown type.*"
        );
    }

    fn try_convert_argument_type(s: &str, float_sz: ElemSize, expected_ty: Type) {
        let s = CString::new(s).unwrap();
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let arg = py.eval(&s, None, None).unwrap();
            let ty = convert_type(&arg, &float_sz).unwrap();
            assert_eq!(
                ty, expected_ty, "Expected type {0} but found {1}",
                expected_ty.pprint_default(), ty.pprint_default());
        });
    }

    #[test]
    fn convert_integer_literal_arg() {
        try_convert_argument_type("0", ElemSize::F32, scalar(ElemSize::I64));
    }

    #[test]
    fn convert_float_literal_arg_f16() {
        try_convert_argument_type("0.0", ElemSize::F16, scalar(ElemSize::F16));
    }

    #[test]
    fn convert_float_literal_arg_f32() {
        try_convert_argument_type("0.0", ElemSize::F32, scalar(ElemSize::F32));
    }

    #[test]
    fn convert_float_literal_arg_f64() {
        try_convert_argument_type("0.0", ElemSize::F64, scalar(ElemSize::F64));
    }

    #[test]
    fn convert_conforming_dict_arg() {
        let fields = vec![
            ("a".to_string(), scalar(ElemSize::I64)),
            ("b".to_string(), scalar(ElemSize::F32)),
        ].into_iter().collect::<BTreeMap<String, Type>>();
        let ty = Type::Dict {fields};
        try_convert_argument_type("{'a': 2, 'b': 4.5}", ElemSize::F32, ty);
    }

    #[test]
    #[should_panic]
    fn convert_list_arg_fails() {
        try_convert_argument_type("[1, 2, 3]", ElemSize::F32, shape(vec![3]));
    }

    #[test]
    fn lub_elem_size_identity() {
        for sz in ElemSize::iter() {
            test_lub_elem_size_ok(&sz, &sz, sz.clone())
        }
    }

    #[test]
    fn lub_elem_size_is_commutative() {
        for sz1 in ElemSize::iter() {
            for sz2 in ElemSize::iter() {
                let i = Info::default();
                let r1 = lub_elem_size(&sz1, &sz2, &i);
                let r2 = lub_elem_size(&sz2, &sz1, &i);
                assert!((r1.is_ok() && r2.is_ok()) || (r1.is_err() && r2.is_err()));
            }
        }
    }

    #[test]
    fn lub_elem_size_f32_f64() {
        test_lub_elem_size_ok(&ElemSize::F32, &ElemSize::F64, ElemSize::F64)
    }

    #[test]
    fn lub_elem_size_int_float() {
        test_lub_elem_size_fail(&ElemSize::I32, &ElemSize::F32)
    }

    fn test_lub_type_ok(lty: Type, rty: Type, expected: Type) {
        let env = TypeCheckEnv::default();
        let r = lub_type(&env, lty, rty, &Info::default());
        assert_eq!(expected, r.unwrap());
    }

    fn test_lub_type_fail(lty: Type, rty: Type) {
        let env = TypeCheckEnv::default();
        let r = lub_type(&env, lty, rty, &Info::default());
        assert!(r.is_err());
    }
    
    #[test]
    fn lub_type_string() {
        test_lub_type_ok(Type::String, Type::String, Type::String)
    }

    #[test]
    fn lub_type_elem_eq() {
        let ty = scalar(ElemSize::I16);
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_elem_compatible() {
        let ty1 = scalar(ElemSize::F32);
        let ty2 = scalar(ElemSize::F64);
        test_lub_type_ok(ty1.clone(), ty2.clone(), ty2.clone())
    }

    #[test]
    fn lub_type_elem_incompatible() {
        let ty1 = scalar(ElemSize::F32);
        let ty2 = scalar(ElemSize::I8);
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_bool_eq() {
        let ty = scalar(ElemSize::Bool);
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tensor_equal_ok() {
        let ty = Type::Tensor {sz: ElemSize::I32, shape: vec![5]};
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tensor_compatible_fails() {
        let ty1 = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let ty2 = Type::Tensor {sz: ElemSize::F64, shape: vec![5]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_tensor_different_shape_fails() {
        let ty1 = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let ty2 = Type::Tensor {sz: ElemSize::F32, shape: vec![4]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_tuple_eq_elems() {
        let ty = Type::Tuple {elems: vec![
            bool_type(),
            scalar(ElemSize::F32)
        ]};
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tuple_compatible_elems_fails() {
        let ty1 = Type::Tuple {elems: vec![scalar(ElemSize::F32)]};
        let ty2 = Type::Tuple {elems: vec![scalar(ElemSize::F64)]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_shapes_only() {
        let ty1 = Type::Tensor {sz: ElemSize::I32, shape: vec![5, 10]};
        let ty2 = Type::Tensor {sz: ElemSize::F64, shape: vec![10]};
        let mut env = TypeCheckEnv::default();
        env.shapes_only = true;
        match lub_type(&env, ty1, ty2, &Info::default()).unwrap() {
            Type::Tensor {shape, ..} => assert_eq!(shape, vec![5, 10]),
            _ => panic!("Unexpected result type")
        }
    }

    fn var(s: &str) -> Name {
        Name::new(s.to_string())
    }

    fn int(v: i64, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(Type::Unknown);
        Expr::Int {v: v as i128, ty, i: Info::default()}
    }

    fn test_tc_unop(op: UnOp, arg: Expr) -> PyResult<Type> {
        let env = TypeCheckEnv::default();
        type_check_unop(&env, &op, &arg, &Info::default())
    }

    #[test]
    fn type_check_unop_signed_int_negation() {
        let ty = scalar(ElemSize::I64);
        let arg = int(1, Some(ty.clone()));
        let res = test_tc_unop(UnOp::Sub, arg).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_unop_float_negation() {
        let ty = scalar(ElemSize::F32);
        let arg = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_unop(UnOp::Sub, arg).unwrap();
        assert_eq!(res, ty);
    }

    fn test_tc_binop(lhs: Expr, op: BinOp, rhs: Expr) -> PyResult<Type> {
        let env = TypeCheckEnv::default();
        let (_, ty, _) = type_check_binop(&env, lhs, &op, rhs, &Info::default())?;
        Ok(ty)
    }

    #[test]
    fn type_check_binop_signed_int_addition() {
        let ty = scalar(ElemSize::I64);
        let lhs = int(1, Some(ty.clone()));
        let rhs = int(2, Some(ty.clone()));
        let res = test_tc_binop(lhs, BinOp::Add, rhs).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_binop_coerced_signed_int_multiplication() {
        let lty = scalar(ElemSize::I32);
        let lhs = Expr::Var {id: var("x"), ty: lty.clone(), i: Info::default()};
        let rty = scalar(ElemSize::I16);
        let rhs = Expr::Var {id: var("y"), ty: rty, i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Mul, rhs).unwrap();
        assert_eq!(res, lty);
    }

    #[test]
    fn type_check_binop_float_subtraction() {
        let ty = scalar(ElemSize::F32);
        let lhs = Expr::Float {v: 3.14, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Sub, rhs).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_int_equality() {
        let ty = scalar(ElemSize::I16);
        let lhs = int(1, Some(ty.clone()));
        let rhs = int(2, Some(ty.clone()));
        let res = test_tc_binop(lhs, BinOp::Eq, rhs).unwrap();
        assert_eq!(res, bool_type());
    }

    #[test]
    fn type_check_float_lt() {
        let ty = scalar(ElemSize::F32);
        let lhs = Expr::Float {v: 2.718, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Lt, rhs).unwrap();
        assert_eq!(res, bool_type());
    }

    fn make_map<'a>(entries: Vec<(&'a str, Type)>) -> BTreeMap<Name, Type> {
        entries.into_iter()
            .map(|(id, ty)| (Name::new(id.to_string()), ty))
            .collect::<BTreeMap<Name, Type>>()
    }

    fn tc_expr(vars: BTreeMap<Name, Type>, e: Expr) -> PyResult<Expr> {
        let mut env = TypeCheckEnv::default();
        env.vars = vars;
        let (_ ,e) = type_check_expr(env, e)?;
        Ok(e)
    }

    #[test]
    fn type_check_expr_known_var() {
        let vars = make_map(vec![("x", bool_type())]);
        let v = Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()};
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), bool_type());
    }

    #[test]
    fn type_check_expr_unknown_var() {
        let vars = make_map(vec![]);
        let v = Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()};
        assert!(tc_expr(vars, v).is_err())
    }

    #[test]
    fn type_check_expr_string_literal() {
        let vars = make_map(vec![]);
        let v = Expr::String {v: "x".to_string(), ty: Type::Unknown, i: Info::default()};
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), Type::String);
    }

    #[test]
    fn type_check_expr_int_literal() {
        let vars = make_map(vec![]);
        let v = int(0, None);
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), scalar(ElemSize::I64));
    }

    #[test]
    fn type_check_expr_float_literal() {
        let vars = make_map(vec![]);
        let v = Expr::Float {v: 0.0, ty: Type::Unknown, i: Info::default()};
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), scalar(ElemSize::F64));
    }

    #[test]
    fn type_check_expr_dict_lookup() {
        let fields = vec![("a", bool_type())].into_iter()
            .map(|(id, ty)| (id.to_string(), ty))
            .collect::<BTreeMap<String, Type>>();
        let dict_ty = Type::Dict {fields};
        let vars = make_map(vec![("x", dict_ty.clone())]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(Expr::String {v: "a".to_string(), ty: Type::Unknown, i: Info::default()}),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, bool_type());
            assert_eq!(target.get_type().clone(), dict_ty);
            assert_eq!(idx.get_type().clone(), Type::String);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_lookup() {
        let tensor_ty = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let vars = make_map(vec![("x", tensor_ty.clone())]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(int(0, None)),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = tc_expr(vars, v);
        assert!(r.is_ok());
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, scalar(ElemSize::F32));
            assert_eq!(target.get_type().clone(), tensor_ty.clone());
            assert_eq!(idx.get_type().clone(), scalar(ElemSize::I64));
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_lookup_with_conversion() {
        let tensor_ty = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let vars = make_map(vec![
            ("x", tensor_ty.clone()),
            ("y", scalar(ElemSize::I32))
        ]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(Expr::Var {id: var("y"), ty: Type::Unknown, i: Info::default()}),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = tc_expr(vars, v);
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, scalar(ElemSize::F32));
            assert_eq!(target.get_type().clone(), tensor_ty);
            // As the variable y is a 32-bit integer, the type-checker should insert a Convert node
            // indicating that it needs to be converted to a 64-bit signed integer value (we always
            // expect this for indexing operations).
            if let Expr::Convert {e, ty} = *idx {
                assert_eq!(e.get_type().clone(), scalar(ElemSize::I32));
                assert_eq!(ty, scalar(ElemSize::I64));
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_return_stmt() {
        let env = TypeCheckEnv::default();
        let ret = Stmt::Return {
            value: Expr::Int {
                v: 1, ty: Type::Unknown, i: Info::default()
            },
            i: Info::default()
        };
        assert!(type_check_stmt(env, ret).is_ok());
    }

    #[test]
    fn type_check_conflicting_return() {
        let env = TypeCheckEnv::default();
        let conds = Stmt::If {
            cond: Expr::Bool {v: true, ty: Type::Unknown, i: Info::default()},
            thn: vec![Stmt::Return {
                value: Expr::Int {
                    v: 1, ty: Type::Unknown, i: Info::default()
                },
                i: Info::default()
            }],
            els: vec![Stmt::Return {
                value: Expr::Bool {
                    v: false, ty: Type::Unknown, i: Info::default()
                },
                i: Info::default()
            }],
            i: Info::default()
        };
        assert!(type_check_stmt(env, conds).is_err());
    }

    #[test]
    fn return_value() {
        let ast = vec![FunDef {
            id: var("x"),
            params: vec![],
            body: vec![Stmt::Return {
                value: Expr::Int {
                    v: 1, ty: Type::Unknown, i: Info::default()
                },
                i: Info::default()
            }],
            res_ty: Type::Unknown,
            i: Info::default()}
        ];
        assert!(type_check_body(ast, ElemSize::F64).is_ok());
    }

    fn test_slicing(
        target_shape: Vec<i64>,
        idx_args: Vec<Expr>,
        result_shape: Vec<i64>
    ) -> () {
        let target_ty = Type::Tensor {sz: ElemSize::F32, shape: target_shape};
        let vars = make_map(vec![
            ("x", target_ty.clone())
        ]);
        let nidxs = idx_args.len();
        let idx = Box::new(Expr::Tuple {
            elems: idx_args,
            ty: Type::Unknown,
            i: Info::default()
        });
        let subscript = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx, ty: Type::Unknown, i: Info::default()
        };
        let idx_ty = Type::Tuple {
            elems: (0..nidxs).into_iter()
                .map(|_| scalar(ElemSize::I64)).collect::<Vec<Type>>()
        };
        let result_ty = Type::Tensor {sz: ElemSize::F32, shape: result_shape};
        let r = tc_expr(vars, subscript);
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, result_ty);
            assert_eq!(target_ty, target.get_type().clone());
            assert_eq!(idx_ty, idx.get_type().clone());
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_partial_index() {
        let target_shape = vec![5,6,4];
        let idx_args = vec![
            int(2, None),
            int(5, None),
        ];
        let result_shape = vec![4];
        test_slicing(target_shape, idx_args, result_shape);
    }

    fn slice(lo: Option<i64>, hi: Option<i64>) -> Expr {
        let lo = lo.map(|i| Box::new(int(i, None)));
        let hi = hi.map(|i| Box::new(int(i, None)));
        Expr::Slice {lo, hi, ty: Type::Unknown, i: Info::default()}
    }

    #[test]
    fn type_check_slice_index() {
        let target_shape = vec![7,4,3];
        let idx_args = vec![
            slice(Some(2), Some(7)),
            int(3, None)
        ];
        let result_shape = vec![5, 3];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    fn unspec_slice_index() {
        let target_shape = vec![10];
        let idx_args = vec![slice(None, None)];
        let result_shape = vec![10];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    fn high_dimensional_slicing() {
        let target_shape = vec![5,8,6,3,5,4];
        let idx_args = vec![
            int(2, None),
            slice(Some(1), Some(7)),
            int(2, None),
            slice(None, Some(3)),
            int(4, None)
        ];
        let result_shape = vec![6,3,4];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    #[should_panic]
    fn out_of_bounds_access() {
        let target_shape = vec![5,8];
        let idx_args = vec![int(4, None), int(9, None)];
        let result_shape = vec![];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    fn negative_index_access() {
        let target_shape = vec![5,8];
        let idx_args = vec![int(-1, None)];
        let result_shape = vec![8];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    #[should_panic]
    fn slice_negative_lower_bound_fails() {
        let target_shape = vec![5,8];
        let idx_args = vec![slice(Some(-1), None)];
        let result_shape = vec![5];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    #[should_panic]
    fn slice_beyond_end() {
        let target_shape = vec![5,8];
        let idx_args = vec![slice(Some(3), Some(6))];
        let result_shape = vec![3,8];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    #[should_panic]
    fn slice_longer_than_target() {
        let target_shape = vec![5,8];
        let idx_args = vec![slice(Some(1), Some(7))];
        let result_shape = vec![6,8];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    fn negative_literal_index() {
        let target_shape = vec![5,8];
        let idx_args = vec![
            Expr::UnOp {
                op: UnOp::Sub, arg: Box::new(int(1, None)), ty: Type::Unknown,
                i: Info::default()
            }
        ];
        let result_shape = vec![8];
        test_slicing(target_shape, idx_args, result_shape);
    }

    #[test]
    #[should_panic]
    fn reject_non_literal_slice() {
        let target_shape = vec![5,8];
        let idx_args = vec![
            Expr::Slice {
                lo: None,
                hi: Some(Box::new(Expr::UnOp {
                    op: UnOp::Sub, arg: Box::new(int(1, None)), ty: Type::Unknown,
                    i: Info::default()
                })),
                ty: Type::Unknown,
                i: Info::default()
            }
        ];
        let result_shape = vec![8];
        test_slicing(target_shape, idx_args, result_shape);
    }
}
