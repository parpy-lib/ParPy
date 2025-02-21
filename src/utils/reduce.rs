use crate::py::ast::{BinOp, Builtin, ElemSize};
use crate::py::ast as py_ast;
use crate::cuda::ast as cu_ast;
use crate::utils::info::Info;

// Trait allowing us to construct an expression given a literal floating-point value and an element
// size determining its type.
pub trait ExprLit {
    fn generate_literal(v: f64, sz: ElemSize, i: Info) -> Self where Self: Sized;

    fn to_int_lit(v: f64) -> i64 {
        v as i64
    }
    fn to_bool_lit(v: f64) -> bool {
        v != 0.0
    }
}

impl ExprLit for py_ast::Expr {
    fn generate_literal(v: f64, sz: ElemSize, i: Info) -> py_ast::Expr {
        let ty = py_ast::Type::Tensor {sz: sz.clone(), shape: vec![]};
        match sz {
            ElemSize::Bool => {
                py_ast::Expr::Bool {v: py_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                py_ast::Expr::Int {v: py_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                py_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for cu_ast::Expr {
    fn generate_literal(v: f64, sz: ElemSize, i: Info) -> cu_ast::Expr {
        let ty = cu_ast::Type::Scalar {sz: sz.clone()};
        match sz {
            ElemSize::Bool => {
                cu_ast::Expr::Bool {v: cu_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                cu_ast::Expr::Int {v: cu_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                cu_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

pub fn reduction_op_neutral_element<T: ExprLit>(
    op: &BinOp,
    sz: ElemSize,
    i: Info
) -> Option<T> {
    match op {
        BinOp::Add => Some(T::generate_literal(0.0, sz, i)),
        BinOp::Mul => Some(T::generate_literal(1.0, sz, i)),
        BinOp::Max => Some(T::generate_literal(f64::NEG_INFINITY, sz, i)),
        BinOp::Min => Some(T::generate_literal(f64::INFINITY, sz, i)),
        BinOp::And => Some(T::generate_literal(1.0, sz, i)),
        BinOp::Or => Some(T::generate_literal(0.0, sz, i)),
        _ => None
    }
}

pub fn builtin_to_reduction_op(func: &Builtin) -> Option<BinOp> {
    match func {
        Builtin::Sum => Some(BinOp::Add),
        Builtin::Max => Some(BinOp::Max),
        Builtin::Min => Some(BinOp::Min),
        _ => None
    }
}
