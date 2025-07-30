use crate::ir::ast as ir_ast;
use crate::py::ast::{BinOp, Builtin, ElemSize};
use crate::py::ast as py_ast;
use crate::gpu::ast as gpu_ast;
use crate::utils::info::*;

// Trait allowing us to construct an expression given a literal floating-point value and an element
// size determining its type.
pub trait ExprLit {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> Self where Self: Sized;

    fn to_int_lit(v: f64) -> i128 {
        v as i128
    }

    fn to_bool_lit(v: f64) -> bool {
        v != 0.0
    }
}

impl ExprLit for py_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> py_ast::Expr {
        let ty = py_ast::Type::Tensor {sz: sz.clone(), shape: vec![]};
        match sz {
            ElemSize::Bool => {
                py_ast::Expr::Bool {v: py_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 |
            ElemSize::U8 | ElemSize::U16 | ElemSize::U32 | ElemSize::U64 => {
                py_ast::Expr::Int {v: py_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                py_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for ir_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> ir_ast::Expr {
        let ty = ir_ast::Type::Tensor {sz: sz.clone(), shape: vec![]};
        match sz {
            ElemSize::Bool => {
                ir_ast::Expr::Bool {v: ir_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 |
            ElemSize::U8 | ElemSize::U16 | ElemSize::U32 | ElemSize::U64 => {
                ir_ast::Expr::Int {v: ir_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                ir_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for gpu_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> gpu_ast::Expr {
        let ty = gpu_ast::Type::Scalar {sz: sz.clone()};
        match sz {
            ElemSize::Bool => {
                gpu_ast::Expr::Bool {v: gpu_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 |
            ElemSize::U8 | ElemSize::U16 | ElemSize::U32 | ElemSize::U64 => {
                gpu_ast::Expr::Int {v: gpu_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                gpu_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

pub fn neutral_element<T: ExprLit>(
    op: &BinOp,
    sz: &ElemSize,
    i: &Info
) -> Option<T> {
    let i = i.clone();
    match op {
        BinOp::Add => Some(T::generate_literal(0.0, sz, i)),
        BinOp::Mul => Some(T::generate_literal(1.0, sz, i)),
        BinOp::Max => Some(T::generate_literal(f64::NEG_INFINITY, sz, i)),
        BinOp::Min => Some(T::generate_literal(f64::INFINITY, sz, i)),
        _ => None
    }
}

pub fn builtin_to_reduction_op(func: &Builtin) -> Option<BinOp> {
    match func {
        Builtin::Sum => Some(BinOp::Add),
        Builtin::Prod => Some(BinOp::Mul),
        Builtin::Max => Some(BinOp::Max),
        Builtin::Min => Some(BinOp::Min),
        _ => None
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::gpu::ast_builder as gpu;
    use crate::ir::ast_builder as ir;
    use crate::py::ast_builder as py;
    use strum::IntoEnumIterator;

    #[test]
    fn py_float_to_int_lit() {
        assert_eq!(py_ast::Expr::to_int_lit(2.0), 2);
    }

    #[test]
    fn py_float_to_bool_lit() {
        assert_eq!(py_ast::Expr::to_bool_lit(1.0), true);
    }

    #[test]
    fn py_convert_to_literal_expr() {
        for sz in ElemSize::iter() {
            let e = py_ast::Expr::generate_literal(1.0, &sz, i());
            if sz.is_integer() {
                assert_eq!(e, py::int(1, Some(sz)));
            } else if sz.is_floating_point() {
                assert_eq!(e, py::float(1.0, Some(sz)));
            } else {
                assert_eq!(e, py::bool_expr(true, Some(sz)));
            }
        }
    }

    #[test]
    fn ir_convert_to_literal_expr() {
        for sz in ElemSize::iter() {
            let e = ir_ast::Expr::generate_literal(1.0, &sz, i());
            if sz.is_integer() {
                assert_eq!(e, ir::int(1, Some(sz)));
            } else if sz.is_floating_point() {
                assert_eq!(e, ir::float(1.0, Some(sz)));
            } else {
                assert_eq!(e, ir::bool_expr(true));
            }
        }
    }

    #[test]
    fn gpu_convert_to_literal_expr() {
        for sz in ElemSize::iter() {
            let e = gpu_ast::Expr::generate_literal(1.0, &sz, i());
            if sz.is_integer() {
                assert_eq!(e, gpu::int(1, Some(sz)));
            } else if sz.is_floating_point() {
                assert_eq!(e, gpu::float(1.0, Some(sz)));
            } else {
                assert_eq!(e, gpu::bool_expr(true));
            }
        }
    }
}
