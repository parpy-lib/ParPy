use crate::utils::info::InfoNode;

use strum_macros::EnumIter;
use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum ElemSize {
    #[default] Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64
}

impl ElemSize {
    pub fn is_boolean(&self) -> bool {
        match self {
            ElemSize::Bool => true,
            _ => false
        }
    }

    pub fn is_signed_integer(&self) -> bool {
        match self {
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => true,
            _ => false
        }
    }

    pub fn is_unsigned_integer(&self) -> bool {
        match self {
            ElemSize::U8 | ElemSize::U16 | ElemSize::U32 | ElemSize::U64 => true,
            _ => false
        }
    }

    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub fn is_floating_point(&self) -> bool {
        match self {
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => true,
            _ => false
        }
    }
}

impl fmt::Display for ElemSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElemSize::Bool => write!(f, "bool"),
            ElemSize::I8 => write!(f, "int8"),
            ElemSize::I16 => write!(f, "int16"),
            ElemSize::I32 => write!(f, "int32"),
            ElemSize::I64 => write!(f, "int64"),
            ElemSize::U8 => write!(f, "uint8"),
            ElemSize::U16 => write!(f, "uint16"),
            ElemSize::U32 => write!(f, "uint32"),
            ElemSize::U64 => write!(f, "uint64"),
            ElemSize::F16 => write!(f, "float16"),
            ElemSize::F32 => write!(f, "float32"),
            ElemSize::F64 => write!(f, "float64"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOp {
    #[default] Sub, Not, BitNeg, Addressof, Exp, Log, Cos, Sin, Sqrt, Tanh, Abs
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOp {
    #[default] Add, Sub, Mul, FloorDiv, Div, Rem, Pow, And, Or,
    BitAnd, BitOr, BitXor, BitShl, BitShr,
    Eq, Neq, Leq, Geq, Lt, Gt,
    Max, Min, Atan2
}

impl BinOp {
    fn prec_idx(&self) -> usize {
        match self {
            BinOp::Or => 2,
            BinOp::And => 3,
            BinOp::BitOr => 4,
            BinOp::BitXor => 5,
            BinOp::BitAnd => 6,
            BinOp::Eq | BinOp::Neq => 7,
            BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => 8,
            BinOp::BitShl | BinOp::BitShr => 10,
            BinOp::Add | BinOp::Sub => 11,
            BinOp::Mul | BinOp::FloorDiv | BinOp::Div | BinOp::Rem => 12,
            BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => 20
        }
    }

    pub fn precedence(l: &BinOp, r: &BinOp) -> Ordering {
        l.prec_idx().cmp(&r.prec_idx())
    }
}

pub trait ExprType<T>: InfoNode {
    fn get_type<'a>(&'a self) -> &'a T;
    fn is_leaf_node(&self) -> bool;
}
