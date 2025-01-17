use crate::info::*;

use strum_macros::EnumIter;

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, EnumIter)]
pub enum ElemSize {
    I8, I16, I32, I64, U8, F16, F32, F64
}

impl ElemSize {
    pub fn is_signed_integer(&self) -> bool {
        match self {
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => true,
            _ => false
        }
    }

    pub fn is_unsigned_integer(&self) -> bool {
        match self {
            ElemSize::U8 => true,
            _ => false
        }
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
            ElemSize::I8 => write!(f, "int8"),
            ElemSize::I16 => write!(f, "int16"),
            ElemSize::I32 => write!(f, "int32"),
            ElemSize::I64 => write!(f, "int64"),
            ElemSize::U8 => write!(f, "uint8"),
            ElemSize::F16 => write!(f, "float16"),
            ElemSize::F32 => write!(f, "float32"),
            ElemSize::F64 => write!(f, "float64"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Boolean,
    String,
    Scalar {sz: ElemSize},
    Tensor {sz: ElemSize},
    Tuple {elems: Vec<Type>},
    Dict {fields: HashMap<String, Type>},
    Unknown
}

impl Type {
    pub fn is_signed_integer(&self) -> bool {
        match self {
            Type::Scalar {sz} => sz.is_signed_integer(),
            _ => false
        }
    }

    pub fn is_unsigned_integer(&self) -> bool {
        match self {
            Type::Scalar {sz} => sz.is_unsigned_integer(),
            _ => false
        }
    }

    pub fn is_floating_point(&self) -> bool {
        match self {
            Type::Scalar {sz} => sz.is_floating_point(),
            _ => false
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Boolean => write!(f, "boolean"),
            Type::String => write!(f, "string"),
            Type::Scalar {sz} => write!(f, "{sz}"),
            Type::Tensor {sz} => write!(f, "tensor[{sz}]"),
            Type::Unknown => write!(f, "?"),
            Type::Tuple {elems} => {
                let elems = elems.iter()
                    .map(|e| format!("{e}"))
                    .collect::<Vec<String>>()
                    .join(",");
                write!(f, "({elems})")
            },
            Type::Dict {fields} => {
                let fields = fields.iter()
                    .map(|(k, v)| format!("{k} {v}"))
                    .collect::<Vec<String>>()
                    .join(",");
                write!(f, "dict {{{fields}}}")
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Builtin {
    Exp, Inf, Log, Max, Min, Sum
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Builtin::Exp => write!(f, "exp"),
            Builtin::Inf => write!(f, "inf"),
            Builtin::Log => write!(f, "log"),
            Builtin::Max => write!(f, "max"),
            Builtin::Min => write!(f, "min"),
            Builtin::Sum => write!(f, "sum"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnOp {
    Sub
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnOp::Sub => write!(f, "-")
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, FloorDiv, Div, Mod, BitAnd, Eq, Neq, Lt, Gt
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::FloorDiv => write!(f, "//"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: String, ty: Type, i: Info},
    String {v: String, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Subscript {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Tuple {elems: Vec<Expr>, ty: Type, i: Info},
    Dict {fields: HashMap<String, Expr>, ty: Type, i: Info},
    Builtin {func: Builtin, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},
}

impl Expr {
    pub fn get_type(&self) -> Type {
        match self {
            Expr::Var {ty, ..} => ty.clone(),
            Expr::String {ty, ..} => ty.clone(),
            Expr::Int {ty, ..} => ty.clone(),
            Expr::Float {ty, ..} => ty.clone(),
            Expr::UnOp {ty, ..} => ty.clone(),
            Expr::BinOp {ty, ..} => ty.clone(),
            Expr::Subscript {ty, ..} => ty.clone(),
            Expr::Tuple {ty, ..} => ty.clone(),
            Expr::Dict {ty, ..} => ty.clone(),
            Expr::Builtin {ty, ..} => ty.clone(),
            Expr::Convert {ty, ..} => ty.clone(),
        }
    }

    pub fn with_type(self, ty: Type) -> Self {
        match self {
            Expr::Var {id, i, ..} => Expr::Var {id, ty, i},
            Expr::String {v, i, ..} => Expr::String {v, ty, i},
            Expr::Int {v, i, ..} => Expr::Int {v, ty, i},
            Expr::Float {v, i, ..} => Expr::Float {v, ty, i},
            Expr::UnOp {op, arg, i, ..} => Expr::UnOp {op, arg, ty, i},
            Expr::BinOp {lhs, op, rhs, i, ..} => Expr::BinOp {lhs, op, rhs, ty, i},
            Expr::Subscript {target, idx, i, ..} => Expr::Subscript {target, idx, ty, i},
            Expr::Tuple {elems, i, ..} => Expr::Tuple {elems, ty, i},
            Expr::Dict {fields, i, ..} => Expr::Dict {fields, ty, i},
            Expr::Builtin {func, args, i, ..} => Expr::Builtin {func, args, ty, i},
            Expr::Convert {e, ..} => Expr::Convert {e, ty},
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Var {id, ..} => write!(f, "{id}"),
            Expr::String {v, ..} => write!(f, "\"{v}\""),
            Expr::Int {v, ..} => write!(f, "{v}"),
            Expr::Float {v, ..} => write!(f, "{v}"),
            Expr::UnOp {op, arg, ..} => write!(f, "{op}{arg}"),
            Expr::BinOp {lhs, op, rhs, ..} => write!(f, "({lhs} {op} {rhs})"),
            Expr::Subscript {target, idx, ..} => write!(f, "{target}[{idx}]"),
            Expr::Tuple {elems, ..} => {
                let elems = elems.iter()
                    .map(|e| format!("{e}"))
                    .collect::<Vec<String>>()
                    .join(",");
                write!(f, "({elems})")
            },
            Expr::Dict {fields, ..} => {
                let fields = fields.iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<String>>()
                    .join(",");
                write!(f, "{{{fields}}}")
            },
            Expr::Builtin {func, args, ..} => {
                if args.is_empty() {
                    write!(f, "{func}")
                } else {
                    let args = args.iter()
                        .map(|a| format!("{a}"))
                        .collect::<Vec<String>>()
                        .join(",");
                    write!(f, "{func}({args})")
                }
            },
            Expr::Convert {e, ty} => {
                write!(f, "({ty}){e}")
            },
        }
    }
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} => i.clone(),
            Expr::String {i, ..} => i.clone(),
            Expr::Int {i, ..} => i.clone(),
            Expr::Float {i, ..} => i.clone(),
            Expr::UnOp {i, ..} => i.clone(),
            Expr::BinOp {i, ..} => i.clone(),
            Expr::Subscript {i, ..} => i.clone(),
            Expr::Tuple {i, ..} => i.clone(),
            Expr::Dict {i, ..} => i.clone(),
            Expr::Builtin {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign {dst: Expr, expr: Expr, i: Info},
    For {var: String, lo: Expr, hi: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: String,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug)]
pub struct FunDef {
    pub id: String,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub i: Info
}

pub type Ast = Vec<FunDef>;
