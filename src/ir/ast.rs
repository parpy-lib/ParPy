use crate::par;
use crate::info::*;

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntSize {
    I8, I16, I32, I64, Any
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FloatSize {
    F16, F32, F64, Any
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Int(IntSize),
    Float(FloatSize),
    IntTensor(IntSize),
    FloatTensor(FloatSize),
    Unknown
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id : String, ty : Type, i : Info},
    Int {v : i64, ty : Type, i : Info},
    Float {v : f64, ty : Type, i : Info},
    BinOp {lhs : Box<Expr>, op : BinOp, rhs : Box<Expr>, ty : Type, i : Info},
    Subscript {target : Box<Expr>, idx : Box<Expr>, ty : Type, i : Info}
}

impl Expr {
    pub fn get_type(&self) -> &Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::Subscript {ty, ..} => ty
        }
    }

    pub fn with_type(self, ty : Type) -> Self {
        match self {
            Expr::Var {id, i, ..} => Expr::Var {id, ty, i},
            Expr::Int {v, i, ..} => Expr::Int {v, ty, i},
            Expr::Float {v, i, ..} => Expr::Float {v, ty, i},
            Expr::BinOp {lhs, op, rhs, i, ..} => Expr::BinOp {lhs, op, rhs, ty, i},
            Expr::Subscript {target, idx, i, ..} => Expr::Subscript {target, idx, ty, i}
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign {dst : Expr, e : Expr, i : Info},
    For {var : String, lo : Expr, hi : Expr, body : Vec<Stmt>, i : Info},
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TypedParam {
    pub id : String,
    pub ty : Type,
    pub i : Info
}

#[derive(Clone, Debug)]
pub enum Def {
    FunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>, i : Info},
    ParFunInst {id : String, par : HashMap<String, Vec<par::ParKind>>, i : Info}
}

pub type Ast = Vec<Def>;
