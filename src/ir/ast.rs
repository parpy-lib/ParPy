use crate::par;

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum IntSize {
    I8, I16, I32, I64, Any
}

#[derive(Clone, Debug)]
pub enum FloatSize {
    F16, F32, F64, Any
}

#[derive(Clone, Debug)]
pub enum Type {
    Int(IntSize),
    Float(FloatSize),
    Tensor(Box<Type>),
    Unknown
}

#[derive(Clone, Debug)]
pub enum BinOp {
    Add, Mul, Lt
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id : String, ty : Type},
    Int {v : i64, ty : Type},
    Float {v : f64, ty : Type},
    BinOp {lhs : Box<Expr>, op : BinOp, rhs : Box<Expr>, ty : Type},
    Subscript {target : Box<Expr>, idx : Box<Expr>, ty : Type}
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
            Expr::Var {id, ..} => Expr::Var {id, ty},
            Expr::Int {v, ..} => Expr::Int {v, ty},
            Expr::Float {v, ..} => Expr::Float {v, ty},
            Expr::BinOp {lhs, op, rhs, ..} => Expr::BinOp {lhs, op, rhs, ty},
            Expr::Subscript {target, idx, ..} => Expr::Subscript {target, idx, ty}
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Assign {dst : Expr, e : Expr},
    For {
        var_ty : Type,
        var : String,
        init : Expr,
        cond : Expr,
        incr : Expr,
        body : Vec<Stmt>
    }
}

#[derive(Clone, Debug)]
pub struct TypedParam {
    pub id : String,
    pub ty : Type
}

#[derive(Clone, Debug)]
pub enum Def {
    FunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>},
    ParFunInst {id : String, par : HashMap<String, Vec<par::ParKind>>}
}

pub type Ast = Vec<Def>;
