use crate::info::*;
use crate::name::Name;

// Reuse the definition of element sizes from the Python AST.
pub use crate::py::ast::ElemSize;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Boolean,
    Tensor {sz: ElemSize, shape: Vec<i64>},
    Struct {id: Name},
}

// Reuse the below enums from the Python AST.
pub use crate::py::ast::Builtin;
pub use crate::py::ast::UnOp;
pub use crate::py::ast::BinOp;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    String {v: String, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    TensorAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
    Builtin {func: Builtin, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},
}

impl Expr {
    pub fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::String {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::StructFieldAccess {ty, ..} => ty,
            Expr::TensorAccess {ty, ..} => ty,
            Expr::Struct {ty, ..} => ty,
            Expr::Builtin {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReductionOp {
    Add,
    Mul,
    Min,
    Max,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LoopProperty {
    Threads {n: i64},
    ParallelReduction {op: ReductionOp, ne: Expr, ty: Type},
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Declaration {ty: Type, id: Name, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {var: Name, lo: Expr, hi: Expr, body: Vec<Stmt>, par: Vec<LoopProperty>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    StructDef {id: Name, fields: Vec<Field>, i: Info},
    FunDef {id: Name, params: Vec<Param>, body: Vec<Stmt>, i: Info},
}

pub type Ast = Vec<Top>;
