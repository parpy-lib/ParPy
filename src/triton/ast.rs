use crate::utils::ast::{BinOp, ElemSize, ExprType, UnOp};
use crate::utils::info::{Info, InfoNode};
use crate::utils::name::Name;

pub use crate::gpu::ast::Dim;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Tensor {shape: Vec<i64>, sz: ElemSize},
    Void,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReduceOp {
    Min, Max, Sum
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Reduce {op: ReduceOp, arg: Box<Expr>, ty: Type, i: Info},
    Call {id: Name, args: Vec<Expr>, ty: Type, i: Info},

    // Triton-specific nodes
    ProgramId {dim: Dim, ty: Type, i: Info},
    Arange {lo: usize, hi: usize, ty: Type, i: Info},
    Load {ptr: Box<Expr>, mask: Option<Box<Expr>>, ty: Type, i: Info},
    Store {ptr: Box<Expr>, value: Box<Expr>, mask: Option<Box<Expr>>, ty: Type, i: Info},
    Full {shape: Vec<i64>, value: Box<Expr>, elem_sz: ElemSize, ty: Type, i: Info},
    Where {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} | Expr::Bool {i, ..} | Expr::Int {i, ..} |
            Expr::Float {i, ..} | Expr::UnOp {i, ..} | Expr::BinOp {i, ..} |
            Expr::Reduce {i, ..} | Expr::Call {i, ..} | Expr::ProgramId {i, ..} |
            Expr::Arange {i, ..} | Expr::Load {i, ..} | Expr::Store {i, ..} |
            Expr::Full {i, ..} | Expr::Where {i, ..} => i.clone(),
        }
    }
}

impl ExprType<Type> for Expr {
    fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} | Expr::Bool {ty, ..} | Expr::Int {ty, ..} |
            Expr::Float {ty, ..} | Expr::UnOp {ty, ..} | Expr::BinOp {ty, ..} |
            Expr::Reduce {ty, ..} | Expr::Call {ty, ..} | Expr::ProgramId {ty, ..} |
            Expr::Arange {ty, ..} | Expr::Load {ty, ..} | Expr::Store {ty, ..} |
            Expr::Full {ty, ..} | Expr::Where {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ProgramId {..} => true,
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::Reduce {..} |
            Expr::Call {..} | Expr::Arange {..} | Expr::Load {..} |
            Expr::Store {..} | Expr::Full {..} | Expr::Where {..} => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign {dst: Name, expr: Expr, i: Info},
    For {var: Name, lo: Expr, hi: Expr, step: i64, body: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    Return {value: Expr, i: Info},
    Expr {e: Expr, i: Info},

    // Triton-specific nodes
    Barrier {i: Info},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Option<ElemSize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    Import {package: String, as_str: Option<String>, i: Info},
    TritonFunDef {id: Name, params: Vec<Param>, body: Vec<Stmt>, i: Info},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub tops: Vec<Top>,
}
