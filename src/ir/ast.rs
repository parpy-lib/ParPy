use crate::utils::info::*;
use crate::utils::name::Name;

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
    Bool {v: bool, ty: Type, i: Info},
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
            Expr::Bool {ty, ..} => ty,
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

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} => i.clone(),
            Expr::Bool {i, ..} => i.clone(),
            Expr::Int {i, ..} => i.clone(),
            Expr::Float {i, ..} => i.clone(),
            Expr::UnOp {i, ..} => i.clone(),
            Expr::BinOp {i, ..} => i.clone(),
            Expr::StructFieldAccess {i, ..} => i.clone(),
            Expr::TensorAccess {i, ..} => i.clone(),
            Expr::Struct {i, ..} => i.clone(),
            Expr::Builtin {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoopParallelism {
    pub nthreads : i64,
    pub reduction : bool
}

impl LoopParallelism {
    pub fn with_threads(self, nthreads: i64) -> Self {
        LoopParallelism {nthreads, ..self}
    }

    pub fn with_reduction(self) -> Self {
        LoopParallelism {reduction: true, ..self}
    }

    pub fn is_parallel(&self) -> bool {
        self.nthreads > 1
    }
}

impl Default for LoopParallelism {
    fn default() -> Self {
        LoopParallelism {nthreads: 1, reduction: false}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {var: Name, lo: Expr, hi: Expr, body: Vec<Stmt>, par: LoopParallelism, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub id: String,
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
pub struct StructDef {
    pub id: Name,
    pub fields: Vec<Field>,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunDef {
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub structs: Vec<StructDef>,
    pub fun: FunDef,
}
