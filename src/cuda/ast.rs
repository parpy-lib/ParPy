use crate::utils::info::*;
use crate::utils::name::Name;

// Re-export nodes from the IR AST that we reuse as is.
pub use crate::ir::ast::ElemSize;
pub use crate::ir::ast::UnOp;
pub use crate::ir::ast::BinOp;

#[derive(Clone, Debug)]
pub enum Type {
    Void,
    Boolean,
    Scalar {sz: ElemSize},
    Pointer {sz: ElemSize},
    Struct {id: Name},
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Scalar {sz} => Some(sz),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dim {
    X, Y, Z
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // CUDA-specific nodes
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},

    // Built-in expression nodes
    Exp {arg: Box<Expr>, ty: Type, i: Info},
    Inf {ty: Type, i: Info},
    Log {arg: Box<Expr>, ty: Type, i: Info},
    Max {lhs: Box<Expr>, rhs: Box<Expr>, ty: Type, i: Info},
    Min {lhs: Box<Expr>, rhs: Box<Expr>, ty: Type, i: Info},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dim3 {
    pub x: i64,
    pub y: i64,
    pub z: i64
}

impl Dim3 {
    pub fn get_dim(&self, dim: Dim) -> i64 {
        match dim {
            Dim::X => self.x,
            Dim::Y => self.y,
            Dim::Z => self.z
        }
    }

    pub fn with_dim(self, dim: Dim, n: i64) -> Dim3 {
        match dim {
            Dim::X => Dim3 {x: n, ..self},
            Dim::Y => Dim3 {y: n, ..self},
            Dim::Z => Dim3 {z: n, ..self}
        }
    }

    pub fn product(&self) -> i64 {
        self.x * self.y * self.z
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Dim3 {x: 1, y: 1, z: 1}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchArgs {
    pub blocks: Dim3,
    pub threads: Dim3
}

impl LaunchArgs {
    pub fn with_blocks_dim(mut self, dim: Dim, n: i64) -> LaunchArgs {
        self.blocks = self.blocks.with_dim(dim, n);
        self
    }

    pub fn with_threads_dim(mut self, dim: Dim, n: i64) -> LaunchArgs {
        self.threads = self.threads.with_dim(dim, n);
        self
    }
}

impl Default for LaunchArgs {
    fn default() -> Self {
        LaunchArgs {blocks: Dim3::default(), threads: Dim3::default()}
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: i64, body: Vec<Stmt>, i: Info
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    Syncthreads {i: Info},
    KernelLaunch {id: Name, launch_args: LaunchArgs, args: Vec<Expr>, i: Info},
}

#[derive(Clone, Debug)]
pub enum Attribute {
    Global, Host
}

#[derive(Clone, Debug)]
pub struct Field {
    pub ty: Type,
    pub id: Name,
    pub i: Info
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug)]
pub enum Top {
    StructDef {id: Name, fields: Vec<Field>, i: Info},
    FunDef {
        attr: Attribute, ret_ty: Type, id: Name, params: Vec<Param>,
        body: Vec<Stmt>, i: Info
    },
}

pub type Ast = Vec<Top>;
