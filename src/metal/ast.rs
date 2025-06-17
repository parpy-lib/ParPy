use crate::utils::info::*;
use crate::utils::name::Name;

// Re-use nodes from the GPU IR AST.
pub use crate::gpu::ast::ElemSize;
pub use crate::gpu::ast::UnOp;
pub use crate::gpu::ast::BinOp;
pub use crate::gpu::ast::Dim;
pub use crate::gpu::ast::Dim3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemSpace {
    Host, Device,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Void,
    Boolean,
    Scalar {sz: ElemSize},
    Pointer {ty: Box<Type>, mem: MemSpace},
    MetalBuffer
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Scalar {sz} => Some(sz),
            _ => None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Ternary {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // Metal-specific nodes
    SimdOp {op: BinOp, arg: Box<Expr>, ty: Type, i: Info},
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
}

impl Expr {
    pub fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} |
            Expr::BlockIdx {..} => true,
            _ => false
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr},
    Assign {dst: Expr, expr: Expr},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: Expr, body: Vec<Stmt>
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>},
    While {cond: Expr, body: Vec<Stmt>},
    ThreadgroupBarrier {},
    KernelLaunch {id: Name, blocks: Dim3, threads: Dim3, args: Vec<Expr>},
    SubmitWork {},

    // Statements related to memory management.
    AllocDevice {elem_ty: Type, id: Name, sz: usize},
    AllocThreadgroup {elem_ty: Type, id: Name, sz: usize},
    FreeDevice {id: Name},
    CopyMemory {
        elem_ty: Type, src: Expr, src_mem: MemSpace,
        dst: Expr, dst_mem: MemSpace, sz: usize
    },
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type
}

#[derive(Clone, Debug)]
pub struct MetalDef {
    pub maxthreads: usize,
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>
}

#[derive(Clone, Debug)]
pub struct HostDef {
   pub ret_ty: Type,
   pub id: Name,
   pub params: Vec<Param>,
   pub body: Vec<Stmt>
}

#[derive(Clone, Debug)]
pub struct Ast {
    pub includes: Vec<String>,
    pub metal_tops: Vec<MetalDef>,
    pub host_tops: Vec<HostDef>
}
