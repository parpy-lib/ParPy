use crate::utils::info::*;
use crate::utils::name::Name;

// Re-export nodes from the IR AST that we reuse as is.
pub use crate::ir::ast::ElemSize;

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

#[derive(Clone, Debug, PartialEq)]
pub enum UnOp {
    Sub, Exp, Log
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem, BoolAnd, BitAnd, Eq, Neq, Lt, Gt,
    Max, Min
}

impl BinOp {
    pub fn is_infix(&self) -> bool {
        match self {
            BinOp::Max | BinOp::Min => false,
            _ => true
        }
    }
}

impl BinOp {
    pub fn precedence(&self) -> usize {
        match self {
            BinOp::BoolAnd => 0,
            BinOp::BitAnd => 1,
            BinOp::Eq | BinOp::Neq => 2,
            BinOp::Lt | BinOp::Gt => 3,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
            BinOp::Max | BinOp::Min => 6,
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
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // CUDA-specific nodes
    ShflXorSync {value: Box<Expr>, idx: Box<Expr>, ty : Type, i: Info},
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
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
            Expr::ArrayAccess {ty, ..} => ty,
            Expr::Struct {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
            Expr::ShflXorSync {ty, ..} => ty,
            Expr::ThreadIdx {ty, ..} => ty,
            Expr::BlockIdx {ty, ..} => ty,
        }
    }

    pub fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} |
            Expr::BlockIdx {..} => true,
            _ => false
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
            Expr::ArrayAccess {i, ..} => i.clone(),
            Expr::Struct {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
            Expr::ShflXorSync {i, ..} => i.clone(),
            Expr::ThreadIdx {i, ..} => i.clone(),
            Expr::BlockIdx {i, ..} => i.clone(),
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        match (self, other) {
            (Expr::Var {id: lid, ..}, Expr::Var {id: rid, ..}) => lid.eq(rid),
            (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) => lv.eq(rv),
            (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => lv.eq(rv),
            (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) => lv.eq(rv),
            ( Expr::UnOp {op: lop, arg: larg, ..}
            , Expr::UnOp {op: rop, arg: rarg, ..} ) =>
                lop.eq(rop) && larg.eq(rarg),
            ( Expr::BinOp {lhs: llhs, op: lop, rhs: lrhs, ..}
            , Expr::BinOp {lhs: rlhs, op: rop, rhs: rrhs, ..} ) =>
                llhs.eq(rlhs) && lop.eq(rop) && lrhs.eq(rrhs),
            ( Expr::StructFieldAccess {target: ltarget, label: llabel, ..}
            , Expr::StructFieldAccess {target: rtarget, label: rlabel, ..} ) =>
                ltarget.eq(rtarget) && llabel.eq(rlabel),
            ( Expr::ArrayAccess {target: ltarget, idx: lidx, ..}
            , Expr::ArrayAccess {target: rtarget, idx: ridx, ..} ) =>
                ltarget.eq(rtarget) && lidx.eq(ridx),
            ( Expr::Struct {id: lid, fields: lfields, ..}
            , Expr::Struct {id: rid, fields: rfields, ..} ) =>
                lid.eq(rid) && lfields.eq(rfields),
            (Expr::Convert {e: le, ..}, Expr::Convert {e: re, ..}) => le.eq(re),
            ( Expr::ShflXorSync {value: lval, idx: lidx, ..}
            , Expr::ShflXorSync {value: rval, idx: ridx, ..} ) => {
                lval.eq(rval) && lidx.eq(ridx)
            },
            (Expr::ThreadIdx {dim: ldim, ..}, Expr::ThreadIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (Expr::BlockIdx {dim: ldim, ..}, Expr::BlockIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (_, _) => false
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dim3 {
    pub x: i64,
    pub y: i64,
    pub z: i64
}

impl Dim3 {
    pub fn get_dim(&self, dim: &Dim) -> i64 {
        match dim {
            Dim::X => self.x,
            Dim::Y => self.y,
            Dim::Z => self.z
        }
    }

    pub fn with_dim(self, dim: &Dim, n: i64) -> Dim3 {
        match dim {
            Dim::X => Dim3 {x: n, ..self},
            Dim::Y => Dim3 {y: n, ..self},
            Dim::Z => Dim3 {z: n, ..self}
        }
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
    pub fn with_blocks_dim(mut self, dim: &Dim, n: i64) -> LaunchArgs {
        self.blocks = self.blocks.with_dim(dim, n);
        self
    }

    pub fn with_threads_dim(mut self, dim: &Dim, n: i64) -> LaunchArgs {
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
    Definition {ty: Type, id: Name, expr: Expr},
    Assign {dst: Expr, expr: Expr},
    AllocShared {ty: Type, id: Name, sz: i64},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: Expr, body: Vec<Stmt>
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>},
    Syncthreads {},
    Dim3Definition {id: Name, args: Dim3},
    KernelLaunch {id: Name, blocks: Name, threads: Name, args: Vec<Expr>},
    Scope {body: Vec<Stmt>},
}

#[derive(Clone, Debug)]
pub enum Attribute {
    Global, Entry
}

#[derive(Clone, Debug)]
pub struct Field {
    pub ty: Type,
    pub id: String,
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
    Include {header: String},
    StructDef {id: Name, fields: Vec<Field>},
    FunDef {
        attr: Attribute, ret_ty: Type, id: Name, params: Vec<Param>,
        body: Vec<Stmt>
    },
}

pub type Ast = Vec<Top>;
