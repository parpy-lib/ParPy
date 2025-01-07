use crate::info::*;

// Reuse the binary operations defined in the Python IR.
pub use crate::py::ast::BinOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntSize {
    I8, I16, I32, I64
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FloatSize {
    F16, F32, F64
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Type {
    Int(IntSize),
    Float(FloatSize),
    IntTensor(IntSize),
    FloatTensor(FloatSize)
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id : String, ty : Type, i : Info},
    Int {v : i64, ty : Type, i : Info},
    Float {v : f64, ty : Type, i : Info},
    BinOp {lhs : Box<Expr>, op : BinOp, rhs : Box<Expr>, ty : Type, i : Info},
    Subscript {target : Box<Expr>, idx : Box<Expr>, ty : Type, i : Info}
}

#[derive(Clone, Debug, PartialEq)]
pub enum Dim {
    X, Y, Z
}

#[derive(Clone, Debug)]
pub enum IterKind {
    Sequential,
    GpuBlock(i64, Dim),
    GpuThread(i64, Dim),
}

impl Default for IterKind {
    fn default() -> Self {
        IterKind::Sequential
    }
}

#[derive(Clone, Debug)]
pub struct LoopProperties {
    iter_kind : IterKind,
    reduction : Option<(String, String)>
}

impl Default for LoopProperties {
    fn default() -> Self {
        LoopProperties {iter_kind: IterKind::default(), reduction: None}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dim3 {
    x: u64, y: u64, z: u64
}

impl Default for Dim3 {
    fn default() -> Self {
        Dim3 {x: 1, y: 1, z: 1}
    }
}

impl Dim3 {
    pub fn x(&self) -> u64 {
        self.x
    }

    pub fn y(&self) -> u64 {
        self.y
    }

    pub fn z(&self) -> u64 {
        self.z
    }

    pub fn with_x(self, x: u64) -> Self {
        Dim3 {x, y: self.y, z: self.z}
    }

    pub fn with_y(self, y: u64) -> Self {
        Dim3 {x: self.x, y, z: self.z}
    }

    pub fn with_z(self, z: u64) -> Self {
        Dim3 {x: self.x, y: self.y, z}
    }

    pub fn count(&self) -> u64 {
        self.x * self.y * self.z
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Decl {ty : Type, var : String, i : Info},
    AssignVar {var : String, e : Expr, i : Info},
    AssignArray {var : String, idx : Expr, e : Expr, i : Info},
    For {
        var : String,
        lo : Expr,
        hi : Expr,
        body : Vec<Stmt>,
        properties : LoopProperties,
        i : Info
    },
    DeviceCall {
        blocks : Dim3,
        threads : Dim3,
        id : String,
        args : Vec<Expr>,
        i : Info
    }
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Decl {i, ..} => i.clone(),
            Stmt::AssignVar {i, ..} => i.clone(),
            Stmt::AssignArray {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::DeviceCall {i, ..} => i.clone()
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
pub enum HostTop {
    FunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>}
}

#[derive(Clone, Debug)]
pub enum DeviceTop {
    KernelFunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>}
}

pub struct Ast {
    // Top-level definitions for the host
    pub host : Vec<HostTop>,

    // Top-level definitions for the device
    pub device : Vec<DeviceTop>
}

impl Ast {
    pub fn new() -> Self {
        Ast {host: vec![], device: vec![]}
    }
}
