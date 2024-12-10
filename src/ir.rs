
use pyo3::prelude::*;

// Implementation of types for controlling parallelism. These types are exposed directly to the
// Python API.
#[pyclass]
#[derive(Clone, Debug)]
pub enum ParKind {
    GpuGrid(),
    GpuThreads(isize)
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ParSpec {
    target : String,
    kind : ParKind
}

#[pymethods]
impl ParSpec {
    #[new]
    pub fn new(target : String, kind : ParKind) -> Self {
        ParSpec { target, kind }
    }
}

///////////
// TYPES //
///////////
#[derive(Clone, Debug, PartialEq)]
pub enum IntSize {
    I8, I16, I32, I64, Any
}

#[derive(Clone, Debug, PartialEq)]
pub enum FloatSize {
    F16, F32, F64, Any
}

#[derive(Clone, Debug)]
pub enum Type {
    Bool(),
    Int(IntSize),
    Float(FloatSize),
    Array(Box<Type>)
}

#[derive(Clone, Debug)]
pub struct TypedParam {
    pub id : String,
    pub ty : Type
}

///////////////////////
// BINARY OPERATIONS //
///////////////////////
#[derive(Debug)]
pub enum BinOp {
    Add, Mul
}

/////////////////
// EXPRESSIONS //
/////////////////
#[derive(Debug)]
pub enum Expr {
    Var {id : String},
    LiteralInt {value : i64, ty : Type},
    LiteralFloat {value : f64, ty : Type},
    BinOp {lhs : Box<Expr>, op : BinOp, rhs : Box<Expr>},
    ArrayAccess {target : Box<Expr>, idx : Box<Expr>}
}

////////////////
// STATEMENTS //
////////////////
#[derive(Debug)]
pub enum Stmt {
    Assign {dst : Expr, e : Expr},
    For {var : String, lo : Expr, hi : Expr, body : Vec<Stmt>}
}

///////////////////////////
// TOP-LEVEL DEFINITIONS //
///////////////////////////
#[derive(Debug)]
pub enum Def {
    ParFun {id : String, params : Vec<TypedParam>, body : Vec<Stmt>},
    FunInst {id : String, par : Vec<ParSpec>}
}

// The IR AST is represented by a vector of top-level definitions.
pub type Program = Vec<Def>;
