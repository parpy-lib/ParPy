use crate::utils::info::*;
use crate::utils::name::Name;

pub use crate::ir::ast::ElemSize;
pub use crate::ir::ast::Type;
pub use crate::ir::ast::Builtin;
pub use crate::ir::ast::UnOp;
pub use crate::ir::ast::BinOp;
pub use crate::ir::ast::Expr;
pub use crate::ir::ast::Field;
pub use crate::ir::ast::Param;

// TODO: Need to add more expressions, including...
// * threadIdx, blockIdx, blockDim
// * 
//
// May also want to consider a different kind of expression for the init, cond, and incr values,
// since they require different kinds of expression-style values.

pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {init: Expr, cond: Expr, incr: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
}

pub enum Attribute {
    Global, Host
}

pub enum Top {
    StructDef {id: Name, fields: Vec<Field>, i: Info},
    FunDef {attr: Attribute, id: Name, params: Vec<Param>, body: Vec<Stmt>, i: Info},
}

pub type Ast = Vec<Top>;
