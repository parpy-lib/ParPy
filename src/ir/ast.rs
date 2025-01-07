use crate::info::*;

// Reuse definitions from the simplified Python AST representation.
pub use crate::py::ast::IntSize;
pub use crate::py::ast::FloatSize;
pub use crate::py::ast::Type;
pub use crate::py::ast::BinOp;
pub use crate::py::ast::Expr;
pub use crate::py::ast::TypedParam;

#[derive(Clone, Debug)]
pub struct LoopProperties {
    init: Option<Expr>,
    incr: i64,
    precond: Option<Expr>
}

impl Default for LoopProperties {
    fn default() -> LoopProperties {
        LoopProperties {
            init: None,
            incr: 1,
            precond: None
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Declaration {ty: Type, id: String, i: Info},
    Assignment {dst: Expr, e: Expr, i: Info},
    For {
        id: String,
        lo: Expr,
        hi: Expr,
        body: Vec<Stmt>,
        properties: LoopProperties,
        i: Info
    },
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Declaration {i, ..} => i.clone(),
            Stmt::Assignment {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone()
        }
    }
}

#[derive(Clone, Debug)]
pub enum Top {
    KernelDef {
        id: String,
        params: Vec<TypedParam>,
        body: Vec<Stmt>,
        nblocks: i64,
        nthreads: i64,
        i: Info
    }
}

pub type Ast = Vec<Top>;
