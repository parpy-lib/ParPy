use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

// Reuse the definition of element sizes from the Python AST.
pub use crate::py::ast::ElemSize;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Boolean,
    Scalar {sz: ElemSize},
    Tensor {sz: ElemSize, shape: Vec<i64>},
    Struct {id: Name},
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnOp {
    Sub, Exp, Log, Cos, Sin, Sqrt, Tanh, Abs
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem, Pow, And, Or, BitAnd, Eq, Neq, Leq, Geq, Lt, Gt,
    Max, Min, Atan2
}

impl BinOp {
    pub fn precedence(&self) -> usize {
        match self {
            BinOp::And | BinOp::Or => 0,
            BinOp::BitAnd => 1,
            BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => 2,
            BinOp::Eq | BinOp::Neq => 3,
            BinOp::Add | BinOp::Sub => 4,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 5,
            BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => 6,
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
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    TensorAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
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
            Expr::Convert {e, ..} => e.get_info(),
        }
    }
}

impl SMapAccum<Expr> for Expr {
    fn smap_accum_l<A>(self, f: impl Fn(A, Expr) -> (A, Expr), acc: A) -> (A, Expr) {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} => {
                (acc, self)
            },
            Expr::UnOp {op, arg, ty, i} => {
                let (acc, arg) = f(acc, *arg);
                (acc, Expr::UnOp {op, arg: Box::new(arg), ty, i})
            },
            Expr::BinOp {lhs, op, rhs, ty, i} => {
                let (acc, lhs) = f(acc, *lhs);
                let (acc, rhs) = f(acc, *rhs);
                (acc, Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
            }
            Expr::StructFieldAccess {target, label, ty, i} => {
                let (acc, target) = f(acc, *target);
                (acc, Expr::StructFieldAccess {target: Box::new(target), label, ty, i})
            },
            Expr::TensorAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc, *target);
                let (acc, idx) = f(acc, *idx);
                (acc, Expr::TensorAccess {target: Box::new(target), idx: Box::new(idx), ty, i})
            },
            Expr::Struct {id, fields, ty, i} => {
                let (acc, fields) = fields.into_iter()
                    .fold((acc, vec![]), |(acc, mut fields), (id, e)| {
                        let (acc, e) = f(acc, e);
                        fields.push((id, e));
                        (acc, fields)
                    });
                (acc, Expr::Struct {id, fields, ty, i})
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc, *e);
                (acc, Expr::Convert {e: Box::new(e), ty})
            },
        }
    }
}

impl SFold<Expr> for Expr {
    fn sfold<A>(&self, f: impl Fn(A, &Expr) -> A, acc: A) -> A {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} => acc,
            Expr::UnOp {arg, ..} => f(acc, arg),
            Expr::BinOp {lhs, rhs, ..} => f(f(acc, lhs), rhs),
            Expr::StructFieldAccess {target, ..} => f(acc, target),
            Expr::TensorAccess {target, idx, ..} => f(f(acc, target), idx),
            Expr::Struct {fields, ..} => {
                fields.into_iter().fold(acc, |acc, (_, e)| f(acc, e))
            },
            Expr::Convert {e, ..} => f(acc, e),
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

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l<A>(self, f: impl Fn(A, Expr) -> (A, Expr), acc: A) -> (A, Stmt) {
        match self {
            Stmt::Definition {ty, id, expr, i} => {
                let (acc, expr) = f(acc, expr);
                (acc, Stmt::Definition {ty, id, expr, i})
            },
            Stmt::Assign {dst, expr, i} => {
                let (acc, dst) = f(acc, dst);
                let (acc, expr) = f(acc, expr);
                (acc, Stmt::Assign {dst, expr, i})
            },
            Stmt::For {var, lo, hi, body, par, i} => {
                let (acc, lo) = f(acc, lo);
                let (acc, hi) = f(acc, hi);
                (acc, Stmt::For {var, lo, hi, body, par, i})
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, cond) = f(acc, cond);
                (acc, Stmt::If {cond, thn, els, i})
            },
            Stmt::While {cond, body, i} => {
                let (acc, cond) = f(acc, cond);
                (acc, Stmt::While {cond, body, i})
            },
        }
    }
}

impl SFold<Expr> for Stmt {
    fn sfold<A>(&self, f: impl Fn(A, &Expr) -> A, acc: A) -> A {
        match self {
            Stmt::Definition {expr, ..} => f(acc, expr),
            Stmt::Assign {dst, expr, ..} => f(f(acc, dst), expr),
            Stmt::For {lo, hi, ..} => f(f(acc, lo), hi),
            Stmt::If {cond, ..} => f(acc, cond),
            Stmt::While {cond, ..} => f(acc, cond),
        }
    }
}

impl SMapAccum<Stmt> for Stmt {
    fn smap_accum_l<A>(self, f: impl Fn(A, Stmt) -> (A, Stmt), acc: A) -> (A, Stmt) {
        match self {
            Stmt::Definition {..} | Stmt::Assign {..} => (acc, self),
            Stmt::For {var, lo, hi, body, par, i} => {
                let (acc, body) = body.smap_accum_l(&f, acc);
                (acc, Stmt::For {var, lo, hi, body, par, i})
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, thn) = thn.smap_accum_l(&f, acc);
                let (acc, els) = els.smap_accum_l(&f, acc);
                (acc, Stmt::If {cond, thn, els, i})
            },
            Stmt::While {cond, body, i} => {
                let (acc, body) = body.smap_accum_l(&f, acc);
                (acc, Stmt::While {cond, body, i})
            },
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
