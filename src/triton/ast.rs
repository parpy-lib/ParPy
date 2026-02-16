use crate::utils::ast::ExprType;
use crate::utils::info::{Info, InfoNode};
use crate::utils::name::Name;
use crate::utils::smap::*;

pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::utils::ast::BinOp;
pub use crate::gpu::ast::Dim;
pub use crate::gpu::ast::Dim3;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Pointer {sz: ElemSize},
    Tensor {sz: ElemSize},
    Void,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReduceOp {
    Min, Max, Sum, Prod
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
    ExtCall {id: String, args: Vec<Expr>, ty: Type, i: Info},

    // Triton-specific nodes
    ProgramId {dim: Dim, ty: Type, i: Info},
    Arange {lo: usize, hi: usize, ty: Type, i: Info},
    Load {ptr: Box<Expr>, mask: Option<Box<Expr>>, ty: Type, i: Info},
    Store {ptr: Box<Expr>, value: Box<Expr>, mask: Option<Box<Expr>>, ty: Type, i: Info},
    Full {shape: i64, value: Box<Expr>, elem_sz: ElemSize, ty: Type, i: Info},
    Where {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} |
            Expr::Bool {i, ..} |
            Expr::Int {i, ..} |
            Expr::Float {i, ..} |
            Expr::UnOp {i, ..} |
            Expr::BinOp {i, ..} |
            Expr::Reduce {i, ..} |
            Expr::Call {i, ..} |
            Expr::ExtCall {i, ..} |
            Expr::ProgramId {i, ..} |
            Expr::Arange {i, ..} |
            Expr::Load {i, ..} |
            Expr::Store {i, ..} |
            Expr::Full {i, ..} |
            Expr::Where {i, ..} => i.clone(),
        }
    }
}

impl ExprType<Type> for Expr {
    fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} |
            Expr::Bool {ty, ..} |
            Expr::Int {ty, ..} |
            Expr::Float {ty, ..} |
            Expr::UnOp {ty, ..} |
            Expr::BinOp {ty, ..} |
            Expr::Reduce {ty, ..} |
            Expr::Call {ty, ..} |
            Expr::ExtCall {ty, ..} |
            Expr::ProgramId {ty, ..} |
            Expr::Arange {ty, ..} |
            Expr::Load {ty, ..} |
            Expr::Store {ty, ..} |
            Expr::Full {ty, ..} |
            Expr::Where {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ProgramId {..} => true,
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::Reduce {..} |
            Expr::Call {..} | Expr::ExtCall {..} |
            Expr::Arange {..} | Expr::Load {..} | Expr::Store {..} |
            Expr::Full {..} | Expr::Where {..} => false,
        }
    }
}

impl SFold<Expr> for Expr {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Expr::UnOp {arg, ..} => f(acc?, arg),
            Expr::BinOp {lhs, rhs, ..} => f(f(acc?, lhs)?, rhs),
            Expr::Reduce {arg, ..} => f(acc?, arg),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::ExtCall {args, ..} => args.sfold_result(acc, &f),
            Expr::Load {ptr, mask, ..} => mask.sfold_result(f(acc?, ptr), &f),
            Expr::Store {ptr, value, mask, ..} => {
                mask.sfold_result(f(f(acc?, ptr)?, value), &f)
            },
            Expr::Full {value, ..} => f(acc?, value),
            Expr::Where {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::ProgramId {..} |
            Expr::Arange {..} => acc
        }
    }
}

impl SMapAccum<Expr> for Expr {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Self), E> {
        match self {
            Expr::UnOp {op, arg, ty, i} => {
                let (acc, arg) = f(acc?, *arg)?;
                Ok((acc, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::BinOp {lhs, op, rhs, ty, i} => {
                let (acc, lhs) = f(acc?, *lhs)?;
                let (acc, rhs) = f(acc, *rhs)?;
                Ok((acc, Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}))
            },
            Expr::Reduce {op, arg, ty, i} => {
                let (acc, arg) = f(acc?, *arg)?;
                Ok((acc, Expr::Reduce {op, arg: Box::new(arg), ty, i}))
            },
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::ExtCall {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::ExtCall {id, args, ty, i}))
            },
            Expr::Load {ptr, mask, ty, i} => {
                let (acc, ptr) = f(acc?, *ptr)?;
                let (acc, mask) = mask.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Expr::Load {ptr: Box::new(ptr), mask, ty, i}))
            },
            Expr::Store {ptr, value, mask, ty, i} => {
                let (acc, ptr) = f(acc?, *ptr)?;
                let (acc, value) = f(acc, *value)?;
                let (acc, mask) = mask.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Expr::Store {ptr: Box::new(ptr), value: Box::new(value), mask, ty, i}))
            },
            Expr::Full {shape, value, elem_sz, ty, i} => {
                let (acc, value) = f(acc?, *value)?;
                Ok((acc, Expr::Full {shape, value: Box::new(value), elem_sz, ty, i}))
            },
            Expr::Where {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::Where {
                    cond: Box::new(cond),
                    thn: Box::new(thn),
                    els: Box::new(els),
                    ty,
                    i
                }))
            },
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::ProgramId {..} |
            Expr::Arange {..} => Ok((acc?, self))
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
    KernelLaunch {id: Name, block_dims: Dim3, args: Vec<Expr>, nwarps: usize, i: Info},
}

impl SFold<Stmt> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Stmt) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Stmt::For {body, ..} => body.sfold_result(acc, &f),
            Stmt::While {body, ..} => body.sfold_result(acc, &f),
            Stmt::If {thn, els, ..} => els.sfold_result(thn.sfold_result(acc, &f), &f),
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Barrier {..} |
            Stmt::KernelLaunch {..} => acc,
        }
    }
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Self), E> {
        match self {
            Stmt::Assign {dst, expr, i} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Assign {dst, expr, i}))
            },
            Stmt::For {var, lo, hi, step, body, i} => {
                let (acc, lo) = f(acc?, lo)?;
                let (acc, hi) = f(acc, hi)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::Return {value, i} => {
                let (acc, value) = f(acc?, value)?;
                Ok((acc, Stmt::Return {value, i}))
            },
            Stmt::Expr {e, i} => {
                let (acc, e) = f(acc?, e)?;
                Ok((acc, Stmt::Expr {e, i}))
            },
            Stmt::Barrier {..} => Ok((acc?, self)),
            Stmt::KernelLaunch {id, block_dims, args, nwarps, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::KernelLaunch {id, block_dims, args, nwarps, i}))
            },
        }
    }
}

impl SMapAccum<Stmt> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Stmt) -> Result<(A, Stmt), E>
    ) -> Result<(A, Self), E> {
        match self {
            Stmt::For {var, lo, hi, step, body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, thn) = thn.smap_accum_l_result(acc, &f)?;
                let (acc, els) = els.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Barrier {..} |
            Stmt::KernelLaunch {..} => {
                Ok((acc?, self))
            },
        }
    }
}

impl SFlatten<Stmt> for Stmt {
    fn sflatten_result<E>(
        self,
        mut acc: Vec<Stmt>,
        f: impl Fn(Vec<Stmt>, Stmt) -> Result<Vec<Stmt>, E>
    ) -> Result<Vec<Stmt>, E> {
        match self {
            Stmt::For {var, lo, hi, step, body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::For {var, lo, hi, step, body, i});
            },
            Stmt::While {cond, body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::While {cond, body, i});
            },
            Stmt::If {cond, thn, els, i} => {
                let thn = thn.sflatten_result(vec![], &f)?;
                let els = els.sflatten_result(vec![], &f)?;
                acc.push(Stmt::If {cond, thn, els, i});
            },
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Barrier {..} |
            Stmt::KernelLaunch {..} => {
                acc.push(self);
            },
        };
        Ok(acc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    Import {package: String, as_str: Option<String>, i: Info},
    FunDef {
        triton_jit: bool,
        id: Name,
        params: Vec<Name>,
        body: Vec<Stmt>,
        i: Info
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub tops: Vec<Top>
}

impl SMapAccum<Top> for Ast {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Top) -> Result<(A, Top), E>
    ) -> Result<(A, Self), E> {
        let (acc, tops) = self.tops.smap_accum_l_result(acc, f)?;
        Ok((acc, Ast {tops}))
    }
}
