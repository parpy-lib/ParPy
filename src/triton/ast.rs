use crate::utils::ast::ExprType;
use crate::utils::info::{Info, InfoNode};
use crate::utils::name::Name;
use crate::utils::smap::*;

pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::utils::ast::BinOp;
pub use crate::gpu::ast::Dim;
pub use crate::gpu::ast::Dim3;

use std::collections::BTreeMap;
use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Var(usize),
    Num(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Pointer {ty: Box<Type>, shape: Shape},
    Tensor {sz: ElemSize, shape: Shape},
    Function {result: Box<Type>, args: Vec<Type>},
    Void,
}

impl Type {
    pub fn get_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Pointer {ty, ..} => ty.get_elem_size(),
            Type::Tensor {sz, ..} => Some(sz),
            Type::Function {..} => None,
            Type::Void => None,
        }
    }

    pub fn get_shape<'a>(&'a self) -> Option<&'a Shape> {
        match self {
            Type::Pointer {shape, ..} => Some(shape),
            Type::Tensor {shape, ..} => Some(shape),
            Type::Function {..} => None,
            Type::Void => None,
        }
    }

    pub fn is_blocked(&self) -> bool {
        match self.get_shape() {
            Some(Shape::Num(n)) => *n > 1,
            Some(_) => false,
            None => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReduceOp {
    Min, Max, Sum, Prod, Any
}

#[derive(Clone, Debug)]
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
    Arange {lo: Box<Expr>, hi: Box<Expr>, ty: Type, i: Info},
    Load {ptr: Box<Expr>, mask: Option<Box<Expr>>, ty: Type, i: Info},
    Full {shape: usize, value: Box<Expr>, elem_sz: ElemSize, ty: Type, i: Info},
    Where {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    Convert {value: Box<Expr>, ty: Type, i: Info},

    // Host-side nodes
    AllocBuffer {nelems: usize, elem_sz: ElemSize, ty: Type, i: Info},
    ToTorch {e: Box<Expr>, ty: Type, i: Info},
}

impl Expr {
    pub fn with_type(self, ty: Type) -> Self {
        match self {
            Expr::Var {id, ty: _, i} => Expr::Var {id, ty, i},
            Expr::Bool {v, ty: _, i} => Expr::Bool {v, ty, i},
            Expr::Int {v, ty: _, i} => Expr::Int {v, ty, i},
            Expr::Float {v, ty: _, i} => Expr::Float {v, ty, i},
            Expr::UnOp {op, arg, ty: _, i} => Expr::UnOp {op, arg, ty, i},
            Expr::BinOp {lhs, op, rhs, ty: _, i} => Expr::BinOp {lhs, op, rhs, ty, i},
            Expr::Reduce {op, arg, ty: _, i} => Expr::Reduce {op, arg, ty, i},
            Expr::Call {id, args, ty: _, i} => Expr::Call {id, args, ty, i},
            Expr::ExtCall {id, args, ty: _, i} => Expr::ExtCall {id, args, ty, i},
            Expr::ProgramId {dim, ty: _, i} => Expr::ProgramId {dim, ty, i},
            Expr::Arange {lo, hi, ty: _, i} => Expr::Arange {lo, hi, ty, i},
            Expr::Load {ptr, mask, ty: _, i} => Expr::Load {ptr, mask, ty, i},
            Expr::Full {shape, value, elem_sz, ty: _, i} =>
                Expr::Full {shape, value, elem_sz, ty, i},
            Expr::Where {cond, thn, els, ty: _, i} =>
                Expr::Where {cond, thn, els, ty, i},
            Expr::Convert {value, ty: _, i} => Expr::Convert {value, ty, i},
            Expr::AllocBuffer {nelems, elem_sz, ty: _, i} =>
                Expr::AllocBuffer {nelems, elem_sz, ty, i},
            Expr::ToTorch {e, ty: _, i} => Expr::ToTorch {e, ty, i},
        }
    }

    pub fn discriminator(&self) -> u8 {
        match self {
            Expr::Var {..} => 0,
            Expr::Bool {..} => 1,
            Expr::Int {..} => 2,
            Expr::Float {..} => 3,
            Expr::UnOp {..} => 4,
            Expr::BinOp {..} => 5,
            Expr::Reduce {..} => 6,
            Expr::Call {..} => 7,
            Expr::ExtCall {..} => 8,
            Expr::ProgramId {..} => 9,
            Expr::Arange {..} => 10,
            Expr::Load {..} => 11,
            Expr::Full {..} => 12,
            Expr::Where {..} => 13,
            Expr::Convert {..} => 14,
            Expr::AllocBuffer {..} => 15,
            Expr::ToTorch {..} => 16,
        }
    }
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
            Expr::Full {i, ..} |
            Expr::Where {i, ..} |
            Expr::Convert {i, ..} |
            Expr::AllocBuffer {i, ..} |
            Expr::ToTorch {i, ..} => i.clone(),
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
            Expr::Full {ty, ..} |
            Expr::Where {ty, ..} |
            Expr::Convert {ty, ..} |
            Expr::AllocBuffer {ty, ..} |
            Expr::ToTorch {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::ProgramId {..} |
            Expr::Arange {..} |
            Expr::AllocBuffer {..} => true,
            Expr::UnOp {..} |
            Expr::BinOp {..} |
            Expr::Reduce {..} |
            Expr::Call {..} |
            Expr::ExtCall {..} |
            Expr::Load {..} |
            Expr::Full {..} |
            Expr::Where {..} |
            Expr::Convert {..} |
            Expr::ToTorch {..} => false,
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
            Expr::Arange {lo, hi, ..} => f(f(acc?, lo)?, hi),
            Expr::Load {ptr, mask, ..} => mask.sfold_result(f(acc?, ptr), &f),
            Expr::Full {value, ..} => f(acc?, value),
            Expr::Where {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::Convert {value, ..} => f(acc?, value),
            Expr::ToTorch {e, ..} => f(acc?, e),
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::ProgramId {..} |
            Expr::AllocBuffer {..} => acc
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
            Expr::Arange {lo, hi, ty, i} => {
                let (acc, lo) = f(acc?, *lo)?;
                let (acc, hi) = f(acc, *hi)?;
                Ok((acc, Expr::Arange {lo: Box::new(lo), hi: Box::new(hi), ty, i}))
            },
            Expr::Load {ptr, mask, ty, i} => {
                let (acc, ptr) = f(acc?, *ptr)?;
                let (acc, mask) = mask.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Expr::Load {ptr: Box::new(ptr), mask, ty, i}))
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
            Expr::Convert {value, ty, i} => {
                let (acc, value) = f(acc?, *value)?;
                Ok((acc, Expr::Convert {
                    value: Box::new(value), ty, i
                }))
            },
            Expr::ToTorch {e, ty, i} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::ToTorch {e: Box::new(e), ty, i}))
            },
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::ProgramId {..} |
            Expr::AllocBuffer {..} => Ok((acc?, self))
        }
    }
}

impl Ord for Expr {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Expr::Var {id: lid, ..}, Expr::Var {id: rid, ..}) => lid.cmp(rid),
            (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) => lv.cmp(rv),
            (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => lv.cmp(rv),
            (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) => f64::total_cmp(lv, rv),
            (Expr::UnOp {op: lop, arg: larg, ..}, Expr::UnOp {op: rop, arg: rarg, ..}) =>
                lop.cmp(rop).then(larg.cmp(rarg)),
            ( Expr::BinOp {lhs: llhs, op: lop, rhs: lrhs, ..}
            , Expr::BinOp {lhs: rlhs, op: rop, rhs: rrhs, ..} ) =>
                llhs.cmp(rlhs).then(lop.cmp(rop)).then(lrhs.cmp(rrhs)),
            ( Expr::Reduce {op: lop, arg: larg, ..}
            , Expr::Reduce {op: rop, arg: rarg, ..} ) =>
                lop.cmp(rop).then(larg.cmp(rarg)),
            ( Expr::Call {id: lid, args: largs, ..}
            , Expr::Call {id: rid, args: rargs, ..} ) =>
                lid.cmp(rid).then(largs.cmp(rargs)),
            ( Expr::ExtCall {id: lid, args: largs, ..}
            , Expr::ExtCall {id: rid, args: rargs, ..} ) =>
                lid.cmp(rid).then(largs.cmp(rargs)),
            (Expr::ProgramId {dim: ldim, ..}, Expr::ProgramId {dim: rdim, ..}) =>
                ldim.cmp(rdim),
            ( Expr::Arange {lo: llo, hi: lhi, ..},
              Expr::Arange {lo: rlo, hi: rhi, ..} ) =>
                llo.cmp(rlo).then(lhi.cmp(rhi)),
            ( Expr::Load {ptr: lptr, mask: lmask, ..},
              Expr::Load {ptr: rptr, mask: rmask, ..} ) =>
                lptr.cmp(rptr).then(lmask.cmp(rmask)),
            ( Expr::Full {shape: lshape, value: lvalue, ..}
            , Expr::Full {shape: rshape, value: rvalue, ..} ) =>
                lshape.cmp(rshape).then(lvalue.cmp(rvalue)),
            ( Expr::Where {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::Where {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.cmp(rcond).then(lthn.cmp(rthn)).then(lels.cmp(rels)),
            (Expr::Convert {value: lv, ..}, Expr::Convert {value: rv, ..}) => lv.cmp(rv),
            ( Expr::AllocBuffer {nelems: ln, elem_sz: lsz, ..}
            , Expr::AllocBuffer {nelems: rn, elem_sz: rsz, ..} ) =>
                ln.cmp(rn).then(lsz.cmp(rsz)),
            (Expr::ToTorch {e: le, ..}, Expr::ToTorch {e: re, ..}) => le.cmp(re),
            _ => self.discriminator().cmp(&other.discriminator()),
        }
    }
}

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Expr {}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {dst: Name, expr: Expr, i: Info},
    Assign {dst: Name, expr: Expr, i: Info},
    For {var: Name, lo: Expr, hi: Expr, step: i128, body: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    Return {value: Expr, i: Info},
    Expr {e: Expr, i: Info},
    Pass {i: Info},

    // Triton-specific nodes
    Barrier {i: Info},
    Store {ptr: Expr, value: Expr, mask: Option<Expr>, i: Info},
    KernelLaunch {id: Name, block_dims: Dim3, args: Vec<Expr>, nwarps: usize, i: Info},
}

impl SFold<Expr> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Stmt::Definition {expr, ..} => f(acc?, expr),
            Stmt::Assign {expr, ..} => f(acc?, expr),
            Stmt::For {lo, hi, ..} => f(f(acc?, hi)?, lo),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::Return {value, ..} => f(acc?, value),
            Stmt::Expr {e, ..} => f(acc?, e),
            Stmt::Pass {..} => acc,
            Stmt::Barrier {..} => acc,
            Stmt::Store {ptr, value, mask, ..} => {
                let acc = f(f(acc?, ptr)?, value)?;
                match mask {
                    Some(m) => f(acc, m),
                    None => Ok(acc)
                }
            },
            Stmt::KernelLaunch {args, ..} => args.sfold_result(acc, &f),
        }
    }
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
            Stmt::Definition {..} |
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Pass {..} |
            Stmt::Barrier {..} |
            Stmt::Store {..} |
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
            Stmt::Definition {dst, expr, i} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Definition {dst, expr, i}))
            },
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
            Stmt::Pass {..} |
            Stmt::Barrier {..} => Ok((acc?, self)),
            Stmt::Store {ptr, value, mask, i} => {
                let (acc, ptr) = f(acc?, ptr)?;
                let (acc, value) = f(acc, value)?;
                let (acc, mask) = match mask {
                    Some(m) => {
                        let (acc, m) = f(acc, m)?;
                        Ok((acc, Some(m)))
                    },
                    None => Ok((acc, None))
                }?;
                Ok((acc, Stmt::Store {ptr, value, mask, i}))
            },
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
            Stmt::Definition {..} |
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Pass {..} |
            Stmt::Barrier {..} |
            Stmt::Store {..} |
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
            Stmt::Definition {..} |
            Stmt::Assign {..} |
            Stmt::Return {..} |
            Stmt::Expr {..} |
            Stmt::Pass {..} |
            Stmt::Barrier {..} |
            Stmt::Store {..} |
            Stmt::KernelLaunch {..} => {
                acc.push(self);
            },
        };
        Ok(acc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AutotuneConfig {
    pub mapping: BTreeMap<Name, Expr>,
    pub warp_count: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Decorator {
    Autotune {configs: Vec<AutotuneConfig>, keys: Vec<String>},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    Import {package: String, as_str: Option<String>, i: Info},
    KernelFunDef {
        decorators: Vec<Decorator>,
        id: Name,
        params: Vec<Param>,
        body: Vec<Stmt>,
        i: Info
    },
    FunDef {
        id: Name,
        params: Vec<Param>,
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
