use crate::utils::ast::ExprType;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::utils::ast::BinOp;
pub use crate::gpu::ast::Dim;
pub use crate::gpu::ast::Dim3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemSpace {
    Host, Device,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Void,
    Scalar {sz: ElemSize},
    Pointer {ty: Box<Type>, mem: MemSpace},

    // Metal-specific types
    Buffer,
    Function,
    Library,
    Uint3,
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
pub enum SimdOp {
    Sum, Prod, Max, Min
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Ternary {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    HostArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Call {id: String, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // Metal-specific nodes
    KernelLaunch {
        id: Name, blocks: Dim3, threads: Dim3, args: Vec<Expr>, ty: Type, i: Info
    },
    AllocDevice {id: Name, elem_ty: Type, sz: usize, ty: Type, i: Info},
    Projection {e: Box<Expr>, label: String, ty: Type, i: Info},
    SimdOp {op: SimdOp, arg: Box<Expr>, ty: Type, i: Info},
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
    GetFun {lib: Name, fun_id: Name, ty: Type, i: Info},
    LoadLibrary {tops: Vec<Top>, ty: Type, i: Info},
}

impl ExprType<Type> for Expr {
    fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::Bool {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::Ternary {ty, ..} => ty,
            Expr::ArrayAccess {ty, ..} => ty,
            Expr::HostArrayAccess {ty, ..} => ty,
            Expr::Call {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
            Expr::KernelLaunch {ty, ..} => ty,
            Expr::AllocDevice {ty, ..} => ty,
            Expr::Projection {ty, ..} => ty,
            Expr::SimdOp {ty, ..} => ty,
            Expr::ThreadIdx {ty, ..} => ty,
            Expr::BlockIdx {ty, ..} => ty,
            Expr::GetFun {ty, ..} => ty,
            Expr::LoadLibrary {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::Call {..} | Expr::ThreadIdx {..} |
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
            Expr::Ternary {i, ..} => i.clone(),
            Expr::ArrayAccess {i, ..} => i.clone(),
            Expr::HostArrayAccess {i, ..} => i.clone(),
            Expr::Call {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
            Expr::KernelLaunch {i, ..} => i.clone(),
            Expr::AllocDevice {i, ..} => i.clone(),
            Expr::Projection {i, ..} => i.clone(),
            Expr::SimdOp {i, ..} => i.clone(),
            Expr::ThreadIdx {i, ..} => i.clone(),
            Expr::BlockIdx {i, ..} => i.clone(),
            Expr::GetFun {i, ..} => i.clone(),
            Expr::LoadLibrary {i, ..} => i.clone(),
        }
    }
}

impl SMapAccum<Expr> for Expr {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Expr), E> {
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
            Expr::Ternary {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::Ternary {
                    cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
                }))
            },
            Expr::ArrayAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                let (acc, idx) = f(acc, *idx)?;
                Ok((acc, Expr::ArrayAccess {
                    target: Box::new(target), idx: Box::new(idx), ty, i
                }))
            },
            Expr::HostArrayAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                let (acc, idx) = f(acc, *idx)?;
                Ok((acc, Expr::HostArrayAccess {
                    target: Box::new(target), idx: Box::new(idx), ty, i
                }))
            },
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty}))
            },
            Expr::KernelLaunch {id, blocks, threads, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::KernelLaunch {id, blocks, threads, args, ty, i}))
            },
            Expr::Projection {e, label, ty, i} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Projection {e: Box::new(e), label, ty, i}))
            },
            Expr::SimdOp {op, arg, ty, i} => {
                let (acc, arg) = f(acc?, *arg)?;
                Ok((acc, Expr::SimdOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::AllocDevice {..} | Expr::ThreadIdx {..} | Expr::BlockIdx {..} |
            Expr::GetFun {..} | Expr::LoadLibrary {..} => {
                Ok((acc?, self))
            }
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
    Return {value: Expr},

    // Metal-specific statements
    AllocThreadgroup {elem_ty: Type, id: Name, sz: usize},
    CopyMemory {
        elem_ty: Type, src: Expr, src_mem: MemSpace,
        dst: Expr, dst_mem: MemSpace, sz: usize
    },
    FreeDevice {id: Name},
    ThreadgroupBarrier,
    SubmitWork,
    CheckError {e: Expr},
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Self), E> {
        match self {
            Stmt::Definition {ty, id, expr} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Definition {ty, id, expr}))
            },
            Stmt::Assign {dst, expr} => {
                let (acc, dst) = f(acc?, dst)?;
                let (acc, expr) = f(acc, expr)?;
                Ok((acc, Stmt::Assign {dst, expr}))
            },
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (acc, init) = f(acc?, init)?;
                let (acc, cond) = f(acc, cond)?;
                let (acc, incr) = f(acc, incr)?;
                Ok((acc, Stmt::For {var_ty, var, init, cond, incr, body}))
            },
            Stmt::If {cond, thn, els} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::If {cond, thn, els}))
            },
            Stmt::While {cond, body} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::While {cond, body}))
            },
            Stmt::Return {value} => {
                let (acc, value) = f(acc?, value)?;
                Ok((acc, Stmt::Return {value}))
            },
            Stmt::CopyMemory {elem_ty, src, src_mem, dst, dst_mem, sz} => {
                let (acc, src) = f(acc?, src)?;
                let (acc, dst) = f(acc, dst)?;
                Ok((acc, Stmt::CopyMemory {elem_ty, src, src_mem, dst, dst_mem, sz}))
            },
            Stmt::CheckError {e} => {
                let (acc, e) = f(acc?, e)?;
                Ok((acc, Stmt::CheckError {e}))
            },
            Stmt::AllocThreadgroup {..} | Stmt::FreeDevice {..} |
            Stmt::ThreadgroupBarrier | Stmt::SubmitWork => {
                Ok((acc?, self))
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
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::For {var_ty, var, init, cond, incr, body}))
            },
            Stmt::If {cond, thn, els} => {
                let (acc, thn) = thn.smap_accum_l_result(acc, &f)?;
                let (acc, els) = els.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Stmt::If {cond, thn, els}))
            },
            Stmt::While {cond, body} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::While {cond, body}))
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::AllocThreadgroup {..} | Stmt::CopyMemory {..} |
            Stmt::FreeDevice {..} | Stmt::ThreadgroupBarrier | Stmt::SubmitWork |
            Stmt::CheckError {..} => {
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
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body});
            },
            Stmt::If {cond, thn, els} => {
                let thn = thn.sflatten_result(vec![], &f)?;
                let els = els.sflatten_result(vec![], &f)?;
                acc.push(Stmt::If {cond, thn, els});
            },
            Stmt::While {cond, body} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::While {cond, body});
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::AllocThreadgroup {..} | Stmt::ThreadgroupBarrier {..} |
            Stmt::CopyMemory {..} | Stmt::FreeDevice {..} | Stmt::SubmitWork |
            Stmt::CheckError {..} => {
                acc.push(self);
            }
        };
        Ok(acc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ParamAttribute {
    Buffer {idx: i64},
    ThreadIndex, BlockIndex
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub attr: Option<ParamAttribute>
}

#[derive(Clone, Debug, PartialEq)]
pub enum FunAttribute {
    LaunchBounds {threads: i64},
    ExternC,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    Include {header: String},
    VarDef {ty: Type, id: Name, init: Option<Expr>},
    FunDef {
        attrs: Vec<FunAttribute>, is_kernel: bool, ret_ty: Type, id: Name,
        params: Vec<Param>, body: Vec<Stmt>
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub tops: Vec<Top>
}
