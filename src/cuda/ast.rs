use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

// Re-use nodes from the GPU IR AST.
pub use crate::gpu::ast::ElemSize;
pub use crate::gpu::ast::UnOp;
pub use crate::gpu::ast::BinOp;
pub use crate::gpu::ast::Dim;
pub use crate::gpu::ast::Dim3;
pub use crate::gpu::ast::SyncScope;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Void,
    Boolean,
    Scalar {sz: ElemSize},
    Pointer {ty: Box<Type>},
    Struct {id: Name},
    CudaError {},
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
pub enum CudaFuncAttribute {
    NonPortableClusterSizeAllowed
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Ternary {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
    Call {id: String, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // CUDA-specific nodes
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
    CudaFuncSetAttribute {
        func: Name, attr: CudaFuncAttribute, value: Box<Expr>, ty: Type, i: Info
    },
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
            Expr::Ternary {ty, ..} => ty,
            Expr::StructFieldAccess {ty, ..} => ty,
            Expr::ArrayAccess {ty, ..} => ty,
            Expr::Struct {ty, ..} => ty,
            Expr::Call {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
            Expr::ThreadIdx {ty, ..} => ty,
            Expr::BlockIdx {ty, ..} => ty,
            Expr::CudaFuncSetAttribute {ty, ..} => ty,
        }
    }

    pub fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::Call {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => true,
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
            Expr::StructFieldAccess {i, ..} => i.clone(),
            Expr::ArrayAccess {i, ..} => i.clone(),
            Expr::Struct {i, ..} => i.clone(),
            Expr::Call {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
            Expr::ThreadIdx {i, ..} => i.clone(),
            Expr::BlockIdx {i, ..} => i.clone(),
            Expr::CudaFuncSetAttribute {i, ..} => i.clone(),
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
            ( Expr::Ternary {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::Ternary {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.eq(rcond) && lthn.eq(rthn) && lels.eq(rels),
            ( Expr::StructFieldAccess {target: ltarget, label: llabel, ..}
            , Expr::StructFieldAccess {target: rtarget, label: rlabel, ..} ) =>
                ltarget.eq(rtarget) && llabel.eq(rlabel),
            ( Expr::ArrayAccess {target: ltarget, idx: lidx, ..}
            , Expr::ArrayAccess {target: rtarget, idx: ridx, ..} ) =>
                ltarget.eq(rtarget) && lidx.eq(ridx),
            ( Expr::Struct {id: lid, fields: lfields, ..}
            , Expr::Struct {id: rid, fields: rfields, ..} ) =>
                lid.eq(rid) && lfields.eq(rfields),
            ( Expr::Call {id: lid, args: largs, ..}
            , Expr::Call {id: rid, args: rargs, ..}) =>
                lid.eq(rid) && largs.eq(rargs),
            (Expr::Convert {e: le, ..}, Expr::Convert {e: re, ..}) => le.eq(re),
            (Expr::ThreadIdx {dim: ldim, ..}, Expr::ThreadIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (Expr::BlockIdx {dim: ldim, ..}, Expr::BlockIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            ( Expr::CudaFuncSetAttribute {func: lfunc, attr: lattr, value: lvalue, ..}
            , Expr::CudaFuncSetAttribute {func: rfunc, attr: rattr, value: rvalue, ..} ) =>
                lfunc.eq(rfunc) && lattr.eq(rattr) && lvalue.eq(rvalue),
            (_, _) => false
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
                Ok((acc, Expr::BinOp {
                    lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i
                }))
            },
            Expr::Ternary {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::Ternary {
                    cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
                }))
            },
            Expr::StructFieldAccess {target, label, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                Ok((acc, Expr::StructFieldAccess {
                    target: Box::new(target), label, ty, i
                }))
            },
            Expr::ArrayAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                let (acc, idx) = f(acc, *idx)?;
                Ok((acc, Expr::ArrayAccess {
                    target: Box::new(target), idx: Box::new(idx), ty, i
                }))
            },
            Expr::Struct {id, fields, ty, i} => {
                let (acc, fields) = fields.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Struct {id, fields, ty, i}))
            },
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty}))
            },
            Expr::CudaFuncSetAttribute {func, attr, value, ty, i} => {
                let (acc, value) = f(acc?, *value)?;
                Ok((acc, Expr::CudaFuncSetAttribute {func, attr, value: Box::new(value), ty, i}))
            },
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => Ok((acc?, self)),
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
            Expr::Ternary {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::StructFieldAccess {target, ..} => f(acc?, target),
            Expr::ArrayAccess {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Struct {fields, ..} => fields.sfold_result(acc, &f),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
            Expr::CudaFuncSetAttribute {value, ..} => f(acc?, value),
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} | Expr::BlockIdx {..} => acc,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Option<Expr>},
    Assign {dst: Expr, expr: Expr},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: Expr, body: Vec<Stmt>
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>},
    While {cond: Expr, body: Vec<Stmt>},
    Return {value: Expr},
    Synchronize {scope: SyncScope},
    KernelLaunch {id: Name, blocks: Dim3, threads: Dim3, args: Vec<Expr>},
    AllocShared {ty: Type, id: Name, sz: usize},
    MallocAsync {id: Name, elem_ty: Type, sz: usize},
    FreeAsync {id: Name}
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Self), E> {
        match self {
            Stmt::Definition {ty, id, expr} => match expr {
                Some(e) => {
                    let (acc, e) = f(acc?, e)?;
                    Ok((acc, Stmt::Definition {ty, id, expr: Some(e)}))
                },
                None => Ok((acc?, Stmt::Definition {ty, id, expr}))
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
            Stmt::KernelLaunch {id, blocks, threads, args} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::KernelLaunch {id, blocks, threads, args}))
            },
            Stmt::AllocShared {..} | Stmt::Synchronize {..} |
            Stmt::MallocAsync {..} | Stmt::FreeAsync {..} => Ok((acc?, self))
        }
    }
}

impl SFold<Expr> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Stmt::Definition {expr, ..} => match expr {
                Some(e) => f(acc?, e),
                None => acc
            },
            Stmt::Assign {dst, expr} => f(f(acc?, dst)?, expr),
            Stmt::For {init, cond, incr, ..} => f(f(f(acc?, init)?, cond)?, incr),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::Return {value, ..} => f(acc?, value),
            Stmt::KernelLaunch {args, ..} => args.sfold_result(acc, &f),
            Stmt::AllocShared {..} | Stmt::Synchronize {..} |
            Stmt::MallocAsync {..} | Stmt::FreeAsync {..} => acc,
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
            Stmt::AllocShared {..} | Stmt::Synchronize {..} | Stmt::MallocAsync {..} |
            Stmt::FreeAsync {..} | Stmt::KernelLaunch {..} => Ok((acc?, self))
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
            Stmt::If {thn, els, ..} => els.sfold_result(thn.sfold_result(acc, &f), &f),
            Stmt::While {body, ..} => body.sfold_result(acc, &f),
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::AllocShared {..} | Stmt::Synchronize {..} | Stmt::MallocAsync {..} |
            Stmt::FreeAsync {..} | Stmt::KernelLaunch {..} => acc,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Global, Device, Entry
}

#[derive(Clone, Debug)]
pub struct Field {
    pub ty: Type,
    pub id: String
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type
}

#[derive(Clone, Debug)]
pub enum KernelAttribute {
    LaunchBounds {threads: i64},
    ClusterDims {dims: Dim3},
}

#[derive(Clone, Debug)]
pub enum Top {
    Include {header: String},
    Namespace {ns: String, alias: Option<String>},
    StructDef {id: Name, fields: Vec<Field>},
    FunDef {
        dev_attr: Attribute, ret_ty: Type, attrs: Vec<KernelAttribute>,
        id: Name, params: Vec<Param>, body: Vec<Stmt>
    },
}

impl SMapAccum<Stmt> for Top {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Stmt) -> Result<(A, Stmt), E>
    ) -> Result<(A, Self), E> {
        match self {
            Top::Include {..} | Top::Namespace {..} | Top::StructDef {..} =>
                Ok((acc?, self)),
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}))
            },
        }
    }
}

pub type Ast = Vec<Top>;
