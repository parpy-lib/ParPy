use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

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

    // CUDA-specific types
    Error,
    Stream,
    Graph,
    GraphExec,
    GraphExecUpdateResultInfo,
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
pub enum FuncAttribute {
    NonPortableClusterSizeAllowed
}

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    Success
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
    Error {e: Error, ty: Type, i: Info},
    GetLastError {ty: Type, i: Info},
    FuncSetAttribute {
        func: Name, attr: FuncAttribute, value: Box<Expr>, ty: Type, i: Info
    },
    MallocAsync {
        id: Name, elem_ty: Type, sz: usize, stream: Stream, ty: Type, i: Info
    },
    FreeAsync {id: Name, stream: Stream, ty: Type, i: Info},
    StreamCreate {id: Name, ty: Type, i: Info},
    StreamDestroy {id: Name, ty: Type, i: Info},
    StreamBeginCapture {stream: Stream, ty: Type, i: Info},
    StreamEndCapture {stream: Stream, graph: Name, ty: Type, i: Info},
    GraphDestroy {id: Name, ty: Type, i: Info},
    GraphExecInstantiate {exec_graph: Name, graph: Name, ty: Type, i: Info},
    GraphExecDestroy {id: Name, ty: Type, i: Info},
    GraphExecUpdate {exec_graph: Name, graph: Name, update: Name, ty: Type, i: Info},
    GraphExecLaunch {id: Name, ty: Type, i: Info},
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
            Expr::Error {ty, ..} => ty,
            Expr::GetLastError {ty, ..} => ty,
            Expr::FuncSetAttribute {ty, ..} => ty,
            Expr::MallocAsync {ty, ..} => ty,
            Expr::FreeAsync {ty, ..} => ty,
            Expr::StreamCreate {ty, ..} => ty,
            Expr::StreamDestroy {ty, ..} => ty,
            Expr::StreamBeginCapture {ty, ..} => ty,
            Expr::StreamEndCapture {ty, ..} => ty,
            Expr::GraphDestroy {ty, ..} => ty,
            Expr::GraphExecInstantiate {ty, ..} => ty,
            Expr::GraphExecDestroy {ty, ..} => ty,
            Expr::GraphExecUpdate {ty, ..} => ty,
            Expr::GraphExecLaunch {ty, ..} => ty,
        }
    }

    pub fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::Call {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} |
            Expr::Error {..} => true,
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
            Expr::Error {i, ..} => i.clone(),
            Expr::GetLastError {i, ..} => i.clone(),
            Expr::FuncSetAttribute {i, ..} => i.clone(),
            Expr::MallocAsync {i, ..} => i.clone(),
            Expr::FreeAsync {i, ..} => i.clone(),
            Expr::StreamCreate {i, ..} => i.clone(),
            Expr::StreamDestroy {i, ..} => i.clone(),
            Expr::StreamBeginCapture {i, ..} => i.clone(),
            Expr::StreamEndCapture {i, ..} => i.clone(),
            Expr::GraphDestroy {i, ..} => i.clone(),
            Expr::GraphExecInstantiate {i, ..} => i.clone(),
            Expr::GraphExecDestroy {i, ..} => i.clone(),
            Expr::GraphExecUpdate {i, ..} => i.clone(),
            Expr::GraphExecLaunch {i, ..} => i.clone(),
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
            Expr::FuncSetAttribute {func, attr, value, ty, i} => {
                let (acc, value) = f(acc?, *value)?;
                Ok((acc, Expr::FuncSetAttribute {func, attr, value: Box::new(value), ty, i}))
            },
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} | Expr::Error {..} |
            Expr::GetLastError {..} | Expr::MallocAsync {..} | Expr::FreeAsync {..} |
            Expr::StreamCreate {..} | Expr::StreamDestroy {..} | Expr::StreamBeginCapture {..} |
            Expr::StreamEndCapture {..} | Expr::GraphDestroy {..} |
            Expr::GraphExecInstantiate {..} | Expr::GraphExecDestroy {..} |
            Expr::GraphExecUpdate {..} | Expr::GraphExecLaunch {..} => Ok((acc?, self)),
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
            Expr::FuncSetAttribute {value, ..} => f(acc?, value),
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::Error {..} | Expr::GetLastError {..} | Expr::ThreadIdx {..} |
            Expr::BlockIdx {..} | Expr::MallocAsync {..} | Expr::FreeAsync {..} |
            Expr::StreamCreate {..} | Expr::StreamDestroy {..} |
            Expr::StreamBeginCapture {..} | Expr::StreamEndCapture {..} |
            Expr::GraphDestroy {..} | Expr::GraphExecInstantiate {..} |
            Expr::GraphExecDestroy {..} | Expr::GraphExecUpdate {..} |
            Expr::GraphExecLaunch {..} => acc
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stream {
    Default,
    Id(Name)
}

#[derive(Clone, Debug)]
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

    // CUDA-specific nodes
    Synchronize {scope: SyncScope},
    KernelLaunch {
        id: Name, blocks: Dim3, threads: Dim3, stream: Stream, args: Vec<Expr>
    },
    AllocShared {ty: Type, id: Name, sz: usize},
    CheckError {e: Expr},
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
            Stmt::KernelLaunch {id, blocks, threads, args, stream} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::KernelLaunch {id, blocks, threads, args, stream}))
            },
            Stmt::CheckError {e} => {
                let (acc, e) = f(acc?, e)?;
                Ok((acc, Stmt::CheckError {e}))
            },
            Stmt::AllocShared {..} | Stmt::Synchronize {..} => Ok((acc?, self))
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
            Stmt::CheckError {e} => f(acc?, e),
            Stmt::AllocShared {..} | Stmt::Synchronize {..} => acc
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
            Stmt::AllocShared {..} | Stmt::Synchronize {..} |
            Stmt::KernelLaunch {..} | Stmt::CheckError {..} => Ok((acc?, self))
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
            Stmt::AllocShared {..} | Stmt::Synchronize {..} |
            Stmt::KernelLaunch {..} | Stmt::CheckError {..} => acc
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
            Stmt::Synchronize {..} | Stmt::KernelLaunch {..} |
            Stmt::AllocShared {..} | Stmt::CheckError {..} => {
                acc.push(self);
            }
        };
        Ok(acc)
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
    VarDef {ty: Type, id: Name, init: Option<Expr>},
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
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}))
            },
            Top::Include {..} | Top::Namespace {..} | Top::StructDef {..} |
            Top::VarDef {..} => {
                Ok((acc?, self))
            },
        }
    }
}

pub type Ast = Vec<Top>;
