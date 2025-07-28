use crate::utils::ast::ExprType;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::utils::ast::BinOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemSpace {
    // Memory allocated on the host (CPU)
    Host,

    // Global memory on the device (e.g., a GPU)
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dim {
    X, Y, Z
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Void,
    Scalar {sz: ElemSize},
    Pointer {ty: Box<Type>, mem: MemSpace},
    Struct {id: Name},
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Scalar {sz} => Some(sz),
            _ => None,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.get_scalar_elem_size().is_some()
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    IfExpr {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Call {id: String, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // High-level representation of a struct literal value.
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},

    // Expressions referring to the thread index and thread block index of an executing thread, in
    // either of the three dimensions.
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
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
            Expr::IfExpr {ty, ..} => ty,
            Expr::StructFieldAccess {ty, ..} => ty,
            Expr::ArrayAccess {ty, ..} => ty,
            Expr::Call {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
            Expr::Struct {ty, ..} => ty,
            Expr::ThreadIdx {ty, ..} => ty,
            Expr::BlockIdx {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::Call {..} | Expr::Struct {..} |
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
            Expr::IfExpr {i, ..} => i.clone(),
            Expr::StructFieldAccess {i, ..} => i.clone(),
            Expr::ArrayAccess {i, ..} => i.clone(),
            Expr::Call {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
            Expr::Struct {i, ..} => i.clone(),
            Expr::ThreadIdx {i, ..} => i.clone(),
            Expr::BlockIdx {i, ..} => i.clone(),
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
            ( Expr::IfExpr {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::IfExpr {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.eq(rcond) && lthn.eq(rthn) && lels.eq(rels),
            ( Expr::StructFieldAccess {target: ltarget, label: llabel, ..}
            , Expr::StructFieldAccess {target: rtarget, label: rlabel, ..} ) =>
                ltarget.eq(rtarget) && llabel.eq(rlabel),
            ( Expr::ArrayAccess {target: ltarget, idx: lidx, ..}
            , Expr::ArrayAccess {target: rtarget, idx: ridx, ..} ) =>
                ltarget.eq(rtarget) && lidx.eq(ridx),
            ( Expr::Call {id: lid, args: largs, ..}
            , Expr::Call {id: rid, args: rargs, ..} ) =>
                lid.eq(rid) && largs.eq(rargs),
            (Expr::Convert {e: le, ..}, Expr::Convert {e: re, ..}) => le.eq(re),
            ( Expr::Struct {id: lid, fields: lfields, ..}
            , Expr::Struct {id: rid, fields: rfields, ..} ) =>
                lid.eq(rid) && lfields.eq(rfields),
            (Expr::ThreadIdx {dim: ldim, ..}, Expr::ThreadIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (Expr::BlockIdx {dim: ldim, ..}, Expr::BlockIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
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
            Expr::IfExpr {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::IfExpr {
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
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty}))
            },
            Expr::Struct {id, fields, ty, i} => {
                let (acc, fields) = fields.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Struct {id, fields, ty, i}))
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
            Expr::IfExpr {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::StructFieldAccess {target, ..} => f(acc?, target),
            Expr::ArrayAccess {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
            Expr::Struct {fields, ..} => fields.sfold_result(acc, &f),
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} | Expr::BlockIdx {..} => acc,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dim3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl Dim3 {
    pub fn get_dim(&self, dim: &Dim) -> i64 {
        match dim {
            Dim::X => self.x,
            Dim::Y => self.y,
            Dim::Z => self.z,
        }
    }

    pub fn with_dim(self, dim: &Dim, n: i64) -> Dim3 {
        match dim {
            Dim::X => Dim3 {x: n, ..self},
            Dim::Y => Dim3 {y: n, ..self},
            Dim::Z => Dim3 {z: n, ..self},
        }
    }

    pub fn prod(&self) -> i64 {
        self.x * self.y * self.z
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Dim3 {x: 1, y: 1, z: 1}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaunchArgs {
    pub blocks: Dim3,
    pub threads: Dim3
}

impl LaunchArgs {
    pub fn with_blocks_dim(mut self, dim: &Dim, n: i64) -> Self {
        self.blocks = self.blocks.with_dim(dim, n);
        self
    }

    pub fn with_threads_dim(mut self, dim: &Dim, n: i64) -> Self {
        self.threads = self.threads.with_dim(dim, n);
        self
    }
}

impl Default for LaunchArgs {
    fn default() -> Self {
        LaunchArgs {blocks: Dim3::default(), threads: Dim3::default()}
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SyncScope {
    Block, Cluster
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: Expr, body: Vec<Stmt>, i: Info
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    Return {value: Expr, i: Info},
    Scope {body: Vec<Stmt>, i: Info},

    // Intermediate node representing a parallel reduction.
    ParallelReduction {
        var_ty: Type, var: Name, init: Expr, cond: Expr, incr: Expr,
        body: Vec<Stmt>, nthreads: i64, tpb: i64, i: Info
    },

    // Synchronization among threads in a given scope within a kernel. Ensures that all included
    // threads have reached the synchronization point before continuing.
    Synchronize {scope: SyncScope, i: Info},

    // Abstract representation of parallel reductions, used to enable backend-specific behavior for
    // these operations, when they are supported.
    WarpReduce {value: Expr, op: BinOp, int_ty: Type, res_ty: Type, i: Info},
    ClusterReduce {
        block_idx: Expr, shared_var: Expr, temp_var: Expr, blocks_per_cluster: i128,
        op: BinOp, int_ty: Type, res_ty: Type, i: Info
    },

    // Represents the launch of a kernel with a specified name invoked with the provided vector of
    // arguments and the specified launch arguments, controlling the number of blocks and threads
    // of the executing grid.
    KernelLaunch {id: Name, args: Vec<Expr>, grid: LaunchArgs, i: Info},

    // Statements related to memory management. Note that we distinguish between different kinds of
    // memory allocations as shared memory allocations introduce a new variable while device
    // allocations refer to an existing variable.
    AllocDevice {elem_ty: Type, id: Name, sz: usize, i: Info},
    AllocShared {elem_ty: Type, id: Name, sz: usize, i: Info},
    FreeDevice {id: Name, i: Info},
    CopyMemory {
        elem_ty: Type, src: Expr, src_mem: MemSpace,
        dst: Expr, dst_mem: MemSpace, sz: usize, i: Info
    }
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
            Stmt::Return {i, ..} => i.clone(),
            Stmt::Scope {i, ..} => i.clone(),
            Stmt::ParallelReduction {i, ..} => i.clone(),
            Stmt::Synchronize {i, ..} => i.clone(),
            Stmt::WarpReduce {i, ..} => i.clone(),
            Stmt::ClusterReduce {i, ..} => i.clone(),
            Stmt::KernelLaunch {i, ..} => i.clone(),
            Stmt::AllocDevice {i, ..} => i.clone(),
            Stmt::AllocShared {i, ..} => i.clone(),
            Stmt::FreeDevice {i, ..} => i.clone(),
            Stmt::CopyMemory {i, ..} => i.clone(),
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
            Stmt::Definition {ty, id, expr, i} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Definition {ty, id, expr, i}))
            },
            Stmt::Assign {dst, expr, i} => {
                let (acc, dst) = f(acc?, dst)?;
                let (acc, expr) = f(acc, expr)?;
                Ok((acc, Stmt::Assign {dst, expr, i}))
            },
            Stmt::For {var_ty, var, init, cond, incr, body, i} => {
                let (acc, init) = f(acc?, init)?;
                let (acc, cond) = f(acc, cond)?;
                let (acc, incr) = f(acc, incr)?;
                Ok((acc, Stmt::For {var_ty, var, init, cond, incr, body, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, cond) = f(acc?, cond)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::Return {value, i} => {
                let (acc, value) = f(acc?, value)?;
                Ok((acc, Stmt::Return {value, i}))
            },
            Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i} => {
                let (acc, init) = f(acc?, init)?;
                let (acc, cond) = f(acc, cond)?;
                let (acc, incr) = f(acc, incr)?;
                Ok((acc, Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i}))
            },
            Stmt::WarpReduce {value, op, int_ty, res_ty, i} => {
                let (acc, value) = f(acc?, value)?;
                Ok((acc, Stmt::WarpReduce {value, op, int_ty, res_ty, i}))
            },
            Stmt::ClusterReduce {block_idx, shared_var, temp_var, blocks_per_cluster,
                                 op, int_ty, res_ty, i} => {
                let (acc, block_idx) = f(acc?, block_idx)?;
                let (acc, shared_var) = f(acc, shared_var)?;
                let (acc, temp_var) = f(acc, temp_var)?;
                Ok((acc, Stmt::ClusterReduce {
                    block_idx, shared_var, temp_var, blocks_per_cluster, op, int_ty, res_ty, i
                }))
            },
            Stmt::KernelLaunch {id, args, grid, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::KernelLaunch {id, args, grid, i}))
            },
            Stmt::CopyMemory {elem_ty, src, dst, sz, src_mem, dst_mem, i} => {
                let (acc, src) = f(acc?, src)?;
                let (acc, dst) = f(acc, dst)?;
                Ok((acc, Stmt::CopyMemory {elem_ty, src, dst, sz, src_mem, dst_mem, i}))
            },
            Stmt::Scope {..} | Stmt::Synchronize {..} | Stmt::AllocDevice {..} |
            Stmt::AllocShared {..} | Stmt::FreeDevice {..} => Ok((acc?, self)),
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
            Stmt::Definition {expr, ..} => f(acc?, expr),
            Stmt::Assign {dst, expr, ..} => f(f(acc?, dst)?, expr),
            Stmt::For {init, cond, incr, ..} => f(f(f(acc?, init)?, cond)?, incr),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::Return {value, ..} => f(acc?, value),
            Stmt::ParallelReduction {init, cond, incr, ..} =>
                f(f(f(acc?, init)?, cond)?, incr),
            Stmt::WarpReduce {value, ..} => f(acc?, value),
            Stmt::ClusterReduce {block_idx, shared_var, temp_var, ..} =>
                f(f(f(acc?, block_idx)?, shared_var)?, temp_var),
            Stmt::KernelLaunch {args, ..} => args.sfold_result(acc, &f),
            Stmt::CopyMemory {src, dst, ..} => f(f(acc?, src)?, dst),
            Stmt::Scope {..} | Stmt::Synchronize {..} | Stmt::AllocDevice {..} |
            Stmt::AllocShared {..} | Stmt::FreeDevice {..} => acc,
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
            Stmt::For {var_ty, var, init, cond, incr, body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::For {var_ty, var, init, cond, incr, body, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (acc, thn) = thn.smap_accum_l_result(acc, &f)?;
                let (acc, els) = els.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Stmt::If {cond, thn, els, i}))
            },
            Stmt::While {cond, body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::While {cond, body, i}))
            },
            Stmt::Scope {body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::Scope {body, i}))
            },
            Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, i}))
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Synchronize {..} | Stmt::WarpReduce {..} | Stmt::ClusterReduce {..} |
            Stmt::KernelLaunch {..} | Stmt::AllocDevice {..} | Stmt::AllocShared {..} |
            Stmt::FreeDevice {..} | Stmt::CopyMemory {..} => {
                Ok((acc?, self))
            }
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
            Stmt::Scope {body, ..} => body.sfold_result(acc, &f),
            Stmt::ParallelReduction {body, ..} => body.sfold_result(acc, &f),
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Synchronize {..} | Stmt::WarpReduce {..} | Stmt::ClusterReduce {..} |
            Stmt::KernelLaunch {..} | Stmt::AllocDevice {..} | Stmt::AllocShared {..} |
            Stmt::FreeDevice {..} | Stmt::CopyMemory {..} => acc,
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
            Stmt::For {var_ty, var, init, cond, incr, body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i});
            },
            Stmt::If {cond, thn, els, i} => {
                let thn = thn.sflatten_result(vec![], &f)?;
                let els = els.sflatten_result(vec![], &f)?;
                acc.push(Stmt::If {cond, thn, els, i});
            },
            Stmt::While {cond, body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::While {cond, body, i});
            },
            Stmt::Scope {body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::Scope {body, i})
            }
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::ParallelReduction {..} | Stmt::Synchronize {..} | Stmt::WarpReduce {..} |
            Stmt::ClusterReduce {..} | Stmt::KernelLaunch {..} | Stmt::AllocDevice {..} |
            Stmt::AllocShared {..} | Stmt::FreeDevice {..} | Stmt::CopyMemory {..} => {
                acc.push(self);
            }
        };
        Ok(acc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub id: String,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    Device, Host
}

#[derive(Clone, Debug)]
pub enum KernelAttribute {
    LaunchBounds {threads: i64},
    ClusterDims {dims: Dim3},
}

#[derive(Clone, Debug)]
pub enum Top {
    KernelFunDef {attrs: Vec<KernelAttribute>, id: Name, params: Vec<Param>, body: Vec<Stmt>},
    FunDef {ret_ty: Type, id: Name, params: Vec<Param>, body: Vec<Stmt>, target: Target},
    StructDef {id: Name, fields: Vec<Field>},
}

impl SMapAccum<Stmt> for Top {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Stmt) -> Result<(A, Stmt), E>
    ) -> Result<(A, Self), E> {
        match self {
            Top::KernelFunDef {attrs, id, params, body} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Top::KernelFunDef {attrs, id, params, body}))
            },
            Top::FunDef {ret_ty, id, params, body, target} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Top::FunDef {ret_ty, id, params, body, target}))
            },
            Top::StructDef {..} => Ok((acc?, self))
        }
    }
}

pub type Ast = Vec<Top>;
