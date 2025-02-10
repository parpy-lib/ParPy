use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

// Re-export nodes from the IR AST that we reuse as is.
pub use crate::ir::ast::ElemSize;

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Void,
    Boolean,
    Scalar {sz: ElemSize},
    Pointer {sz: ElemSize},
    Struct {id: Name},
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Scalar {sz} => Some(sz),
            _ => None,
        }
    }
}

// Reuse the unary and binary operators defined in the IR AST
pub use crate::ir::ast::UnOp;
pub use crate::ir::ast::BinOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dim {
    X, Y, Z
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    Ternary {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    StructFieldAccess {target: Box<Expr>, label: String, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Struct {id: Name, fields: Vec<(String, Expr)>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},

    // CUDA-specific nodes
    ShflXorSync {value: Box<Expr>, idx: Box<Expr>, ty : Type, i: Info},
    ThreadIdx {dim: Dim, ty: Type, i: Info},
    BlockIdx {dim: Dim, ty: Type, i: Info},
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
            Expr::Convert {ty, ..} => ty,
            Expr::ShflXorSync {ty, ..} => ty,
            Expr::ThreadIdx {ty, ..} => ty,
            Expr::BlockIdx {ty, ..} => ty,
        }
    }

    pub fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} |
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
            Expr::StructFieldAccess {i, ..} => i.clone(),
            Expr::ArrayAccess {i, ..} => i.clone(),
            Expr::Struct {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
            Expr::ShflXorSync {i, ..} => i.clone(),
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
            (Expr::Convert {e: le, ..}, Expr::Convert {e: re, ..}) => le.eq(re),
            ( Expr::ShflXorSync {value: lval, idx: lidx, ..}
            , Expr::ShflXorSync {value: rval, idx: ridx, ..} ) => {
                lval.eq(rval) && lidx.eq(ridx)
            },
            (Expr::ThreadIdx {dim: ldim, ..}, Expr::ThreadIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (Expr::BlockIdx {dim: ldim, ..}, Expr::BlockIdx {dim: rdim, ..}) =>
                ldim.eq(rdim),
            (_, _) => false
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
            },
            Expr::Ternary {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc, *cond);
                let (acc, thn) = f(acc, *thn);
                let (acc, els) = f(acc, *els);
                (acc, Expr::Ternary {
                    cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
                })
            },
            Expr::StructFieldAccess {target, label, ty, i} => {
                let (acc, target) = f(acc, *target);
                (acc, Expr::StructFieldAccess {target: Box::new(target), label, ty, i})
            },
            Expr::ArrayAccess {target, idx, ty, i} => {
                let (acc, target) = f(acc, *target);
                let (acc, idx) = f(acc, *idx);
                (acc, Expr::ArrayAccess {target: Box::new(target), idx: Box::new(idx), ty, i})
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
            Expr::ShflXorSync {value, idx, ty, i} => {
                let (acc, value) = f(acc, *value);
                let (acc, idx) = f(acc, *idx);
                (acc, Expr::ShflXorSync {value: Box::new(value), idx: Box::new(idx), ty, i})
            },
            Expr::ThreadIdx {..} | Expr::BlockIdx {..} => (acc, self),
        }
    }
}

impl SFold<Expr> for Expr {
    fn sfold<A>(&self, f: impl Fn(A, &Expr) -> A, acc: A) -> A {
        match self {
            Expr::UnOp {arg, ..} => f(acc, arg),
            Expr::BinOp {lhs, rhs, ..} => f(f(acc, lhs), rhs),
            Expr::Ternary {cond, thn, els, ..} => f(f(f(acc, cond), thn), els),
            Expr::StructFieldAccess {target, ..} => f(acc, target),
            Expr::ArrayAccess {target, idx, ..} => f(f(acc, target), idx),
            Expr::Struct {fields, ..} => {
                fields.into_iter().fold(acc, |acc, (_, e)| f(acc, e))
            },
            Expr::Convert {e, ..} => f(acc, e),
            Expr::ShflXorSync {value, idx, ..} => f(f(acc, value), idx),
            Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::ThreadIdx {..} | Expr::BlockIdx {..} => acc,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dim3 {
    pub x: i64,
    pub y: i64,
    pub z: i64
}

impl Dim3 {
    pub fn get_dim(&self, dim: &Dim) -> i64 {
        match dim {
            Dim::X => self.x,
            Dim::Y => self.y,
            Dim::Z => self.z
        }
    }

    pub fn with_dim(self, dim: &Dim, n: i64) -> Dim3 {
        match dim {
            Dim::X => Dim3 {x: n, ..self},
            Dim::Y => Dim3 {y: n, ..self},
            Dim::Z => Dim3 {z: n, ..self}
        }
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
    pub fn with_blocks_dim(mut self, dim: &Dim, n: i64) -> LaunchArgs {
        self.blocks = self.blocks.with_dim(dim, n);
        self
    }

    pub fn with_threads_dim(mut self, dim: &Dim, n: i64) -> LaunchArgs {
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
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr},
    Assign {dst: Expr, expr: Expr},
    AllocShared {ty: Type, id: Name, sz: i64},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr,
        incr: Expr, body: Vec<Stmt>
    },
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>},
    While {cond: Expr, body: Vec<Stmt>},
    Syncthreads {},
    Dim3Definition {id: Name, args: Dim3},
    KernelLaunch {id: Name, blocks: Name, threads: Name, args: Vec<Expr>},
    Scope {body: Vec<Stmt>},
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l<A>(self, f: impl Fn(A, Expr) -> (A, Expr), acc: A) -> (A, Self) {
        match self {
            Stmt::Definition {ty, id, expr} => {
                let (acc, expr) = f(acc, expr);
                (acc, Stmt::Definition {ty, id, expr})
            },
            Stmt::Assign {dst, expr} => {
                let (acc, dst) = f(acc, dst);
                let (acc, expr) = f(acc, expr);
                (acc, Stmt::Assign {dst, expr})
            },
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (acc, init) = f(acc, init);
                let (acc, cond) = f(acc, cond);
                let (acc, incr) = f(acc, incr);
                (acc, Stmt::For {var_ty, var, init, cond, incr, body})
            },
            Stmt::If {cond, thn, els} => {
                let (acc, cond) = f(acc, cond);
                (acc, Stmt::If {cond, thn, els})
            },
            Stmt::While {cond, body} => {
                let (acc, cond) = f(acc, cond);
                (acc, Stmt::While {cond, body})
            },
            Stmt::KernelLaunch {id, blocks, threads, args} => {
                let (acc, args) = args.into_iter()
                    .fold((acc, vec![]), |(acc, mut args), a| {
                        let (acc, a) = f(acc, a);
                        args.push(a);
                        (acc, args)
                    });
                (acc, Stmt::KernelLaunch {id, blocks, threads, args})
            },
            Stmt::AllocShared {..} | Stmt::Syncthreads {..} |
            Stmt::Dim3Definition {..} | Stmt::Scope {..} => (acc, self),
        }
    }
}

impl SFold<Expr> for Stmt {
    fn sfold<A>(&self, f: impl Fn(A, &Expr) -> A, acc: A) -> A {
        match self {
            Stmt::Definition {expr, ..} => f(acc, expr),
            Stmt::Assign {dst, expr} => {
                let acc = f(acc, dst);
                f(acc, expr)
            },
            Stmt::For {init, cond, incr, ..} => {
                let acc = f(acc, init);
                let acc = f(acc, cond);
                f(acc, incr)
            },
            Stmt::If {cond, ..} => f(acc, cond),
            Stmt::While {cond, ..} => f(acc, cond),
            Stmt::KernelLaunch {args, ..} => args.sfold(&f, acc),
            Stmt::AllocShared {..} | Stmt::Syncthreads {} |
            Stmt::Dim3Definition {..} | Stmt::Scope {..} => acc,
        }
    }
}

impl SMapAccum<Stmt> for Stmt {
    fn smap_accum_l<A>(self, f: impl Fn(A, Stmt) -> (A, Stmt), acc: A) -> (A, Self) {
        match self {
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::AllocShared {..} |
            Stmt::Syncthreads {} | Stmt::Dim3Definition {..} |
            Stmt::KernelLaunch {..} => (acc, self),
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (acc, body) = body.smap_accum_l(&f, acc);
                (acc, Stmt::For {var_ty, var, init, cond, incr, body})
            },
            Stmt::If {cond, thn, els} => {
                let (acc, thn) = thn.smap_accum_l(&f, acc);
                let (acc, els) = els.smap_accum_l(&f, acc);
                (acc, Stmt::If {cond, thn, els})
            },
            Stmt::While {cond, body} => {
                let (acc, body) = body.smap_accum_l(&f, acc);
                (acc, Stmt::While {cond, body})
            },
            Stmt::Scope {body} => {
                let (acc, body) = body.smap_accum_l(&f, acc);
                (acc, Stmt::Scope {body})
            },
        }
    }
}

impl SFold<Stmt> for Stmt {
    fn sfold<A>(&self, f: impl Fn(A, &Stmt) -> A, acc: A) -> A {
        match self {
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::AllocShared {..} |
            Stmt::Syncthreads {} | Stmt::Dim3Definition {..} |
            Stmt::KernelLaunch {..} => acc,
            Stmt::For {body, ..} => body.sfold(&f, acc),
            Stmt::If {thn, els, ..} => {
                let acc = thn.sfold(&f, acc);
                els.sfold(&f, acc)
            },
            Stmt::While {body, ..} => body.sfold(&f, acc),
            Stmt::Scope {body} => body.sfold(&f, acc),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Global, Entry
}

#[derive(Clone, Debug)]
pub struct Field {
    pub ty: Type,
    pub id: String,
    pub i: Info
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug)]
pub enum Top {
    Include {header: String},
    StructDef {id: Name, fields: Vec<Field>},
    FunDef {
        attr: Attribute, ret_ty: Type, id: Name, params: Vec<Param>,
        body: Vec<Stmt>
    },
}

impl SMapAccum<Stmt> for Top {
    fn smap_accum_l<A>(self, f: impl Fn(A, Stmt) -> (A, Stmt), acc: A) -> (A, Self) {
        match self {
            Top::Include {..} | Top::StructDef {..} => (acc, self),
            Top::FunDef {attr, ret_ty, id, params, body} => {
                let (acc, body) = body.into_iter()
                    .fold((acc, vec![]), |(acc, mut stmts), s| {
                        let (acc, s) = f(acc, s);
                        stmts.push(s);
                        (acc, stmts)
                    });
                (acc, Top::FunDef {attr, ret_ty, id, params, body})
            },
        }
    }
}

pub type Ast = Vec<Top>;
