use crate::option::CompileBackend;
use crate::utils::ast::ExprType;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use strum_macros::EnumIter;
use std::cmp::Ordering;
use std::collections::BTreeMap;

pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::utils::ast::BinOp;
pub use crate::utils::ast::Target;
pub use crate::par::LoopPar;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum TensorShape {
    Num {n: i64},
    Symbol {id: Name},
}

impl TensorShape {
    pub fn extract_num(&self) -> Option<i64> {
        match self {
            TensorShape::Num {n} => Some(*n),
            TensorShape::Symbol {..} => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum TensorElemSize {
    Fixed {sz: ElemSize},
    Variable {id: Name},
}

impl Default for TensorElemSize {
    fn default() -> Self {
        TensorElemSize::Fixed {sz: ElemSize::default()}
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum Type {
    String,
    Tensor {sz: TensorElemSize, shape: Vec<TensorShape>},
    Tuple {elems: Vec<Type>},
    Dict {fields: BTreeMap<String, Type>},
    Void,
    #[default] Unknown
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Tensor {sz, shape} if shape.is_empty() => match sz {
                TensorElemSize::Fixed {sz} => Some(sz),
                TensorElemSize::Variable {..} => None
            },
            _ => None
        }
    }

    pub fn fixed_scalar(sz: ElemSize) -> Type {
        Type::Tensor {sz: TensorElemSize::Fixed {sz}, shape: vec![]}
    }

    pub fn is_scalar(&self) -> bool {
        self.get_scalar_elem_size().is_some()
    }

    pub fn is_bool_scalar(&self) -> bool {
        match self.get_scalar_elem_size() {
            Some(ElemSize::Bool) => true,
            _ => false
        }
    }

    pub fn is_int_scalar(&self) -> bool {
        match self.get_scalar_elem_size() {
            Some(sz) => sz.is_integer(),
            _ => false
        }
    }

    pub fn is_float_scalar(&self) -> bool {
        match self.get_scalar_elem_size() {
            Some(sz) => sz.is_floating_point(),
            None => false
        }
    }

    pub fn get_dict_type_fields(&self) -> BTreeMap<String, Type> {
        if let Type::Dict {fields} = self {
            fields.clone()
        } else {
            panic!("Internal error: expected dictionary type, found {self:?}")
        }
    }
}

impl SMapAccum<Type> for Type {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Type) -> Result<(A, Type), E>
    ) -> Result<(A, Type), E> {
        match self {
            Type::Tuple {elems} => {
                let (acc, elems) = elems.smap_accum_l_result(acc, &f)?;
                Ok((acc, Type::Tuple {elems}))
            },
            Type::Dict {fields} => {
                let (acc, fields) = fields.smap_accum_l_result(acc, &f)?;
                Ok((acc, Type::Dict {fields}))
            },
            Type::String | Type::Tensor {..} | Type::Void | Type::Unknown => {
                Ok((acc?, self))
            }
        }
    }
}

impl SFold<Type> for Type {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Type) -> Result<A, E>
    ) -> Result<A, E> {
        match self {
            Type::Tuple {elems} => elems.sfold_result(acc, f),
            Type::Dict {fields} => fields.sfold_result(acc, f),
            Type::String | Type::Tensor {..} | Type::Void | Type::Unknown => acc
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReduceOp {
    #[default] Max, Min, Sum, Prod
}

impl ReduceOp {
    pub fn to_bin_op(&self) -> BinOp {
        match self {
            ReduceOp::Max => BinOp::Max,
            ReduceOp::Min => BinOp::Min,
            ReduceOp::Sum => BinOp::Add,
            ReduceOp::Prod => BinOp::Mul,
        }
    }
}

#[derive(Clone, Debug, EnumIter)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    String {v: String, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    ReduceOp {op: ReduceOp, arg: Box<Expr>, ty: Type, i: Info},
    IfExpr {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    Subscript {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Slice {lo: Option<Box<Expr>>, hi: Option<Box<Expr>>, ty: Type, i: Info},
    Tuple {elems: Vec<Expr>, ty: Type, i: Info},
    List {elems: Vec<Expr>, ty: Type, i: Info},
    Call {id: Name, args: Vec<Expr>, ty: Type, i: Info},
    Callback {id: Name, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type, i: Info},
    GpuContext {ty: Type, i: Info},
    Inline {e: Box<Expr>, ty: Type, i: Info},
    Label {label: String, ty: Type, i: Info},
    StaticBackendEq {backend: CompileBackend, ty: Type, i: Info},
    StaticTypesEq {lhs: TensorElemSize, rhs: TensorElemSize, ty: Type, i: Info},
    StaticFail {msg: String, ty: Type, i: Info},
    AllocShared {shape: Vec<Expr>, sz: TensorElemSize, ty: Type, i: Info},
}

impl Expr {
    pub fn discriminator(&self) -> u8 {
        match self {
            Expr::Var {..} => 0,
            Expr::String {..} => 1,
            Expr::Bool {..} => 2,
            Expr::Int {..} => 3,
            Expr::Float {..} => 4,
            Expr::UnOp {..} => 5,
            Expr::BinOp {..} => 6,
            Expr::ReduceOp {..} => 7,
            Expr::IfExpr {..} => 8,
            Expr::Subscript {..} => 9,
            Expr::Slice {..} => 10,
            Expr::Tuple {..} => 11,
            Expr::List {..} => 12,
            Expr::Call {..} => 13,
            Expr::Callback {..} => 14,
            Expr::Convert {..} => 15,
            Expr::GpuContext {..} => 16,
            Expr::Inline {..} => 17,
            Expr::Label {..} => 18,
            Expr::StaticBackendEq {..} => 19,
            Expr::StaticTypesEq {..} => 20,
            Expr::StaticFail {..} => 21,
            Expr::AllocShared {..} => 22,
        }
    }

    pub fn with_info(self, i: Info) -> Self {
        match self {
            Expr::Var {id, ty, ..} => Expr::Var {id, ty, i},
            Expr::String {v, ty, ..} => Expr::String {v, ty, i},
            Expr::Bool {v, ty, ..} => Expr::Bool {v, ty, i},
            Expr::Int {v, ty, ..} => Expr::Int {v, ty, i},
            Expr::Float {v, ty, ..} => Expr::Float {v, ty, i},
            Expr::UnOp {op, arg, ty, ..} => Expr::UnOp {op, arg, ty, i},
            Expr::BinOp {lhs, op, rhs, ty, ..} => Expr::BinOp {lhs, op, rhs, ty, i},
            Expr::ReduceOp {op, arg, ty, ..} => Expr::ReduceOp {op, arg, ty, i},
            Expr::IfExpr {cond, thn, els, ty, ..} => Expr::IfExpr {cond, thn, els, ty, i},
            Expr::Subscript {target, idx, ty, ..} => Expr::Subscript {target, idx, ty, i},
            Expr::Slice {lo, hi, ty, ..} => Expr::Slice {lo, hi, ty, i},
            Expr::Tuple {elems, ty, ..} => Expr::Tuple {elems, ty, i},
            Expr::List {elems, ty, ..} => Expr::List {elems, ty, i},
            Expr::Call {id, args, ty, ..} => Expr::Call {id, args, ty, i},
            Expr::Callback {id, args, ty, ..} => Expr::Callback {id, args, ty, i},
            Expr::Convert {e, ty, ..} => Expr::Convert {e, ty, i},
            Expr::GpuContext {ty, ..} => Expr::GpuContext {ty, i},
            Expr::Inline {e, ty, ..} => Expr::Inline {e, ty, i},
            Expr::Label {label, ty, ..} => Expr::Label {label, ty, i},
            Expr::StaticBackendEq {backend, ty, ..} => Expr::StaticBackendEq {backend, ty, i},
            Expr::StaticTypesEq {lhs, rhs, ty, ..} => Expr::StaticTypesEq {lhs, rhs, ty, i},
            Expr::StaticFail {msg, ty, ..} => Expr::StaticFail {msg, ty, i},
            Expr::AllocShared {shape, sz, ty, ..} => Expr::AllocShared {shape, sz, ty, i},
        }
    }
}

impl ExprType<Type> for Expr {
    fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::String {ty, ..} => ty,
            Expr::Bool {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::ReduceOp {ty, ..} => ty,
            Expr::IfExpr {ty, ..} => ty,
            Expr::Subscript {ty, ..} => ty,
            Expr::Slice {ty, ..} => ty,
            Expr::Tuple {ty, ..} => ty,
            Expr::List {ty, ..} => ty,
            Expr::Call {ty, ..} => ty,
            Expr::Callback {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
            Expr::GpuContext {ty, ..} => ty,
            Expr::Inline {ty, ..} => ty,
            Expr::Label {ty, ..} => ty,
            Expr::StaticBackendEq {ty, ..} => ty,
            Expr::StaticTypesEq {ty, ..} => ty,
            Expr::StaticFail {ty, ..} => ty,
            Expr::AllocShared {ty, ..} => ty,
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} | Expr::String {..} | Expr::Bool {..} |
            Expr::Int {..} | Expr::Float {..} | Expr::Inline {..} |
            Expr::Label {..} | Expr::StaticFail {..} => {
                true
            },
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::ReduceOp {..} | Expr::IfExpr {..} |
            Expr::Subscript {..} | Expr::Slice {..} | Expr::Tuple {..} | Expr::List {..} |
            Expr::Call {..} | Expr::Callback {..} | Expr::Convert {..} | Expr::GpuContext {..} |
            Expr::StaticBackendEq {..} | Expr::StaticTypesEq {..} | Expr::AllocShared {..} => {
                false
            },
        }
    }
}

impl Ord for Expr {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Expr::Var {id: lid, ..}, Expr::Var {id: rid, ..}) => lid.cmp(rid),
            (Expr::String {v: lv, ..}, Expr::String {v: rv, ..}) => lv.cmp(rv),
            (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) => lv.cmp(rv),
            (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => lv.cmp(rv),
            (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) => f64::total_cmp(lv, rv),
            (Expr::UnOp {op: lop, arg: larg, ..}, Expr::UnOp {op: rop, arg: rarg, ..}) =>
                lop.cmp(rop).then(larg.cmp(rarg)),
            ( Expr::BinOp {lhs: llhs, op: lop, rhs: lrhs, ..}
            , Expr::BinOp {lhs: rlhs, op: rop, rhs: rrhs, ..} ) =>
                llhs.cmp(rlhs).then(lop.cmp(rop)).then(lrhs.cmp(rrhs)),
            ( Expr::ReduceOp {op: lop, arg: larg, ..}
            , Expr::ReduceOp {op: rop, arg: rarg, ..} ) =>
                lop.cmp(rop).then(larg.cmp(rarg)),
            ( Expr::IfExpr {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::IfExpr {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.cmp(rcond).then(lthn.cmp(rthn)).then(lels.cmp(rels)),
            ( Expr::Subscript {target: ltarget, idx: lidx, ..}
            , Expr::Subscript {target: rtarget, idx: ridx, ..} ) =>
                ltarget.cmp(rtarget).then(lidx.cmp(ridx)),
            (Expr::Slice {lo: llo, hi: lhi, ..}, Expr::Slice {lo: rlo, hi: rhi, ..}) =>
                llo.cmp(rlo).then(lhi.cmp(rhi)),
            (Expr::Tuple {elems: lelems, ..}, Expr::Tuple {elems: relems, ..}) =>
                lelems.cmp(relems),
            (Expr::List {elems: lelems, ..}, Expr::List {elems: relems, ..}) =>
                lelems.cmp(relems),
            (Expr::Call {id: lid, args: largs, ..}, Expr::Call {id: rid, args: rargs, ..}) =>
                lid.cmp(rid).then(largs.cmp(rargs)),
            ( Expr::Callback {id: lid, args: largs, ..}
            , Expr::Callback {id: rid, args: rargs, ..} ) =>
                lid.cmp(rid).then(largs.cmp(rargs)),
            (Expr::Convert {e: le, ty: lty, ..}, Expr::Convert {e: re, ty: rty, ..}) =>
                le.cmp(re).then(lty.cmp(rty)),
            (Expr::Inline {e: le, ..}, Expr::Inline {e: re, ..}) => le.cmp(re),
            (Expr::StaticBackendEq {backend: lb, ..}, Expr::StaticBackendEq {backend: rb, ..}) =>
                lb.cmp(rb),
            ( Expr::StaticTypesEq {lhs: llhs, rhs: lrhs, ..}
            , Expr::StaticTypesEq {lhs: rlhs, rhs: rrhs, ..} ) =>
                llhs.cmp(rlhs).then(lrhs.cmp(rrhs)),
            (Expr::StaticFail {msg: lmsg, ..}, Expr::StaticFail {msg: rmsg, ..}) =>
                lmsg.cmp(rmsg),
            ( Expr::AllocShared {shape: lshape, sz: lsz, ..}
            , Expr::AllocShared {shape: rshape, sz: rsz, ..} ) =>
                lshape.cmp(rshape).then(lsz.cmp(rsz)),
            (lhs, rhs) => lhs.discriminator().cmp(&rhs.discriminator())
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

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} => i.clone(),
            Expr::String {i, ..} => i.clone(),
            Expr::Bool {i, ..} => i.clone(),
            Expr::Int {i, ..} => i.clone(),
            Expr::Float {i, ..} => i.clone(),
            Expr::UnOp {i, ..} => i.clone(),
            Expr::BinOp {i, ..} => i.clone(),
            Expr::ReduceOp {i, ..} => i.clone(),
            Expr::IfExpr {i, ..} => i.clone(),
            Expr::Subscript {i, ..} => i.clone(),
            Expr::Slice {i, ..} => i.clone(),
            Expr::Tuple {i, ..} => i.clone(),
            Expr::List {i, ..} => i.clone(),
            Expr::Call {i, ..} => i.clone(),
            Expr::Callback {i, ..} => i.clone(),
            Expr::Convert {i, ..} => i.clone(),
            Expr::GpuContext {i, ..} => i.clone(),
            Expr::Inline {i, ..} => i.clone(),
            Expr::Label {i, ..} => i.clone(),
            Expr::StaticBackendEq {i, ..} => i.clone(),
            Expr::StaticTypesEq {i, ..} => i.clone(),
            Expr::StaticFail {i, ..} => i.clone(),
            Expr::AllocShared {i, ..} => i.clone(),
        }
    }
}

impl Default for Expr {
    fn default() -> Expr {
        Expr::Var {id: Name::default(), ty: Type::default(), i: Info::default()}
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
            Expr::ReduceOp {op, arg, ty, i} => {
                let (acc, arg) = f(acc?, *arg)?;
                Ok((acc, Expr::ReduceOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::IfExpr {cond, thn, els, ty, i} => {
                let (acc, cond) = f(acc?, *cond)?;
                let (acc, thn) = f(acc, *thn)?;
                let (acc, els) = f(acc, *els)?;
                Ok((acc, Expr::IfExpr {
                    cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
                }))
            },
            Expr::Subscript {target, idx, ty, i} => {
                let (acc, target) = f(acc?, *target)?;
                let (acc, idx) = f(acc, *idx)?;
                Ok((acc, Expr::Subscript {
                    target: Box::new(target), idx: Box::new(idx), ty, i
                }))
            },
            Expr::Slice {lo, hi, ty, i} => {
                let (acc, lo) = lo.smap_accum_l_result(acc, &f)?;
                let (acc, hi) = hi.smap_accum_l_result(Ok(acc), &f)?;
                Ok((acc, Expr::Slice {lo, hi, ty, i}))
            },
            Expr::Tuple {elems, ty, i} => {
                let (acc, elems) = elems.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Tuple {elems, ty, i}))
            },
            Expr::List {elems, ty, i} => {
                let (acc, elems) = elems.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::List {elems, ty, i}))
            },
            Expr::Call {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Call {id, args, ty, i}))
            },
            Expr::Callback {id, args, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Callback {id, args, ty, i}))
            },
            Expr::Convert {e, ty, i} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty, i}))
            },
            Expr::Inline {e, ty, i} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Inline {e: Box::new(e), ty, i}))
            },
            Expr::AllocShared {shape, sz, ty, i} => {
                let (acc, shape) = shape.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::AllocShared {shape, sz, ty, i}))
            },
            Expr::Var {..} | Expr::String {..} | Expr::Bool {..} |
            Expr::Int {..} | Expr::Float {..} | Expr::GpuContext {..} |
            Expr::Label {..} | Expr::StaticBackendEq {..} |
            Expr::StaticTypesEq {..} | Expr::StaticFail {..} => {
                Ok((acc?, self))
            },
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
            Expr::ReduceOp {arg, ..} => f(acc?, arg),
            Expr::IfExpr {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::Subscript {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Slice {lo, hi, ..} => hi.sfold_result(lo.sfold_result(acc, &f), &f),
            Expr::Tuple {elems, ..} => elems.sfold_result(acc, &f),
            Expr::List {elems, ..} => elems.sfold_result(acc, &f),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::Callback {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
            Expr::Inline {e, ..} => f(acc?, e),
            Expr::AllocShared {shape, ..} => shape.sfold_result(acc, &f),
            Expr::Var {..} | Expr::String {..} | Expr::Bool {..} |
            Expr::Int {..} | Expr::Float {..} | Expr::GpuContext {..} |
            Expr::Label {..} | Expr::StaticBackendEq {..} |
            Expr::StaticTypesEq {..} | Expr::StaticFail {..} => acc
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, labels: Vec<String>, i: Info},
    Assign {dst: Expr, expr: Expr, labels: Vec<String>, i: Info},
    For {
        var: Name, lo: Expr, hi: Expr, step: Expr, body: Vec<Stmt>,
        labels: Vec<String>, i: Info
    },
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    Return {value: Expr, i: Info},
    WithGpuContext {body: Vec<Stmt>, i: Info},
    Expr {e: Expr, i: Info},
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::Return {i, ..} => i.clone(),
            Stmt::WithGpuContext {i, ..} => i.clone(),
            Stmt::Expr {i, ..} => i.clone(),
        }
    }
}

impl SMapAccum<Expr> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Expr) -> Result<(A, Expr), E>
    ) -> Result<(A, Stmt), E> {
        match self {
            Stmt::Definition {ty, id, expr, labels, i} => {
                let (acc, expr) = f(acc?, expr)?;
                Ok((acc, Stmt::Definition {ty, id, expr, labels, i}))
            },
            Stmt::Assign {dst, expr, labels, i} => {
                let (acc, dst) = f(acc?, dst)?;
                let (acc, expr) = f(acc, expr)?;
                Ok((acc, Stmt::Assign {dst, expr, labels, i}))
            },
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let (acc, lo) = f(acc?, lo)?;
                let (acc, hi) = f(acc, hi)?;
                let (acc, step) = f(acc, step)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, labels, i}))
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
            Stmt::WithGpuContext {..} => {
                Ok((acc?, self))
            },
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
            Stmt::For {lo, hi, step, ..} => f(f(f(acc?, lo)?, hi)?, step),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::Return {value, ..} => f(acc?, value),
            Stmt::Expr {e, ..} => f(acc?, e),
            Stmt::WithGpuContext {..} => acc,
        }
    }
}

impl SMapAccum<Stmt> for Stmt {
    fn smap_accum_l_result<A, E>(
        self,
        acc: Result<A, E>,
        f: impl Fn(A, Stmt) -> Result<(A, Stmt), E>
    ) -> Result<(A, Stmt), E> {
        match self {
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::For {var, lo, hi, step, body, labels, i}))
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
            Stmt::WithGpuContext {body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::WithGpuContext {body, i}))
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Expr {..} => {
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
            Stmt::While {body, ..} => body.sfold_result(acc, &f),
            Stmt::If {thn, els, ..} => els.sfold_result(thn.sfold_result(acc, &f), &f),
            Stmt::WithGpuContext {body, ..} => body.sfold_result(acc, &f),
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Expr {..} => acc
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
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::For {var, lo, hi, step, body, labels, i});
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
            Stmt::WithGpuContext {body, i} => {
                let body = body.sflatten_result(vec![], &f)?;
                acc.push(Stmt::WithGpuContext {body, i});
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Expr {..} => {
                acc.push(self);
            },
        };
        Ok(acc)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunDef {
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub res_ty: Type,
    pub i: Info,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Top {
    CallbackDecl {id: Name, params: Vec<Param>, i: Info},
    ExtDecl {
        id: Name, ext_id: String, params: Vec<Param>, res_ty: Type,
        target: Target, header: Option<String>, par: LoopPar, i: Info
    },
    FunDef {v: FunDef},
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ast {
    pub tops: Vec<Top>,
    pub main: FunDef,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;

    use strum::IntoEnumIterator;

    #[test]
    fn scalar_elem_size_unknown() {
        let ty = Type::Unknown;
        assert_eq!(ty.get_scalar_elem_size(), None);
    }

    #[test]
    fn scalar_elem_size_scalar_tensor() {
        let ty = scalar(ElemSize::I64);
        assert_eq!(ty.get_scalar_elem_size(), Some(&ElemSize::I64));
    }

    #[test]
    fn scalar_elem_size_vector() {
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I64),
            shape: vec![TensorShape::Num {n: 10}]
        };
        assert_eq!(ty.get_scalar_elem_size(), None);
    }

    #[test]
    fn scalar_elem_size_multi_dim_tensor() {
        let ty = Type::Tensor {
            sz: fixed_elem_sz(ElemSize::I64),
            shape: vec![
                TensorShape::Num {n: 10},
                TensorShape::Num {n: 20}
            ]
        };
        assert_eq!(ty.get_scalar_elem_size(), None);
    }

    #[test]
    fn compare_types() {
        for (i, ty1) in Type::iter().enumerate() {
            for (j, ty2) in Type::iter().enumerate() {
                assert_eq!(ty1.cmp(&ty2), i.cmp(&j));
            }
        }
    }

    #[test]
    fn compare_expr_discriminators() {
        for (i, e1) in Expr::iter().enumerate() {
            for (j, e2) in Expr::iter().enumerate() {
                assert_eq!(e1.discriminator().cmp(&e2.discriminator()), i.cmp(&j));
            }
        }
    }

    #[test]
    fn compare_exprs() {
        for (i, e1) in Expr::iter().enumerate() {
            for (j, e2) in Expr::iter().enumerate() {
                assert_eq!(e1.cmp(&e2), i.cmp(&j));
            }
        }
    }
}
