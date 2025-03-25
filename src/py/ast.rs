use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

use strum_macros::EnumIter;
use itertools::Itertools;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum ElemSize {
    Bool, I8, I16, I32, I64, F16, F32, F64
}

impl ElemSize {
    pub fn is_boolean(&self) -> bool {
        match self {
            ElemSize::Bool => true,
            _ => false
        }
    }

    pub fn is_signed_integer(&self) -> bool {
        match self {
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => true,
            _ => false
        }
    }

    pub fn is_floating_point(&self) -> bool {
        match self {
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => true,
            _ => false
        }
    }
}

impl fmt::Display for ElemSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElemSize::Bool => write!(f, "bool"),
            ElemSize::I8 => write!(f, "int8"),
            ElemSize::I16 => write!(f, "int16"),
            ElemSize::I32 => write!(f, "int32"),
            ElemSize::I64 => write!(f, "int64"),
            ElemSize::F16 => write!(f, "float16"),
            ElemSize::F32 => write!(f, "float32"),
            ElemSize::F64 => write!(f, "float64"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    String,
    Tensor {sz: ElemSize, shape: Vec<i64>},
    Tuple {elems: Vec<Type>},
    Dict {fields: BTreeMap<String, Type>},
    Unknown
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Tensor {sz, shape} if shape.is_empty() => Some(sz),
            _ => None
        }
    }

    pub fn get_dict_type_fields(&self) -> BTreeMap<String, Type> {
        if let Type::Dict {fields} = self {
            fields.clone()
        } else {
            panic!("Parir internal error: expected dictionary type, found {self}")
        }
    }
}

impl Ord for Type {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Type::String, Type::String) => Ordering::Equal,
            (Type::String, _) => Ordering::Less,
            (Type::Tensor {..}, Type::String) =>
                Ordering::Greater,
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                lsz.cmp(rsz).then(lsh.cmp(rsh))
            },
            (Type::Tensor {..}, _) => Ordering::Less,
            (Type::Tuple {..}, Type::Dict {..} | Type::Unknown) => Ordering::Less,
            (Type::Tuple {elems: lelems}, Type::Tuple {elems: relems}) =>
                lelems.cmp(relems),
            (Type::Tuple {..}, _) => Ordering::Greater,
            (Type::Dict {..}, Type::Unknown) => Ordering::Less,
            (Type::Dict {fields: lfields}, Type::Dict {fields: rfields}) =>
                lfields.iter()
                    .zip(rfields.iter())
                    .fold(Ordering::Equal, |acc, ((lk, lv), (rk, rv))| {
                        acc.then(lk.cmp(rk)).then(lv.cmp(rv))
                    }),
            (Type::Dict {..}, _) => Ordering::Greater,
            (Type::Unknown, Type::Unknown) => Ordering::Equal,
            (Type::Unknown, _) => Ordering::Greater,
        }
    }
}

impl PartialOrd for Type {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Type {}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::String => write!(f, "string"),
            Type::Tensor {sz, shape} if shape.is_empty() => write!(f, "{sz}"),
            Type::Tensor {sz, shape} => {
                let sh = shape.iter().map(|i| i.to_string()).join(",");
                write!(f, "tensor<{sz}>[{sh}]")
            },
            Type::Unknown => write!(f, "?"),
            Type::Tuple {elems} => {
                let elems = elems.iter()
                    .map(|e| format!("{e}"))
                    .join(",");
                write!(f, "({elems})")
            },
            Type::Dict {fields} => {
                let fields = fields.iter()
                    .map(|(k, v)| format!("{k} {v}"))
                    .join(",");
                write!(f, "dict {{{fields}}}")
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Builtin {
    Exp, Inf, Log, Max, Min, Abs, Cos, Sin, Sqrt, Tanh, Atan2,
    Sum, Prod,
    Convert {sz: ElemSize}, Label, GpuContext, Ext {id: String}
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
            Type::String | Type::Tensor {..} | Type::Unknown => Ok((acc?, self))
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
            Type::String | Type::Tensor {..} | Type::Unknown => acc
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOp {
    Sub, Not, BitNeg, Exp, Log, Cos, Sin, Sqrt, Tanh, Abs
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOp {
    Add, Sub, Mul, FloorDiv, Div, Rem, Pow, And, Or,
    BitAnd, BitOr, BitXor, BitShl, BitShr,
    Eq, Neq, Leq, Geq, Lt, Gt,
    Max, Min, Atan2
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    String {v: String, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    IfExpr {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    Subscript {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Slice {lo: Option<Box<Expr>>, hi: Option<Box<Expr>>, ty: Type, i: Info},
    Tuple {elems: Vec<Expr>, ty: Type, i: Info},
    NeutralElement {op: BinOp, tyof: Box<Expr>, i: Info},
    Builtin {func: Builtin, args: Vec<Expr>, axis: Option<i64>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},
}

impl Expr {
    pub fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::String {ty, ..} => ty,
            Expr::Bool {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::IfExpr {ty, ..} => ty,
            Expr::Subscript {ty, ..} => ty,
            Expr::Slice {ty, ..} => ty,
            Expr::Tuple {ty, ..} => ty,
            Expr::NeutralElement {tyof, ..} => tyof.get_type(),
            Expr::Builtin {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
        }
    }

    pub fn discriminator(&self) -> u8 {
        match self {
            Expr::Var {..} => 0,
            Expr::String {..} => 1,
            Expr::Bool {..} => 2,
            Expr::Int {..} => 3,
            Expr::Float {..} => 4,
            Expr::UnOp {..} => 5,
            Expr::BinOp {..} => 6,
            Expr::IfExpr {..} => 7,
            Expr::Subscript {..} => 8,
            Expr::Slice {..} => 9,
            Expr::Tuple {..} => 10,
            Expr::NeutralElement {..} => 11,
            Expr::Builtin {..} => 12,
            Expr::Convert {..} => 13,
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
            Expr::IfExpr {cond, thn, els, ty, ..} => Expr::IfExpr {cond, thn, els, ty, i},
            Expr::Subscript {target, idx, ty, ..} => Expr::Subscript {target, idx, ty, i},
            Expr::Slice {lo, hi, ty, ..} => Expr::Slice {lo, hi, ty, i},
            Expr::Tuple {elems, ty, ..} => Expr::Tuple {elems, ty, i},
            Expr::NeutralElement {op, tyof, ..} => Expr::NeutralElement {op, tyof, i},
            Expr::Builtin {func, args, axis, ty, ..} => Expr::Builtin {func, args, axis, ty, i},
            Expr::Convert {e, ty} => Expr::Convert {e: Box::new(e.with_info(i)), ty},
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
            ( Expr::NeutralElement {op: lop, tyof: ltyof, ..}
            , Expr::NeutralElement {op: rop, tyof: rtyof, ..} ) =>
                lop.cmp(rop).then(ltyof.cmp(rtyof)),
            ( Expr::Builtin {func: lfunc, args: largs, ..}
            , Expr::Builtin {func: rfunc, args: rargs, ..} ) =>
                lfunc.cmp(rfunc).then(largs.cmp(rargs)),
            (Expr::Convert {e: le, ty: lty}, Expr::Convert {e: re, ty: rty}) =>
                le.cmp(re).then(lty.cmp(rty)),
            (lhs, rhs) => lhs.discriminator().cmp(&rhs.discriminator())
        }
    }
}

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
            Expr::IfExpr {i, ..} => i.clone(),
            Expr::Subscript {i, ..} => i.clone(),
            Expr::Slice {i, ..} => i.clone(),
            Expr::Tuple {i, ..} => i.clone(),
            Expr::NeutralElement {i, ..} => i.clone(),
            Expr::Builtin {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
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
            Expr::Builtin {func, args, axis, ty, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Expr::Builtin {func, args, axis, ty, i}))
            },
            Expr::NeutralElement {op, tyof, i} => {
                let (acc, tyof) = f(acc?, *tyof)?;
                Ok((acc, Expr::NeutralElement {op, tyof: Box::new(tyof), i}))
            },
            Expr::Convert {e, ty} => {
                let (acc, e) = f(acc?, *e)?;
                Ok((acc, Expr::Convert {e: Box::new(e), ty}))
            },
            Expr::Var {..} | Expr::String {..} | Expr::Bool {..} |
            Expr::Int {..} | Expr::Float {..} => {
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
            Expr::IfExpr {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::Subscript {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Slice {lo, hi, ..} => hi.sfold_result(lo.sfold_result(acc, &f), &f),
            Expr::Tuple {elems, ..} => elems.sfold_result(acc, &f),
            Expr::NeutralElement {tyof, ..} => f(acc?, tyof),
            Expr::Builtin {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
            Expr::Var {..} | Expr::String {..} | Expr::Bool {..} |
            Expr::Int {..} | Expr::Float {..} => acc
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, labels: Vec<String>, i: Info},
    Assign {dst: Expr, expr: Expr, labels: Vec<String>, i: Info},
    For {
        var: Name, lo: Expr, hi: Expr, step: i64, body: Vec<Stmt>,
        labels: Vec<String>, i: Info
    },
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    WithGpuContext {body: Vec<Stmt>, i: Info},
    Scope {body: Vec<Stmt>, i: Info},
    Call {func: String, args: Vec<Expr>, i: Info},
    Label {label: String, i: Info}
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::WithGpuContext {i, ..} => i.clone(),
            Stmt::Scope {i, ..} => i.clone(),
            Stmt::Call {i, ..} => i.clone(),
            Stmt::Label {i, ..} => i.clone(),
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
            Stmt::Call {func, args, i} => {
                let (acc, args) = args.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::Call {func, args, i}))
            },
            Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Label {..} =>
                Ok((acc?, self)),
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
            Stmt::For {lo, hi, ..} => f(f(acc?, lo)?, hi),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::Call {args, ..} => args.sfold_result(acc, &f),
            Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Label {..} => acc,
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
            Stmt::Scope {body, i} => {
                let (acc, body) = body.smap_accum_l_result(acc, &f)?;
                Ok((acc, Stmt::Scope {body, i}))
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Label {..} |
            Stmt::Call {..} => {
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
            Stmt::Scope {body, ..} => body.sfold_result(acc, &f),
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Label {..} |
            Stmt::Call {..} => acc
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug)]
pub struct FunDef {
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub i: Info
}
