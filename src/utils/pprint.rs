use crate::utils::ast::{ExprType, BinOp, UnOp};
use crate::utils::name::Name;

use itertools::Itertools;
use rand::distributions::{Alphanumeric, DistString};

use std::collections::{BTreeMap, BTreeSet};
use std::cmp::Ordering;

pub const DEFAULT_INDENT: usize = 2;

#[derive(Debug)]
pub struct PrettyPrintEnv {
    strs: BTreeSet<String>,
    vars: BTreeMap<Name, String>,
    ignore_symbols: bool,
    indent: usize,
    indent_increment: usize,
}

impl PrettyPrintEnv {
    pub fn new() -> Self {
        PrettyPrintEnv::default()
    }

    pub fn incr_indent(self) -> Self {
        let indent = self.indent + self.indent_increment;
        PrettyPrintEnv {indent, ..self}
    }

    pub fn decr_indent(self) -> Self {
        let indent = self.indent - self.indent_increment;
        PrettyPrintEnv {indent, ..self}
    }

    pub fn print_indent(&self) -> String {
        " ".repeat(self.indent)
    }
}

impl Default for PrettyPrintEnv {
    fn default() -> PrettyPrintEnv {
        PrettyPrintEnv {
            strs: BTreeSet::new(),
            vars: BTreeMap::new(),
            ignore_symbols: false,
            indent: 0,
            indent_increment: DEFAULT_INDENT,
        }
    }
}

pub trait PrettyPrint {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String);

    fn pprint_default(&self) -> String {
        let (_, s) = self.pprint(PrettyPrintEnv::new());
        s
    }

    fn pprint_ignore_symbols(&self) -> String {
        let mut env = PrettyPrintEnv::new();
        env.ignore_symbols = true;
        let (_, s) = self.pprint(env);
        s
    }
}

fn rand_alphanum(n: usize) -> String {
    Alphanumeric.sample_string(&mut rand::thread_rng(), n)
}

fn alloc_free_string(mut env: PrettyPrintEnv, id: &Name) -> (PrettyPrintEnv, String) {
    let mut s = id.get_str().clone();
    if env.strs.contains(&s) {
        s = id.print_with_sym();
        while env.strs.contains(&s) {
            s = format!("{0}_{1}", id.get_str(), rand_alphanum(5));
        }
    }
    env.strs.insert(s.clone());
    env.vars.insert(id.clone(), s.clone());
    (env, s)
}

impl PrettyPrint for Name {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        if env.ignore_symbols {
            (env, format!("{}", self.get_str()))
        } else if let Some(s) = env.vars.get(&self) {
            let s = s.clone();
            (env, s)
        } else {
            alloc_free_string(env, self)
        }
    }
}

pub fn pprint_iter<'a, T: PrettyPrint + 'a, I: Iterator<Item=&'a T>>(
    it: I,
    env: PrettyPrintEnv,
    separator: &str
) -> (PrettyPrintEnv, String) {
    let (env, strs) = it.fold((env, vec![]), |(env, mut strs), v| {
            let (env, v) = v.pprint(env);
            strs.push(v);
            (env, strs)
        });
    (env, strs.into_iter().join(separator))
}

// Reusable functionality for printing floating-point numbers, properly parenthesized expressions
// with respect to operator precedence, and nested if-statements.

pub fn print_float(
    env: PrettyPrintEnv,
    v: &f64,
    inf_str: &str
) -> (PrettyPrintEnv, String) {
    if v.is_infinite() {
        if v.is_sign_negative() {
            (env, format!("-{inf_str}"))
        } else {
            (env, inf_str.to_string())
        }
    } else {
        // Debug printing adds a trailing '.0' to floats without a decimal component, which is what
        // we want to distinguish them from integers.
        (env, format!("{v:?}"))
    }
}

fn parenthesize_if_predicate(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String,
    p: impl Fn(Ordering) -> bool
) -> String {
    match inner_op_opt {
        Some(inner_op) if p(BinOp::precedence(&inner_op, outer_op)) => {
            format!("({s})")
        },
        _ => s
    }
}

pub fn parenthesize_if_lt_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_predicate(inner_op_opt, outer_op, s, |p| p == Ordering::Less)
}

pub fn parenthesize_if_le_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_predicate(inner_op_opt, outer_op, s, |p| p != Ordering::Greater)
}

pub trait PrettyPrintUnOp<T>: PrettyPrint + ExprType<T> + Sized {
    fn extract_unop<'a>(&'a self) -> Option<(&'a UnOp, &'a Self)>;
    fn is_function(op: &UnOp) -> bool;
    fn print_unop(op: &UnOp, argty: &T) -> String;

    fn print_parenthesized_unop(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (op, arg) = self.extract_unop().unwrap();
        let op_str = Self::print_unop(op, arg.get_type());
        let (env, arg_str) = arg.pprint(env);
        if Self::is_function(op) {
            (env, format!("{op_str}({arg_str})"))
        } else {
            (env, format!("{op_str}{arg_str}"))
        }
    }
}

pub enum Assoc { Left, Right }

pub trait PrettyPrintBinOp<T>: PrettyPrint + ExprType<T> + Sized {
    fn extract_binop<'a>(&'a self) -> Option<(&'a Self, &'a BinOp, &'a Self, &'a T)>;
    fn is_infix(op: &BinOp, argty: &T) -> bool;
    fn print_binop(op: &BinOp, argty: &T, ty: &T) -> String;
    fn associativity(op: &BinOp) -> Assoc;

    fn try_get_binop<'a>(&'a self) -> Option<BinOp> {
        if let Some((_, op, _, _)) = self.extract_binop() {
            Some(op.clone())
        } else {
            None
        }
    }

    fn print_parenthesized_binop(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (lhs, op, rhs, ty) = self.extract_binop().unwrap();
        let argty = lhs.get_type();
        let (env, lhs_str) = lhs.pprint(env);
        let op_str = Self::print_binop(op, argty, ty);
        let (env, rhs_str) = rhs.pprint(env);
        if Self::is_infix(op, argty) {
            let lhs_op = lhs.try_get_binop();
            let rhs_op = rhs.try_get_binop();
            let (lstr, rstr) = match Self::associativity(op) {
                Assoc::Left => {
                    ( parenthesize_if_lt_precedence(lhs_op, op, lhs_str)
                    , parenthesize_if_le_precedence(rhs_op, op, rhs_str) )
                },
                Assoc::Right => {
                    ( parenthesize_if_le_precedence(lhs_op, op, lhs_str)
                    , parenthesize_if_lt_precedence(rhs_op, op, rhs_str) )
                }
            };
            (env, format!("{lstr} {op_str} {rstr}"))
        } else {
            (env, format!("{op_str}({lhs_str}, {rhs_str})"))
        }
    }
}

pub trait PrettyPrintCond<E: PrettyPrint>: PrettyPrint + Sized {
    fn extract_if<'a>(&'a self) -> Option<(&'a E, &'a Vec<Self>, &'a Vec<Self>)>;
    fn extract_elseif<'a>(&'a self) -> Option<(&'a E, &'a Vec<Self>, &'a Vec<Self>)>;

    fn print_cond(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (cond, thn, els) = self.extract_if().unwrap();
        let indent = env.print_indent();
        let (env, cond_str) = cond.pprint(env);
        let env = env.incr_indent();
        let (env, thn_str) = pprint_iter(thn.iter(), env, "\n");
        let (env, thn_str, els) = match self.extract_elseif() {
            Some((elif_cond, elif_thn, elif_els)) => {
                let (env, elif_cond) = elif_cond.pprint(env);
                let (env, elif_thn) = pprint_iter(elif_thn.iter(), env, "\n");
                let s = format!(
                    "{0}\n{1}}} else if ({2}) {{\n{3}",
                    thn_str, indent, elif_cond, elif_thn
                );
                (env, s, elif_els)
            },
            None => (env, thn_str, els)
        };
        let (env, s) = if els.is_empty() {
            (env, format!("{0}if ({1}) {{\n{2}\n{0}}}", indent, cond_str, thn_str))
        } else {
            let (env, els_str) = pprint_iter(els.iter(), env, "\n");
            (env, format!(
                "{0}if ({1}) {{\n{2}\n{0}}} else {{\n{3}\n{0}}}",
                indent, cond_str, thn_str, els_str
            ))
        };
        let env = env.decr_indent();
        (env, s)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_root_indent_print() {
        let env = PrettyPrintEnv::new();
        assert_eq!(env.print_indent(), "");
    }

    #[test]
    fn test_incr_indent_print() {
        let env = PrettyPrintEnv::new().incr_indent();
        assert_eq!(env.print_indent(), " ".repeat(DEFAULT_INDENT));
    }

    #[test]
    fn test_incr_custom_indent_print() {
        let mut env = PrettyPrintEnv::new();
        env.indent_increment = 4;
        let env = env.incr_indent();
        assert_eq!(env.print_indent(), " ".repeat(4));
    }

    #[test]
    fn test_multi_incr_indent_print() {
        let env = PrettyPrintEnv::new().incr_indent().incr_indent();
        assert_eq!(env.print_indent(), " ".repeat(2 * DEFAULT_INDENT));
    }

    #[test]
    fn test_incr_decr_indent_print() {
        let env = PrettyPrintEnv::new().incr_indent().decr_indent();
        assert_eq!(env.print_indent(), "");
    }

    #[test]
    fn print_integer_float() {
        let env = PrettyPrintEnv::new();
        let (_, s) = print_float(env, &1.0, "inf");
        assert_eq!(s, "1.0");
    }

    #[test]
    fn print_inf_float() {
        let env = PrettyPrintEnv::new();
        let (_, s) = print_float(env, &f64::INFINITY, "inf");
        assert_eq!(s, "inf");
    }

    #[test]
    fn print_neg_inf_float() {
        let env = PrettyPrintEnv::new();
        let (_, s) = print_float(env, &f64::NEG_INFINITY, "inf");
        assert_eq!(s, "-inf");
    }

    #[test]
    fn test_distinct_names_print() {
        let n1 = Name::sym_str("x");
        let n2 = n1.clone().with_new_sym();
        let (env, s1) = n1.pprint(PrettyPrintEnv::new());
        let (env, s2) = n2.pprint(env);
        assert_eq!(env.strs.len(), 2);
        assert!(s1 != s2);
        let (_, s3) = n2.pprint(env);
        assert_eq!(s2, s3);
    }

    #[test]
    fn test_print_paren_predicate() {
        let s = "a + b".to_string();
        let s = parenthesize_if_lt_precedence(Some(BinOp::Add), &BinOp::Mul, s);
        assert_eq!(s, "(a + b)");
    }

    #[test]
    fn test_no_paren_predicate() {
        let s = "a + b".to_string();
        let s = parenthesize_if_lt_precedence(Some(BinOp::Add), &BinOp::Add, s);
        assert_eq!(s, "a + b");
    }

    #[test]
    fn test_paren_predicate_same_precedence() {
        let s = "a + b".to_string();
        let s = parenthesize_if_le_precedence(Some(BinOp::Add), &BinOp::Add, s);
        assert_eq!(s, "(a + b)");
    }
}
