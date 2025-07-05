use crate::py::ast::BinOp;
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
        PrettyPrintEnv {
            strs: BTreeSet::new(),
            vars: BTreeMap::new(),
            ignore_symbols: false,
            indent: 0,
            indent_increment: DEFAULT_INDENT,
        }
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
        Some(inner_op) => {
            if p(inner_op.precedence().cmp(&outer_op.precedence())) {
                format!("({s})")
            } else {
                s
            }
        },
        None => s
    }
}

pub fn parenthesize_if_lower_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_predicate(inner_op_opt, outer_op, s, |p| p == Ordering::Less)
}

pub fn parenthesize_if_lower_or_same_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_predicate(inner_op_opt, outer_op, s, |p| p != Ordering::Greater)
}

pub fn print_if_condition<'a, E: PrettyPrint + 'a, S: Clone + PrettyPrint + 'a>(
    env: PrettyPrintEnv,
    cond: &E,
    thn: &'a Vec<S>,
    els: &'a Vec<S>,
    extract_elseif: impl Fn(Vec<S>) -> Option<(E, Vec<S>, Vec<S>)>
) -> (PrettyPrintEnv, String) {
    let indent = env.print_indent();
    let (env, cond) = cond.pprint(env);
    let env = env.incr_indent();
    let (env, thn_str) = pprint_iter(thn.iter(), env, "\n");
    let (env, thn, els) = match extract_elseif(els.clone()) {
        Some((elif_cond, elif_thn, elif_els)) => {
            let (env, elif_cond) = elif_cond.pprint(env);
            let (env, elif_thn) = pprint_iter(elif_thn.iter(), env, "\n");
            let s = format!(
                "{0}\n{1}}} else if ({2}) {{\n{3}",
                thn_str, indent, elif_cond, elif_thn
            );
            (env, s, elif_els)
        },
        None => (env, thn_str, els.clone())
    };
    let (env, s) = if els.is_empty() {
        let s = format!("{0}if ({1}) {{\n{2}\n{0}}}", indent, cond, thn);
        (env, s)
    } else {
        let (env, els) = pprint_iter(els.iter(), env, "\n");
        let s = format!(
            "{0}if ({1}) {{\n{2}\n{0}}} else {{\n{3}\n{0}}}",
            indent, cond, thn, els
        );
        (env, s)
    };
    let env = env.decr_indent();
    (env, s)
}

#[cfg(test)]
mod test {
    use super::*;

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
}
