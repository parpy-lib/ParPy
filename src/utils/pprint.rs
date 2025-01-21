use crate::utils::name::Name;

use itertools::Itertools;
use rand::distributions::{Alphanumeric, DistString};

use std::collections::{BTreeMap, BTreeSet};

pub struct PrettyPrintEnv {
    strs: BTreeSet<String>,
    vars: BTreeMap<Name, String>,
    indent: usize,
    indent_increment: usize,
}

impl PrettyPrintEnv {
    pub fn new() -> Self {
        PrettyPrintEnv {
            strs: BTreeSet::new(),
            vars: BTreeMap::new(),
            indent: 0,
            indent_increment: 2,
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
        (0..self.indent).map(|_| ' ').collect::<String>()
    }
}

fn rand_alphanum(n: usize) -> String {
    Alphanumeric.sample_string(&mut rand::thread_rng(), n)
}

fn alloc_free_string(mut env: PrettyPrintEnv, id: &Name) -> (PrettyPrintEnv, String) {
    let mut s = id.print_with_sym();
    while env.strs.contains(&s) {
        s = format!("{0}_{1}", id.get_str(), rand_alphanum(5));
    }
    env.strs.insert(s.clone());
    env.vars.insert(id.clone(), s.clone());
    (env, s)
}

pub fn pprint_var(env: PrettyPrintEnv, id: &Name) -> (PrettyPrintEnv, String) {
    if let Some(x) = &env.vars.get(id) {
        let s = x.to_string();
        (env, s)
    } else {
        let (mut env, s) = alloc_free_string(env, id);
        env.vars.insert(id.clone(), s.clone());
        (env, s)
    }
}

pub trait PrettyPrint {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String);
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
