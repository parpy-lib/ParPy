use std::sync::atomic;
use std::cmp;
use std::fmt;
use std::hash;

pub type Sym = i64;

static COUNTER: atomic::AtomicI64 = atomic::AtomicI64::new(0);

fn gensym() -> Sym {
    let v = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
    v
}

#[derive(Clone, Debug)]
pub struct Name {
    s: String,
    sym: Option<Sym>
}

impl Name {
    pub fn new(s: String) -> Name {
        Name {s, sym: None}
    }

    pub fn new_sym(self, n: Name) -> Name {
        let Name {s, ..} = n;
        let sym = Some(gensym());
        Name {s, sym}
    }

    pub fn with_sym(self, n: Name, sym: Sym) -> Name {
        let Name {s, ..} = n;
        Name {s, sym: Some(sym)}
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{0}", self.s)
    }
}

impl Ord for Name {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (self.sym, other.sym) {
            (Some(l), Some(r)) => l.cmp(&r),
            (Some(_), None) => cmp::Ordering::Greater,
            (None, Some(_)) => cmp::Ordering::Less,
            (None, None) => self.s.cmp(&other.s),
        }
    }
}

impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Self) -> bool {
        match (self.sym, other.sym) {
            (Some(l), Some(r)) => l.eq(&r),
            (Some(_), None) => false,
            (None, Some(_)) => false,
            (None, None) => self.s.eq(&other.s)
        }
    }
}

impl Eq for Name {}

impl hash::Hash for Name {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        if self.sym.is_some() {
            self.s.hash(state);
        } else {
            self.sym.hash(state);
        }
    }
}
