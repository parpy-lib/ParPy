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

    pub fn sym_str(s: &str) -> Name {
        Name {s: s.to_string(), sym: Some(gensym())}
    }

    pub fn with_new_sym(self) -> Name {
        let Name {s, ..} = self;
        let sym = Some(gensym());
        Name {s, sym}
    }

    pub fn has_sym(&self) -> bool {
        self.sym.is_some()
    }

    pub fn get_str<'a>(&'a self) -> &'a String {
        &self.s
    }

    pub fn print_with_sym(&self) -> String {
        if let Some(sym) = self.sym {
            format!("{0}_{1}", self.s, sym)
        } else {
            self.s.clone()
        }
    }
}

impl Default for Name {
    fn default() -> Name {
        Name::new("".to_string())
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

#[cfg(test)]
mod test {
    use super::*;

    use std::cmp::Ordering;

    fn name(s: &str) -> Name {
        Name::new(s.to_string())
    }

    #[test]
    fn default_name_no_sym() {
        assert_eq!(Name::default().sym, None);
    }

    #[test]
    fn sym_str_has_sym() {
        assert!(Name::sym_str("").sym.is_some());
    }

    #[test]
    fn unsymb_name_eq() {
        assert_eq!(name("a"), name("a"));
    }

    #[test]
    fn unsymb_name_neq() {
        let n1 = name("a");
        let n2 = name("aa");
        assert!(n1 != n2);
    }

    #[test]
    fn symb_name_eq() {
        let n = Name::sym_str("abc");
        assert_eq!(n, n);
    }

    #[test]
    fn distinct_symb_neq() {
        let n1 = Name::sym_str("a");
        let n2 = Name::sym_str("a");
        assert!(n1 != n2);
    }

    #[test]
    fn symb_unsymb_neq() {
        let n1 = Name::sym_str("a");
        let n2 = name("a");
        assert!(n1 != n2);
    }

    #[test]
    fn print_unsymb() {
        assert_eq!(name("abc").print_with_sym(), "abc");
    }

    #[test]
    fn print_symb() {
        let n = Name::sym_str("abc");
        let s = n.sym.unwrap_or(0);
        assert_eq!(n.print_with_sym(), format!("abc_{s}"));
    }

    #[test]
    fn distinct_new_symb() {
        let n1 = Name::sym_str("a");
        let n2 = n1.clone().with_new_sym();
        assert!(n1 != n2);
    }

    #[test]
    fn unsymb_ordering() {
        let n1 = name("abc");
        let n2 = name("def");
        assert_eq!(n1.cmp(&n2), Ordering::Less);
    }

    #[test]
    fn symb_greater_than_unsymb() {
        let n1 = Name::sym_str("abc");
        let n2 = name("def");
        assert_eq!(n1.cmp(&n2), Ordering::Greater);
    }

    #[test]
    fn symb_unsymb_ordering_reversed() {
        let n1 = Name::sym_str("abc");
        let n2 = name("def");
        assert_eq!(n2.cmp(&n1), Ordering::Less);
    }

    #[test]
    fn symb_ordering() {
        let n1 = Name::sym_str("abc");
        let n2 = Name::sym_str("def");
        assert_eq!(n1.cmp(&n2), n1.sym.cmp(&n2.sym));
    }
}
