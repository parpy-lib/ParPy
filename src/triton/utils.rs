use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

fn contains_arange_helper(acc: bool, e: &Expr) -> bool {
    match e {
        Expr::Arange {..} => true,
        _ => e.sfold(acc, contains_arange_helper)
    }
}

pub fn contains_arange(e: &Expr) -> bool {
    contains_arange_helper(false, e)
}

pub fn try_extract_blocked_var(s: &Stmt) -> Option<Name> {
    match s {
        Stmt::Definition {dst, expr, ..} if contains_arange(&expr) => {
            Some(dst.clone())
        },
        _ => None
    }
}
