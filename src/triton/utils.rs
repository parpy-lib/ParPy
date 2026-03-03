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

// Attempts to extrac the name of the initialized variable and the step size used in its
// assignment.
pub fn try_extract_blocked_initialization(s: &Stmt) -> Option<(Name, i128)> {
    match s {
        Stmt::Definition {dst, expr: Expr::BinOp {rhs, ..}, ..} => {
            match &**rhs {
                Expr::Arange {..} => Some((dst.clone(), 1)),
                Expr::BinOp {lhs, rhs, ..} => {
                    match (&**lhs, &**rhs) {
                        (Expr::Arange {..}, Expr::Int {v, ..}) => {
                            Some((dst.clone(), *v))
                        },
                        _ => None
                    }
                },
                _ => None
            }
        },
        _ => None
    }
}

pub fn try_extract_blocked_var(s: &Stmt) -> Option<Name> {
    try_extract_blocked_initialization(s).map(|o| o.0)
}
