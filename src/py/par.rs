use super::ast::*;
use crate::py_runtime_error;
use crate::par::LoopPar;
use crate::utils::err::*;
use crate::utils::smap::SFold;

use pyo3::prelude::*;

use std::collections::BTreeMap;

/// We consider the AST to contain parallelism if it contains:
/// 1. A label which is associated with a parallel argument.
/// 2. A GPU context introduction, corresponding to the use of 'with parir.gpu:'.
fn ensure_parallelism_stmt(
    acc: bool,
    s: &Stmt,
    par: &BTreeMap<String, LoopPar>
) -> bool {
    match s {
        Stmt::Definition {labels, ..} | Stmt::Assign {labels, ..} |
        Stmt::For {labels, ..} if labels.iter().any(|l| par.contains_key(l)) => true,
        Stmt::Label {label, ..} if par.contains_key(label) => true,
        Stmt::WithGpuContext {..} => true,
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::For {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Scope {..} | Stmt::Label {..} |
        Stmt::Call {..} => {
            s.sfold(acc, |acc, s| ensure_parallelism_stmt(acc, s, par))
        }
    }
}

/// Ensures that the provided function AST contains at least some use of parallelism. If it does
/// not, we produce a clear error message explaining what the problem is and how to fix it.
pub fn ensure_parallelism(
    ast: &Ast,
    par: &BTreeMap<String, LoopPar>
) -> PyResult<()> {
    let def = ast.last().unwrap();
    let contains_parallelism = def.body.sfold(false, |acc, s| {
        ensure_parallelism_stmt(acc, s, par)
    });
    if !contains_parallelism {
        let msg = format!(
            "The function {0} does not contain any parallelism, which is not \
             allowed. Try adding a label ('parir.label') in front of a \
             parallelizable statement, and specify its parallelism using the \
             'parallelize' keyword argument. Alternatively, if you want to run \
             sequential code on the GPU, wrap the code in a GPU context as \
             'with parir.gpu: ...'.",
             def.id
        );
        py_runtime_error!(def.i, "{}", msg)
    } else {
        Ok(())
    }
}
