use super::ast::*;
use crate::py_runtime_error;
use crate::par::LoopPar;
use crate::utils::err::*;
use crate::utils::smap::SFold;

use pyo3::prelude::*;

use std::collections::BTreeMap;

/// We consider the AST to contain parallelism if it contains:
/// 1. A label which is associated with a parallel argument.
/// 2. A GPU context introduction, corresponding to the use of 'with parpy.gpu:'.
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
        Stmt::While {..} | Stmt::If {..} | Stmt::Return {..} | Stmt::Label {..} |
        Stmt::Call {..} => {
            s.sfold(acc, |acc, s| ensure_parallelism_stmt(acc, s, par))
        }
    }
}

/// Ensures that the provided function AST contains at least some use of parallelism. If it does
/// not, we produce a clear error message explaining what the problem is and how to fix it.
pub fn ensure_parallelism(
    def: &FunDef,
    par: &BTreeMap<String, LoopPar>
) -> PyResult<()> {
    let contains_parallelism = def.body.sfold(false, |acc, s| {
        ensure_parallelism_stmt(acc, s, par)
    });
    if !contains_parallelism {
        let msg = format!(
            "The function {0} does not contain any parallelism, which is not \
             allowed. Try adding a label ('parpy.label') in front of a \
             parallelizable statement, and specify its parallelism using the \
             'parallelize' keyword argument. Alternatively, if you want to run \
             sequential code on the GPU, wrap the code in a GPU context as \
             'with parpy.gpu: ...'.",
             def.id
        );
        py_runtime_error!(def.i, "{}", msg)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;
    use crate::utils::info::*;

    fn fun_with_body(body: Vec<Stmt>) -> FunDef {
        FunDef {
            id: id("f"),
            params: vec![],
            body,
            res_ty: Type::Void,
            i: Info::default()
        }
    }

    #[test]
    fn ensure_parallelism_label() {
        let def = fun_with_body(vec![label("x")]);
        let p = vec![
            ("x".to_string(), LoopPar::default())
        ].into_iter().collect::<BTreeMap<String, LoopPar>>();
        assert!(ensure_parallelism(&def, &p).is_ok());
    }

    #[test]
    fn ensure_parallelism_with_gpu_ctx() {
        let def = fun_with_body(vec![Stmt::WithGpuContext {
            body: vec![],
            i: Info::default()
        }]);
        assert!(ensure_parallelism(&def, &BTreeMap::new()).is_ok());
    }

    #[test]
    fn ensure_parallelism_labelled_for_loop() {
        let def = fun_with_body(vec![Stmt::For {
            var: id("x"),
            lo: int(0, None),
            hi: int(10, None),
            step: 1,
            body: vec![],
            labels: vec!["x".to_string()],
            i: Info::default()
        }]);
        let p = vec![
            ("x".to_string(), LoopPar::default())
        ].into_iter().collect::<BTreeMap<String, LoopPar>>();
        assert!(ensure_parallelism(&def, &p).is_ok());
    }

    #[test]
    fn ensure_parallelism_fails() {
        let def = fun_with_body(vec![]);
        assert!(ensure_parallelism(&def, &BTreeMap::new()).is_err());
    }
}
