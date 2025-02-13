/// Associates each label statement with the following non-label statement. That is, we construct a
/// tree-like structure where each label contains the statement it refers to (possibly including
/// further label statements due to nesting).
///
/// This transformation fails when a label has no subsequent statement to be associated with, or
/// when a label is associated with an unsupported statement (currently, anything other than a
/// for-loop).

use super::ast::*;
use crate::py_runtime_error;
use crate::utils::err::*;

use pyo3::prelude::*;

fn assert_contains_labels_stmt(acc: bool, stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Definition {..} => acc,
        Stmt::Assign {..} => acc,
        Stmt::For {body, ..} =>
            body.iter().fold(acc, assert_contains_labels_stmt),
        Stmt::If {thn, els, ..} => {
            let acc = thn.iter().fold(acc, assert_contains_labels_stmt);
            els.iter().fold(acc, assert_contains_labels_stmt)
        },
        Stmt::While {body, ..} =>
            body.iter().fold(acc, assert_contains_labels_stmt),
        Stmt::WithGpuContext {..} => true,
        Stmt::Label {..} => true,
    }
}

fn assert_contains_labels(fun: &FunDef) -> PyResult<()> {
    let res = fun.body.iter().fold(false, assert_contains_labels_stmt);
    if !res {
        let msg = concat!(
            "This function contains no code that runs or could be configured ",
            "to run on the GPU, due to missing labels on parallelizable ",
            "statements.\n",
        );
        py_runtime_error!(fun.i, "{}", msg)
    } else {
        Ok(())
    }
}

fn associate_labels_stmt(
    mut acc: Vec<Stmt>,
    stmt: Stmt
) -> PyResult<Vec<Stmt>> {
    let stmt = match stmt {
        Stmt::For {var, lo, hi, step, body, i} => {
            let body = associate_labels_stmts(vec![], body)?;
            Stmt::For {var, lo, hi, step, body, i}
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = associate_labels_stmts(vec![], thn)?;
            let els = associate_labels_stmts(vec![], els)?;
            Stmt::If {cond, thn, els, i}
        },
        Stmt::While {cond, body, i} => {
            let body = associate_labels_stmts(vec![], body)?;
            Stmt::While {cond, body, i}
        },
        Stmt::WithGpuContext {body, i} => {
            let body = associate_labels_stmts(vec![], body)?;
            Stmt::WithGpuContext {body, i}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Label {..} => {
            stmt
        }
    };
    match acc.pop() {
        Some(Stmt::Label {label, assoc: None, i}) => {
            match stmt {
                Stmt::Label {..} | Stmt::For {..} => {
                    acc.push(Stmt::Label {label, assoc: Some(Box::new(stmt)), i});
                    Ok(acc)
                },
                _ => py_runtime_error!(i, "Labels cannot be associated with non-parallelizable statements")
            }
        },
        Some(top_stmt) => {
            acc.push(top_stmt);
            acc.push(stmt);
            Ok(acc)
        },
        _ => {
            acc.push(stmt);
            Ok(acc)
        }
    }
}

fn associate_labels_stmts(
    acc: Vec<Stmt>,
    body: Vec<Stmt>
) -> PyResult<Vec<Stmt>> {
    body.into_iter().fold(Ok(acc), |acc, stmt| associate_labels_stmt(acc?, stmt))
}

pub fn associate_labels(fun: FunDef) -> PyResult<FunDef> {
    assert_contains_labels(&fun)?;
    let body = associate_labels_stmts(vec![], fun.body)?;
    Ok(FunDef {body, ..fun})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::info::Info;
    use crate::utils::name::Name;

    fn i() -> Info {
        Info::default()
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: Type::Unknown, i: i()}
    }

    #[test]
    fn assoc_for_loop() {
        let x = Name::sym_str("x");
        let body = vec![
            Stmt::Label {label: "x".to_string(), assoc: None, i: i()},
            Stmt::For {var: x.clone(), lo: int(1), hi: int(7), step: 2, body: vec![], i: i()}
        ];
        let res = associate_labels_stmts(vec![], body).unwrap();
        assert_eq!(res, vec![Stmt::Label {
            label: "x".to_string(), assoc: Some(Box::new(Stmt::For {
                var: x, lo: int(1), hi: int(7), step: 2, body: vec![], i: i()
            })),
            i: i()
        }]);
    }

    #[test]
    fn assoc_nested_loops() {
        let x = Name::sym_str("x");
        let y = Name::sym_str("y");
        let inner_for = Stmt::For {
            var: y.clone(), lo: int(1), hi: int(10), step: 1, body: vec![], i: i()
        };
        let body = vec![
            Stmt::For {
                var: x.clone(), lo: int(1), hi: int(7), step: 1, body: vec![
                    Stmt::Label {label: "i".to_string(), assoc: None, i: i()},
                    inner_for.clone()
                ],
                i: i()
            }
        ];
        let res = associate_labels_stmts(vec![], body).unwrap();
        assert_eq!(res, vec![Stmt::For {
            var: x.clone(), lo: int(1), hi: int(7), step: 1, body: vec![Stmt::Label {
                label: "i".to_string(), assoc: Some(Box::new(inner_for)), i: i()
            }],
            i: i()
        }]);
    }
}
