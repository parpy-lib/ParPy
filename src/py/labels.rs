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
use crate::utils::info::*;

use pyo3::prelude::*;

fn add_labels_stmt(
    stmt: Stmt,
    mut l: Vec<String>
) -> PyResult<Stmt> {
    match stmt {
        Stmt::Definition {ty, id, expr, mut labels, i} => {
            labels.append(&mut l);
            Ok(Stmt::Definition {ty, id, expr, labels, i})
        },
        Stmt::Assign {dst, expr, mut labels, i} => {
            labels.append(&mut l);
            Ok(Stmt::Assign {dst, expr, labels, i})
        },
        Stmt::For {var, lo, hi, step, body, mut labels, i} => {
            labels.append(&mut l);
            Ok(Stmt::For {var, lo, hi, step, body, labels, i})
        },
        Stmt::If {..} | Stmt::While {..} | Stmt::Return {..} |
        Stmt::WithGpuContext {..} | Stmt::Scope {..} | Stmt::Call {..} |
        Stmt::Label {..} =>
            py_runtime_error!(
                stmt.get_info(),
                "Cannot associate label with non-parallelizable statement"
            )
    }
}

fn associate_labels_stmt(
    acc: PyResult<(Vec<Stmt>, Vec<String>)>,
    s: Stmt
) -> PyResult<(Vec<Stmt>, Vec<String>)> {
    let (mut stmts, mut labels) = acc?;
    if let Stmt::Label {label, ..} = s {
        labels.push(label);
        Ok((stmts, labels))
    } else {
        let s = if !labels.is_empty() {
            add_labels_stmt(s, labels)
        } else {
            Ok(s)
        }?;
        let s = match s {
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let body = associate_labels_stmts(body)?;
                Stmt::For {var, lo, hi, step, body, labels, i}
            },
            Stmt::While {cond, body, i} => {
                let body = associate_labels_stmts(body)?;
                Stmt::While {cond, body, i}
            },
            Stmt::If {cond, thn, els, i} => {
                let thn = associate_labels_stmts(thn)?;
                let els = associate_labels_stmts(els)?;
                Stmt::If {cond, thn, els, i}
            },
            Stmt::WithGpuContext {body, i} => {
                let body = associate_labels_stmts(body)?;
                Stmt::WithGpuContext {body, i}
            },
            Stmt::Scope {body, i} => {
                let body = associate_labels_stmts(body)?;
                Stmt::Scope {body, i}
            },
            Stmt::Definition {..} | Stmt::Assign {..} | Stmt::Return {..} |
            Stmt::Call {..} | Stmt::Label {..} => s
        };
        stmts.push(s);
        Ok((stmts, vec![]))
    }
}

fn associate_labels_stmts(stmts: Vec<Stmt>) -> PyResult<Vec<Stmt>> {
    let (stmts, labels) = stmts.into_iter()
        .fold(Ok((vec![], vec![])), associate_labels_stmt)?;
    if labels.is_empty() {
        Ok(stmts)
    } else {
        let i = stmts.last().map(|s| s.get_info()).unwrap_or(Info::default());
        py_runtime_error!(i, "Found labels not associated with any statement")
    }
}


pub fn associate_labels(fun: FunDef) -> PyResult<FunDef> {
    let body = associate_labels_stmts(fun.body)?;
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
        Expr::Int {v: v as i128, ty: Type::Unknown, i: i()}
    }

    #[test]
    fn assoc_for_loop() {
        let x = Name::sym_str("x");
        let body = vec![
            Stmt::Label {label: "x".to_string(), i: i()},
            Stmt::For {
                var: x.clone(), lo: int(1), hi: int(7), step: 2, body: vec![],
                labels: vec![], i: i()
            }
        ];
        let res = associate_labels_stmts(body).unwrap();
        assert_eq!(res, vec![Stmt::For {
            var: x, lo: int(1), hi: int(7), step: 2, body: vec![],
            labels: vec!["x".to_string()], i: i()
        }]);
    }

    #[test]
    fn assoc_nested_loops() {
        let x = Name::sym_str("x");
        let y = Name::sym_str("y");
        let inner_for = |l| Stmt::For {
            var: y.clone(), lo: int(1), hi: int(10), step: 1, body: vec![],
            labels: l, i: i()
        };
        let body = vec![
            Stmt::For {
                var: x.clone(), lo: int(1), hi: int(7), step: 1, body: vec![
                    Stmt::Label {label: "i".to_string(), i: i()},
                    inner_for(vec![])
                ],
                labels: vec![], i: i()
            }
        ];
        let res = associate_labels_stmts(body).unwrap();
        assert_eq!(res, vec![Stmt::For {
            var: x.clone(), lo: int(1), hi: int(7), step: 1,
            body: vec![inner_for(vec!["i".to_string()])],
            labels: vec![], i: i()
        }]);
    }
}
