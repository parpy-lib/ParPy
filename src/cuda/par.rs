/// Identifies the structure of parallelism given an AST and a parallel specification belonging to
/// a particular function.

use crate::parir_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;

use std::collections::BTreeMap;

type ParResult = CompileResult<BTreeMap<Name, Vec<i64>>>;

struct Par {
    n: Vec<i64>,
    i: Info
}

impl PartialEq for Par {
    fn eq(&self, other: &Self) -> bool {
        self.n.eq(&other.n)
    }
}

// Attempts to unify the parallel structure of two statements. An empty structure represents
// sequential code. We can unify it with any kind of parallelization. On the other hand, two
// parallel structures have to be equivalent to merge them.
fn unify_par(acc: Par, p: Par) -> CompileResult<Par> {
    if acc.n.is_empty() {
        Ok(p)
    } else if p.n.is_empty() {
        Ok(acc)
    } else if acc.n.eq(&p.n) {
        Ok(acc)
    } else {
        let msg = concat!(
            "The parallel structure of this statement is inconsistent relative to ",
            "previous statements on the same level of nesting"
        );
        parir_compile_error!(p.i, "{}", msg)
    }
}

fn find_parallel_structure_stmt_par(stmt: &Stmt) -> CompileResult<Par> {
    match stmt {
        Stmt::Definition {i, ..} | Stmt::Assign {i, ..} =>
            Ok(Par {n: vec![], i: i.clone()}),
        Stmt::If {thn, els, i, ..} => {
            let thn = find_parallel_structure_stmts_par(thn)?;
            let els = find_parallel_structure_stmts_par(els)?;
            // For if-conditions, we require that both branches have exactly the same parallel
            // structure. That is, we do not allow parallelism only in one branch - this is due to
            // the performance implications this might have.
            if thn.n.eq(&els.n) {
                Ok(Par {n: thn.n, i: i.clone()})
            } else {
                let msg = concat!(
                    "Branches cannot have inconsistent parallel structure: {thn.n:?} != {els.n:?}",
                    "\n\nThis may cause significantly imbalanced workloads."
                );
                parir_compile_error!(i, "{}", msg)
            }
        },
        Stmt::For {body, par, i, ..} => {
            let mut inner_par = find_parallel_structure_stmts_par(body)?;
            match par {
                Some(LoopProperty::Threads {n}) => inner_par.n.insert(0, *n),
                None => ()
            };
            Ok(inner_par)
        },
    }
}

fn find_parallel_structure_stmts_par(stmts: &Vec<Stmt>) -> CompileResult<Par> {
    stmts.iter()
        .map(find_parallel_structure_stmt_par)
        .fold(Ok(Par {n: vec![], i: Info::default()}), |acc, stmt_par| {
            unify_par(acc?, stmt_par?)
        })
}

fn find_parallel_structure_stmt_seq(
    mut acc: BTreeMap<Name, Vec<i64>>,
    stmt: &Stmt
) -> ParResult {
    match stmt {
        Stmt::Definition {..} | Stmt::Assign {..} => Ok(acc),
        Stmt::If {thn, els, ..} => {
            let acc = find_parallel_structure_stmts_seq(acc, thn)?;
            find_parallel_structure_stmts_seq(acc, els)
        },
        Stmt::For {var, body, par, i, ..} => {
            match par {
                Some(LoopProperty::Threads {n}) => {
                    let mut p = find_parallel_structure_stmts_par(body)?;
                    p.n.insert(0, *n);
                    acc.insert(var.clone(), p.n);
                    Ok(acc)
                },
                None => find_parallel_structure_stmts_seq(acc, body)
            }
        }
    }
}

fn find_parallel_structure_stmts_seq(
    acc: BTreeMap<Name, Vec<i64>>,
    stmts: &Vec<Stmt>
) -> ParResult {
    stmts.iter()
        .fold(Ok(acc), |acc, stmt| find_parallel_structure_stmt_seq(acc?, stmt))
}

fn find_parallel_structure_fun_def(fun_def: &FunDef) -> ParResult {
    let map = BTreeMap::new();
    find_parallel_structure_stmts_seq(map, &fun_def.body)
}

pub fn find_parallel_structure(ast: &Ast) -> ParResult {
    find_parallel_structure_fun_def(&ast.fun)
}

#[cfg(test)]
mod test {
    use super::*;

    fn id(s: &str) -> Name {
        Name::new(s.to_string()).with_new_sym()
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]}, i: Info::default()}
    }

    fn for_loop(var: Name, nthreads: i64, body: Vec<Stmt>) -> Stmt {
        let par = if nthreads == 1 {
            None
        } else {
            Some(LoopProperty::Threads {n: nthreads})
        };
        Stmt::For {var, lo: int(0), hi: int(10), body, par, i: Info::default()}
    }

    fn if_cond(thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        Stmt::If {cond: int(0), thn, els, i: Info::default()}
    }

    fn fun_def(body: Vec<Stmt>) -> FunDef {
        FunDef {id: id("x"), params: vec![], body, i: Info::default()}
    }

    fn find_structure(def: FunDef) -> BTreeMap<Name, Vec<i64>> {
        find_parallel_structure_fun_def(&def).unwrap()
    }

    fn to_map(v: Vec<(Name, Vec<i64>)>) -> BTreeMap<Name, Vec<i64>> {
        v.into_iter().collect::<BTreeMap<Name, Vec<i64>>>()
    }

    #[test]
    fn seq_loops() {
        let def = fun_def(vec![for_loop(id("x"), 1, vec![for_loop(id("y"), 1, vec![])])]);
        assert_eq!(find_structure(def), BTreeMap::new());
    }

    #[test]
    fn single_par_loop() {
        let x = id("x");
        let def = fun_def(vec![for_loop(x.clone(), 10, vec![])]);
        let expected = to_map(vec![(x, vec![10])]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn consistent_par_if() {
        let x = id("x");
        let def = fun_def(vec![for_loop(x.clone(), 2, vec![if_cond(
            vec![for_loop(id("y"), 3, vec![])],
            vec![for_loop(id("z"), 3, vec![for_loop(id("w"), 1, vec![])])],
        )])]);
        let expected = to_map(vec![
            (x, vec![2, 3])
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn inconsistent_par_if() {
        let def = fun_def(vec![for_loop(id("x"), 2, vec![if_cond(
            vec![for_loop(id("y"), 3, vec![])],
            vec![for_loop(id("z"), 4, vec![])]
        )])]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn equal_par_but_distinct_structure() {
        let def = fun_def(vec![for_loop(id("x"), 5, vec![
            if_cond(
                vec![for_loop(id("y"), 4, vec![for_loop(id("z"), 7, vec![])])],
                vec![for_loop(id("z"), 7, vec![for_loop(id("y"), 4, vec![])])]
            )
        ])]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn nested_parallelism() {
        let x = id("x");
        let def = fun_def(vec![
            for_loop(x.clone(), 10, vec![for_loop(id("y"), 15, vec![for_loop(id("z"), 1, vec![])])])
        ]);
        let expected = to_map(vec![
            (x, vec![10, 15])
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn nested_consistent_if() {
        let x = id("x");
        let def = fun_def(vec![
            for_loop(x.clone(), 10, vec![if_cond(
                vec![for_loop(id("y"), 15, vec![])],
                vec![for_loop(id("y"), 1, vec![for_loop(id("z"), 15, vec![])])]
            )])
        ]);
        let expected = to_map(vec![
            (x, vec![10, 15])
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn nested_inconsistent_if() {
        let def = fun_def(vec![
            for_loop(id("x"), 10, vec![if_cond(
                vec![for_loop(id("y"), 10, vec![])],
                vec![for_loop(id("z"), 15, vec![])]
            )])
        ]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn subsequent_par() {
        let x1 = id("x");
        let x2 = id("x");
        let def = fun_def(vec![
            for_loop(x1.clone(), 5, vec![]),
            for_loop(x2.clone(), 7, vec![])
        ]);
        let expected = to_map(vec![
            (x1, vec![5]),
            (x2, vec![7])
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn subsequent_distinct_struct_par() {
        let x1 = id("x");
        let x2 = id("x");
        let def = fun_def(vec![
            for_loop(x1.clone(), 5, vec![
                if_cond(
                    vec![for_loop(id("y"), 6, vec![])],
                    vec![for_loop(id("y"), 1, vec![for_loop(id("z"), 6, vec![])])]
                )
            ]),
            for_loop(x2.clone(), 7, vec![for_loop(id("w"), 12, vec![])])
        ]);
        let expected = to_map(vec![
            (x1, vec![5, 6]),
            (x2, vec![7, 12])
        ]);
        assert_eq!(find_structure(def), expected);
    }
}
