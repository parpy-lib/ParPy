use crate::parir_runtime_error;
use crate::err::*;
use crate::info::Info;
use crate::par::*;
use super::ast::*;

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum ParRepr {
    Seq(Vec<ParRepr>),
    Par(u64, Vec<ParRepr>)
}

fn ensure_consistent_parallelization(
    par: &HashMap<String, ParSpec>
) -> CompileResult<()> {
    if !par.is_empty() {
        let mut cpu_par = false;
        let mut gpu_par = false;
        for (_, ParSpec {kind}) in par.iter() {
            match kind {
                ParKind::CpuThreads(_) => cpu_par = true,
                ParKind::GpuThreads(_) => gpu_par = true,
                ParKind::GpuBlocks(_) => panic!("GpuBlocks are disallowed")
            }
        }
        // The parallelization is inconsistent if we use both CPU and GPU parallelism in the same
        // instantiation of a function.
        if cpu_par && gpu_par {
            let i = Info::default();
            parir_runtime_error!(i, "Using CPU and GPU parallelism in the same function is not permitted")
        } else {
            Ok(())
        }
    } else {
        Ok(())
    }
}

fn remove_used_variables(
    body: &Vec<Stmt>,
    mut par: HashMap<String, ParSpec>
) -> HashMap<String, ParSpec> {
    for stmt in body {
        if let Stmt::For {id, body, ..} = stmt {
            par = remove_used_variables(body, par);
            par.remove(id);
        }
    }
    par
}

fn ensure_all_variables_are_used(
    body: &Vec<Stmt>,
    par: &HashMap<String, ParSpec>
) -> CompileResult<()> {
    let par = remove_used_variables(body, par.clone());
    if !par.is_empty() {
        let i = Info::default();
        let msg = vec![
            "Parallel pattern refers to variables ",
            &par.clone().into_keys().collect::<Vec<String>>().join(" "),
            " which are not present among for-loops of the provided program"
        ].join("\n");
        parir_runtime_error!(i, "{msg}")
    } else {
        Ok(())
    }
}

fn validate_parallelization(
    body: &Vec<Stmt>,
    par: &HashMap<String, ParSpec>
) -> CompileResult<()> {
    ensure_consistent_parallelization(par)?;
    ensure_all_variables_are_used(body, par)
}

fn construct_parallel_representation_stmt(
    mut acc: Vec<ParRepr>,
    s: &Stmt,
    par: &HashMap<String, ParSpec>
) -> CompileResult<Vec<ParRepr>> {
    match s {
        Stmt::For {id, body, i, ..} => {
            let body_repr = construct_parallel_representation_stmts(body, par)?;
            let repr = match par.get(id) {
                Some(ParSpec {kind: ParKind::CpuThreads(n)}) => Ok(ParRepr::Par(*n, body_repr)),
                Some(ParSpec {kind: ParKind::GpuThreads(n)}) => Ok(ParRepr::Par(*n, body_repr)),
                Some(ParSpec {kind: ParKind::GpuBlocks(_)}) => panic!("GpuBlocks not supported"),
                None => Ok(ParRepr::Seq(body_repr))
            }?;
            acc.push(repr);
            Ok(acc)
        },
        _ => Ok(acc)
    }
}

fn construct_parallel_representation_stmts(
    stmts: &Vec<Stmt>,
    par: &HashMap<String, ParSpec>
) -> CompileResult<Vec<ParRepr>> {
    stmts.iter()
        .fold(Ok(vec![]), |acc, s| construct_parallel_representation_stmt(acc?, s, par))
}

pub fn parallelize_loops(
    body: Vec<Stmt>,
    par: HashMap<String, ParSpec>
) -> CompileResult<(Vec<Stmt>, i64, i64)> {
    let _ = validate_parallelization(&body, &par)?;
    let par_repr = construct_parallel_representation_stmts(&body, &par)?;

    // Remains to be implemented:
    // 1. Compute the number of threads and blocks to use.
    // 2. Determine how to map individual for-loops to the threads and blocks of the function.
    // 3. Update the body by annotating the for-loops based on the second step.

    Ok((body, 1, 1))
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! var {
        ($s:tt) => {
            $s.to_string()
        }
    }

    fn for_loop0(id: String, hi: i64, body: Vec<Stmt>) -> Stmt {
        Stmt::For {
            id,
            lo: Expr::Int {v: 0, ty: Type::Int(IntSize::I64), i: Info::default()},
            hi: Expr::Int {v: hi, ty: Type::Int(IntSize::I64), i: Info::default()},
            body,
            properties: LoopProperties::default(),
            i: Info::default()
        }
    }

    fn example_body() -> Vec<Stmt> {
        let n1 = 1024;
        let n2 = 512;
        let n3 = 768;
        let n4 = 128;
        vec![
            for_loop0(var!("a"), n1, vec![
                for_loop0(var!("b"), n2, vec![
                    for_loop0(var!("c"), n3, vec![]),
                    for_loop0(var!("d"), n4, vec![])
                ]),
                for_loop0(var!("e"), n3, vec![])
            ]),
            for_loop0(var!("f"), n4, vec![])
        ]
    }

    fn validate(par: HashMap<String, ParSpec>) -> bool {
        let body = example_body();
        validate_parallelization(&body, &par).is_ok()
    }

    fn par_repr(par: HashMap<String, ParSpec>) -> CompileResult<Vec<ParRepr>> {
        let body = example_body();
        construct_parallel_representation_stmts(&body, &par)
    }

    fn seq_loops() -> HashMap<String, ParSpec> {
        HashMap::new()
    }

    #[test]
    fn test_valid_seq() {
        assert!(validate(seq_loops()))
    }

    #[test]
    fn test_par_repr_seq_loops() {
        assert_eq!(par_repr(seq_loops()), Ok(vec![
            ParRepr::Seq(vec![
                ParRepr::Seq(vec![
                    ParRepr::Seq(vec![]),
                    ParRepr::Seq(vec![]),
                ]),
                ParRepr::Seq(vec![])
            ]),
            ParRepr::Seq(vec![])
        ]));
    }

    fn par_outer_only() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("a"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("f"), ParSpec {kind: ParKind::GpuThreads(8)})
        ])
    }

    #[test]
    fn test_valid_par_outer() {
        assert!(validate(par_outer_only()))
    }

    #[test]
    fn test_par_repr_par_outer_only() {
        assert_eq!(par_repr(par_outer_only()), Ok(vec![
            ParRepr::Par(128, vec![
                ParRepr::Seq(vec![
                    ParRepr::Seq(vec![]),
                    ParRepr::Seq(vec![]),
                ]),
                ParRepr::Seq(vec![])
            ]),
            ParRepr::Par(8, vec![])
        ]));
    }

    fn par_inner_only() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("b"), ParSpec {kind: ParKind::GpuThreads(32)}),
            (var!("c"), ParSpec {kind: ParKind::GpuThreads(64)}),
            (var!("e"), ParSpec {kind: ParKind::GpuThreads(128)}),
        ])
    }

    #[test]
    fn test_valid_par_inner_only() {
        assert!(validate(par_inner_only()))
    }

    #[test]
    fn test_par_repr_par_inner_only() {
        assert_eq!(par_repr(par_inner_only()), Ok(vec![
            ParRepr::Seq(vec![
                ParRepr::Par(32, vec![
                    ParRepr::Par(64, vec![]),
                    ParRepr::Seq(vec![]),
                ]),
                ParRepr::Par(128, vec![])
            ]),
            ParRepr::Seq(vec![])
        ]))
    }

    fn par_nested() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("a"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("c"), ParSpec {kind: ParKind::GpuThreads(256)}),
            (var!("f"), ParSpec {kind: ParKind::GpuThreads(64)}),
        ])
    }

    #[test]
    fn test_valid_par_nested() {
        assert!(validate(par_nested()))
    }

    #[test]
    fn test_par_repr_par_nested() {
        let par = HashMap::from([
            (var!("a"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("c"), ParSpec {kind: ParKind::GpuThreads(256)}),
            (var!("f"), ParSpec {kind: ParKind::GpuThreads(64)}),
        ]);
        assert_eq!(par_repr(par), Ok(vec![
            ParRepr::Par(128, vec![
                ParRepr::Seq(vec![
                    ParRepr::Par(256, vec![]),
                    ParRepr::Seq(vec![]),
                ]),
                ParRepr::Seq(vec![])
            ]),
            ParRepr::Par(64, vec![])
        ]));
    }

    #[test]
    fn test_invalid_mixed_cpu_gpu() {
        let par = HashMap::from([
            (var!("a"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("f"), ParSpec {kind: ParKind::CpuThreads(8)})
        ]);
        assert!(!validate(par))
    }
}
