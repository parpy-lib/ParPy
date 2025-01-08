use crate::parir_runtime_error;
use crate::err::*;
use crate::info::Info;
use crate::par::*;
use super::ast::*;

use std::collections::{HashSet, HashMap};

#[derive(Clone, Debug, PartialEq)]
enum ParRepr {
    Seq(String, Vec<ParRepr>),
    Par(String, u64, Vec<ParRepr>)
}

impl ParRepr {
    fn is_leaf(&self) -> bool {
        match self {
            ParRepr::Seq(_, v) => v.is_empty(),
            ParRepr::Par(_, _, v) => v.is_empty(),
        }
    }

    fn children<'a>(&'a self) -> &'a Vec<ParRepr> {
        match self {
            ParRepr::Seq(_, v) => v,
            ParRepr::Par(_, _, v) => v,
        }
    }
}

fn paths_to_root<'a>(
    acc: (HashSet<Vec<(String, u64)>>, Vec<&'a ParRepr>),
    curr: &'a ParRepr,
) -> (HashSet<Vec<(String, u64)>>, Vec<&'a ParRepr>) {
    if curr.is_leaf() {
        let (mut paths, mut path) = acc;
        path.push(curr);
        let mut par_path = vec![];
        for repr in path.iter() {
            if let ParRepr::Par(id, n, _) = repr {
                par_path.push((id.clone(), *n));
            }
        }
        if !par_path.is_empty() {
            paths.insert(par_path);
        }
        path.pop();
        (paths, path)
    } else {
        let (paths, mut path) = acc;
        path.push(curr);
        let (paths, mut path) = curr.children()
            .iter()
            .fold((paths, path), paths_to_root);
        path.pop();
        (paths, path)
    }
}

/// Finds all parallel paths starting from a leaf node up to the root. The result includes the
/// name and number of threads of all parallel for-loops along each path.
fn parallel_paths(reprs: &Vec<ParRepr>) -> Vec<Vec<(String, u64)>> {
    reprs.iter()
        .map(|repr| {
             let (paths, _) = paths_to_root((HashSet::new(), vec![]), repr);
             paths.into_iter().collect::<Vec<Vec<(String, u64)>>>()
        })
        .flatten()
        .collect()
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
                Some(ParSpec {kind: ParKind::CpuThreads(n)}) => Ok(ParRepr::Par(id.clone(), *n, body_repr)),
                Some(ParSpec {kind: ParKind::GpuThreads(n)}) => Ok(ParRepr::Par(id.clone(), *n, body_repr)),
                Some(ParSpec {kind: ParKind::GpuBlocks(_)}) => panic!("GpuBlocks not supported"),
                None => Ok(ParRepr::Seq(id.clone(), body_repr))
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

#[derive(Clone, Debug, PartialEq)]
enum Device {
    Unspec, Cpu, Gpu
}

fn find_parallelization_device(
    par: &HashMap<String, ParSpec>
) -> CompileResult<Device> {
    let i = Info::default();
    par.values()
        .fold(Ok(Device::Unspec), |acc, ParSpec {kind}| match kind {
            ParKind::CpuThreads(_) => match acc? {
                Device::Unspec | Device::Cpu => Ok(Device::Cpu),
                Device::Gpu => parir_runtime_error!(i, "Cannot mix CPU and GPU parallelization")
            },
            ParKind::GpuThreads(_) => match acc? {
                Device::Unspec | Device::Gpu => Ok(Device::Gpu),
                Device::Cpu => parir_runtime_error!(i, "Cannot mix CPU and GPU parallelization")
            },
            ParKind::GpuBlocks(_) => panic!("GpuBlocks not supported")
        })
}

fn gpu_alloc(
    mut par_path: Vec<(String, u64)>,
    dims: (u64, u64)
) -> (u64, u64) {
    let v = par_path.pop();
    if let Some((_, n)) = v {
        let dims = if dims.0 > 1 || dims.1 * n > 1024 {
            (dims.0 * n, dims.1)
        } else {
            (dims.0, dims.1 * n)
        };
        gpu_alloc(par_path, dims)
    } else {
        dims
    }
}

fn determine_gpu_grid_size(par_repr: &Vec<ParRepr>) -> (u64, u64) {
    let mut dims = (1, 1);
    for par_path in parallel_paths(par_repr) {
        let path_dims = gpu_alloc(par_path, (1, 1));
        dims.0 = u64::max(dims.0, path_dims.0);
        dims.1 = u64::max(dims.1, path_dims.1);
    }
    dims
}

pub fn parallelize_loops(
    body: Vec<Stmt>,
    par: HashMap<String, ParSpec>
) -> CompileResult<(Vec<Stmt>, u64, u64)> {
    let _ = validate_parallelization(&body, &par)?;
    let par_repr = construct_parallel_representation_stmts(&body, &par)?;

    if let Device::Gpu = find_parallelization_device(&par)? {
        let (nblocks, nthreads) = determine_gpu_grid_size(&par_repr);

        // Remains to be implemented:
        // - Determine how to map individual for-loops to the threads and blocks of the function.
        // - Update the body by annotating the for-loops based on the second step.

        Ok((body, nblocks, nthreads))
    } else {
        unimplemented!()
    }
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

    fn par_device(par: HashMap<String, ParSpec>) -> CompileResult<Device> {
        find_parallelization_device(&par)
    }

    fn par_paths(par: HashMap<String, ParSpec>) -> CompileResult<Vec<Vec<(String, u64)>>> {
        let repr = par_repr(par)?;
        Ok(parallel_paths(&repr))
    }

    fn grid_size(par: HashMap<String, ParSpec>) -> CompileResult<(u64, u64)> {
        let repr = par_repr(par)?;
        Ok(determine_gpu_grid_size(&repr))
    }

    fn seq_loops() -> HashMap<String, ParSpec> {
        HashMap::new()
    }

    fn assert_sorted_eq<T: std::fmt::Debug + std::cmp::Ord + std::cmp::PartialEq>(
        mut lhs: CompileResult<Vec<T>>,
        mut rhs: CompileResult<Vec<T>>
    ) {
        let sort_vec = |mut v: Vec<T>| {
            v.sort();
            v
        };
        assert_eq!(lhs.map(sort_vec), rhs.map(sort_vec));
    }

    #[test]
    fn test_valid_seq() {
        assert!(validate(seq_loops()))
    }

    #[test]
    fn test_par_repr_seq_loops() {
        assert_eq!(par_repr(seq_loops()), Ok(vec![
            ParRepr::Seq(var!("a"), vec![
                ParRepr::Seq(var!("b"), vec![
                    ParRepr::Seq(var!("c"), vec![]),
                    ParRepr::Seq(var!("d"), vec![]),
                ]),
                ParRepr::Seq(var!("e"), vec![])
            ]),
            ParRepr::Seq(var!("f"), vec![])
        ]));
    }

    #[test]
    fn test_par_device_seq_loops() {
        assert_eq!(par_device(seq_loops()), Ok(Device::Unspec));
    }

    #[test]
    fn test_par_paths_seq_loops() {
        assert_sorted_eq(par_paths(seq_loops()), Ok(vec![]));
    }

    #[test]
    fn test_grid_size_seq_loops() {
        assert_eq!(grid_size(seq_loops()), Ok((1, 1)));
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
            ParRepr::Par(var!("a"), 128, vec![
                ParRepr::Seq(var!("b"), vec![
                    ParRepr::Seq(var!("c"), vec![]),
                    ParRepr::Seq(var!("d"), vec![]),
                ]),
                ParRepr::Seq(var!("e"), vec![])
            ]),
            ParRepr::Par(var!("f"), 8, vec![])
        ]));
    }

    #[test]
    fn test_par_device_par_outer_only() {
        assert_eq!(par_device(par_outer_only()), Ok(Device::Gpu));
    }

    #[test]
    fn test_par_paths_par_outer_only() {
        assert_sorted_eq(par_paths(par_outer_only()), Ok(vec![
            vec![(var!("a"), 128)],
            vec![(var!("f"), 8)]
        ]));
    }

    #[test]
    fn test_grid_size_par_outer_only() {
        assert_eq!(grid_size(par_outer_only()), Ok((1, 128)));
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
            ParRepr::Seq(var!("a"), vec![
                ParRepr::Par(var!("b"), 32, vec![
                    ParRepr::Par(var!("c"), 64, vec![]),
                    ParRepr::Seq(var!("d"), vec![]),
                ]),
                ParRepr::Par(var!("e"), 128, vec![])
            ]),
            ParRepr::Seq(var!("f"), vec![])
        ]))
    }

    #[test]
    fn test_par_device_par_inner_only() {
        assert_eq!(par_device(par_inner_only()), Ok(Device::Gpu));
    }

    #[test]
    fn test_par_paths_par_inner_only() {
        assert_sorted_eq(par_paths(par_inner_only()), Ok(vec![
            vec![(var!("b"), 32), (var!("c"), 64)],
            vec![(var!("b"), 32)],
            vec![(var!("e"), 128)]
        ]));
    }

    #[test]
    fn test_grid_size_par_inner_only() {
        assert_eq!(grid_size(par_inner_only()), Ok((32, 128)));
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
            ParRepr::Par(var!("a"), 128, vec![
                ParRepr::Seq(var!("b"), vec![
                    ParRepr::Par(var!("c"), 256, vec![]),
                    ParRepr::Seq(var!("d"), vec![]),
                ]),
                ParRepr::Seq(var!("e"), vec![])
            ]),
            ParRepr::Par(var!("f"), 64, vec![])
        ]));
    }

    #[test]
    fn test_par_device_par_nested() {
        assert_eq!(par_device(par_nested()), Ok(Device::Gpu));
    }

    #[test]
    fn test_par_paths_par_nested() {
        assert_sorted_eq(par_paths(par_nested()), Ok(vec![
            vec![(var!("a"), 128), (var!("c"), 256)],
            vec![(var!("a"), 128)],
            vec![(var!("f"), 64)]
        ]));
    }

    #[test]
    fn test_grid_size_par_nested() {
        assert_eq!(grid_size(par_nested()), Ok((128, 256)));
    }

    fn mixed_cpu_gpu() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("a"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("f"), ParSpec {kind: ParKind::CpuThreads(8)})
        ])
    }

    #[test]
    fn test_invalid_mixed_cpu_gpu() {
        assert!(!validate(mixed_cpu_gpu()))
    }

    #[test]
    fn test_par_device_mixed_cpu_gpu() {
        assert!(par_device(mixed_cpu_gpu()).is_err())
    }
}
