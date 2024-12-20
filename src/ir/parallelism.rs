use super::ast::*;

use crate::parir_runtime_error;
use crate::err::*;
use crate::info::*;
use crate::par::*;

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
struct ParEntry {
    loop_var: String,
    nthreads: u64
}

type ParMap = HashMap<String, Vec<ParEntry>>;

fn try_unify_parallelism(
    lhs: Vec<ParEntry>,
    rhs: Vec<ParEntry>,
    i: &Info
) -> CompileResult<Vec<ParEntry>> {
    if lhs.is_empty() { Ok(rhs) }
    else if rhs.is_empty() { Ok(lhs) }
    else {
        let f = |&ParEntry {nthreads, ..}| nthreads;
        if lhs.iter().map(f).eq(rhs.iter().map(f)) {
            Ok(lhs)
        } else {
            parir_runtime_error!(i, "Found incompatible parallelism in for-loop: {lhs:?} != {rhs:?}")
        }
    }
}

fn find_parallelism_stmt_device(
    stmt: &Stmt,
    par_spec: &HashMap<String, ParSpec>,
    i: &Info,
) -> CompileResult<Vec<ParEntry>> {
    match stmt {
        Stmt::Decl {..} | Stmt::AssignVar {..} | Stmt::AssignArray {..} |
        Stmt::DeviceCall {..} => Ok(vec![]),
        Stmt::For {var, body, i, ..} => {
            let mut par = find_parallelism_stmts_device(vec![], body, par_spec, i)?;
            let outer_par = match par_spec.get(var) {
                Some(ParSpec {kind: ParKind::GpuThreads(n)}) => Ok(Some(n)),
                Some(ParSpec {kind: ParKind::CpuThreads(_)}) => {
                    parir_runtime_error!(i, "Mixed CPU/GPU parallelization is not supported")
                },
                _ => Ok(None)
            }?;
            if let Some(nthreads) = outer_par {
                par.push(ParEntry {loop_var: var.clone(), nthreads: *nthreads});
            };
            Ok(par)
        }
    }
}

fn find_parallelism_stmts_device(
    mut acc: Vec<ParEntry>,
    stmts: &Vec<Stmt>,
    par_spec: &HashMap<String, ParSpec>,
    i: &Info
) -> CompileResult<Vec<ParEntry>> {
    let mut inner_par = vec![];
    for stmt in stmts {
        let par = find_parallelism_stmt_device(stmt, par_spec, i)?;
        inner_par = try_unify_parallelism(inner_par, par, i)?;
    }
    inner_par.append(&mut acc);
    Ok(inner_par)
}

fn find_parallelism_stmt_host(
    mut par_map: ParMap,
    stmt: &Stmt,
    par_spec: &HashMap<String, ParSpec>,
    i: &Info
) -> CompileResult<ParMap> {
    match stmt {
        Stmt::For {var, body, i, ..} => {
            match par_spec.get(var) {
                Some(ParSpec {kind: ParKind::GpuThreads(nthreads)}) => {
                    let init = if *nthreads > 1 {
                        vec![ParEntry {loop_var: var.clone(), nthreads: *nthreads}]
                    } else {
                        vec![]
                    };
                    let par_vec = find_parallelism_stmts_device(init, body, par_spec, i)?;
                    par_map.insert(var.clone(), par_vec);
                    Ok(par_map)
                },
                Some(ParSpec {kind: ParKind::CpuThreads(_)}) => {
                    parir_runtime_error!(i, "Mixed CPU/GPU parallelization is not supported")
                },
                _ => find_parallelism_stmts_host(par_map, body, par_spec, i),
            }
        },
        _ => Ok(par_map)
    }
}

fn find_parallelism_stmts_host(
    par_map: ParMap,
    stmts: &Vec<Stmt>,
    par_spec: &HashMap<String, ParSpec>,
    i: &Info
) -> CompileResult<ParMap> {
    stmts.into_iter()
        .fold(Ok(par_map), |acc, stmt| {
            find_parallelism_stmt_host(acc?, stmt, par_spec, i)
        })
}

#[derive(Clone, Debug, PartialEq)]
enum GridType {
    Block,
    Thread,
    Both
}

#[derive(Clone, Debug, PartialEq)]
struct GridEntry {
    ty: GridType,
    dim: Dim,
    divisor: u64
}

#[derive(Clone, Debug, PartialEq)]
struct KernelGridState {
    blocks: Dim3,
    threads: Dim3,
    mapping: Vec<GridEntry>
}

impl Default for KernelGridState {
    fn default() -> Self {
        KernelGridState {
            blocks: Dim3::default(),
            threads: Dim3::default(),
            mapping: Vec::new()
        }
    }
}

impl KernelGridState {
    fn add_block_dim(mut self, nthreads: u64) -> Self {
        let ty = GridType::Block;
        if self.blocks.x() == 1 {
            self.blocks = self.blocks.with_x(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::X, divisor: 1});
            self
        } else if self.blocks.y() == 1 {
            self.blocks = self.blocks.with_y(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::Y, divisor: 1});
            self
        } else if self.blocks.z() == 1 {
            self.blocks = self.blocks.with_z(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::Z, divisor: 1});
            self
        } else {
            let divisor = self.blocks.x();
            self.blocks = self.blocks.with_x(divisor * nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::X, divisor});
            self
        }
    }

    fn add_inner_dim(mut self, nthreads: u64) -> Self {
        // TODO: This value should be configurable somehow...
        const THREADS_PER_BLOCK : u64 = 512;
        let nblocks = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        self.threads = self.threads.with_x(THREADS_PER_BLOCK);
        self.blocks = self.blocks.with_x(nblocks);
        self.mapping.push(GridEntry {ty: GridType::Both, dim: Dim::X, divisor: 1});
        self
    }

    fn try_add_thread_dim(mut self, nthreads: u64) -> Self {
        let ty = GridType::Thread;
        if self.threads.x() == 1 {
            self.threads = self.threads.with_x(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::X, divisor: 1});
            self
        } else if self.threads.y() == 1 {
            self.threads = self.threads.with_y(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::Y, divisor: 1});
            self
        } else if self.threads.z() == 1 {
            self.threads = self.threads.with_z(nthreads);
            self.mapping.push(GridEntry {ty, dim: Dim::Z, divisor: 1});
            self
        } else {
            self.add_block_dim(nthreads)
        }
    }

    pub fn add_dim(self, nthreads: u64) -> Self {
        let curr_nblocks = self.blocks.count();
        let curr_nthreads = self.threads.count();

        // Once we have mapped a for-loop to blocks, we do not want an outer for-loop to be mapped
        // to threads, as this could lead to poor memory access patterns.
        if curr_nblocks > 1 {
            self.add_block_dim(nthreads)
        } else if curr_nthreads == 1 && nthreads > 1024 {
            self.add_inner_dim(nthreads)
        } else if curr_nthreads * nthreads <= 1024 {
            self.try_add_thread_dim(nthreads)
        } else {
            self.add_block_dim(nthreads)
        }
    }
}

fn map_parallelism_to_kernel_dimensions(
    par: Vec<ParEntry>
) -> KernelGridState {
    par.into_iter()
        .fold(KernelGridState::default(), |acc, ParEntry {nthreads, ..}| {
            acc.add_dim(nthreads)
        })
}

fn decide_kernel_dimensions(
    mut par_map : ParMap,
) -> HashMap<String, KernelGridState> {
    par_map.drain()
        .map(|(entry_var_id, v)| {
            let kernel_state = map_parallelism_to_kernel_dimensions(v);
            (entry_var_id, kernel_state)
        })
        .collect()
}

pub fn map_parallelism_function(
    body: Vec<Stmt>,
    par_spec: &HashMap<String, ParSpec>,
    i: &Info
) -> CompileResult<(Vec<Stmt>, HashMap<String, (Dim3, Dim3)>)> {
    let gpu_target = par_spec.iter().any(|(_, &ref v)| {
        let ParSpec { kind } = v;
        match kind {
            ParKind::CpuThreads(_) => false,
            ParKind::GpuThreads(_) => true,
            ParKind::GpuBlocks(_) => true
        }
    });
    if gpu_target {
        // 1. Identify the parallelism in the function body given the parallel specification
        //    associated with the function.
        let par_map = find_parallelism_stmts_host(HashMap::new(), &body, par_spec, i)?;

        // 2. Determine how to map the parallel for-loops within the function to GPU grids that can
        //    be launched.
        let kernel_grids = decide_kernel_dimensions(par_map);

        // 3. Use the resulting kernel grids to annotate the for-loops with metadata indicating how
        //    they are to be parallelized. These annotations are used to guide the GPU code
        //    generation.
        parir_runtime_error!(i, "TODO: map_parallelism_function")
    } else {
        // When the user does not use GPU parallelism at all, we report an error for now...
        parir_runtime_error!(i, "Parallelism mapping only supported for a GPU target")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::parallelism::CompileError;

    // Constants and definitions below are used for reusability across different kinds of unit
    // tests on the same underlying example.
    macro_rules! var {
        ($s:tt) => {
            $s.to_string()
        }
    }

    fn for_loop<'a>(var: String, lo: i64, hi: i64, body: Vec<Stmt>) -> Stmt {
        Stmt::For {
            var,
            lo: Expr::Int {v: lo, ty: Type::Int(IntSize::I64), i: Info::default()},
            hi: Expr::Int {v: hi, ty: Type::Int(IntSize::I64), i: Info::default()},
            body,
            properties: LoopProperties::default(),
            i: Info::default()
        }
    }

    fn test_example_loops() -> Stmt {
        let n1 = 1024;
        let n2 = 512;
        let n3 = 2048;
        for_loop(var!("i"), 0, n1, vec![
            for_loop(var!("j"), 0, n2, vec![
                for_loop(var!("k"), 0, n3, vec![]),
                for_loop(var!("l"), 0, n2, vec![])])])
    }

    fn map_parallelism(
        p: HashMap<String, ParSpec>
    ) -> CompileResult<ParMap> {
        let body = vec![test_example_loops()];
        let info = Info::default();
        find_parallelism_stmts_host(HashMap::new(), &body, &p, &info)
    }

    fn kernel_dims(
        p: HashMap<String, ParSpec>
    ) -> CompileResult<HashMap<String, KernelGridState>> {
        let par = map_parallelism(p)?;
        Ok(decide_kernel_dimensions(par))
    }

    fn par_seq() -> HashMap<String, ParSpec> {
        HashMap::new()
    }

    #[test]
    fn test_parallel_map_seq() {
        assert_eq!(map_parallelism(par_seq()), Ok(HashMap::new()));
    }

    #[test]
    fn test_kernel_dimensions_seq() {
        assert_eq!(kernel_dims(par_seq()), Ok(HashMap::new()));
    }

    fn par_outer() -> HashMap<String, ParSpec> {
        HashMap::from([(var!("i"), ParSpec {kind: ParKind::GpuThreads(128)})])
    }

    #[test]
    fn test_parallel_map_outer() {
        let entries = vec![ParEntry {loop_var: var!("i"), nthreads: 128}];
        let expected = HashMap::from([
            (var!("i"), entries)
        ]);
        assert_eq!(map_parallelism(par_outer()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_par_outer() {
        let expected = HashMap::from([
            (var!("i"), KernelGridState {
                blocks: Dim3::default(),
                threads: Dim3::default().with_x(128),
                mapping: vec![GridEntry {ty: GridType::Thread, dim: Dim::X, divisor: 1}]
            }),
        ]);
        assert_eq!(kernel_dims(par_outer()), Ok(expected));
    }

    fn par_two_layers() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("i"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("j"), ParSpec {kind: ParKind::GpuThreads(256)})
        ])
    }

    #[test]
    fn test_parallel_map_two_layers() {
        let entries = vec![
            ParEntry {loop_var: var!("j"), nthreads: 256},
            ParEntry {loop_var: var!("i"), nthreads: 128}
        ];
        let expected = HashMap::from([
            (var!("i"), entries)
        ]);
        assert_eq!(map_parallelism(par_two_layers()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_two_layers() {
        let expected = HashMap::from([
            (var!("i"), KernelGridState {
                blocks: Dim3::default().with_x(128),
                threads: Dim3::default().with_x(256),
                mapping: vec![
                    GridEntry {ty: GridType::Thread, dim: Dim::X, divisor: 1},
                    GridEntry {ty: GridType::Block, dim: Dim::X, divisor: 1}
                ]
            })
        ]);
        assert_eq!(kernel_dims(par_two_layers()), Ok(expected));
    }

    fn par_three_layers() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("i"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("j"), ParSpec {kind: ParKind::GpuThreads(256)}),
            (var!("k"), ParSpec {kind: ParKind::GpuThreads(64)})
        ])
    }

    #[test]
    fn test_parallel_map_three_layers() {
        let entries = vec![
            ParEntry {loop_var: var!("k"), nthreads: 64},
            ParEntry {loop_var: var!("j"), nthreads: 256},
            ParEntry {loop_var: var!("i"), nthreads: 128}
        ];
        let expected = HashMap::from([
            (var!("i"), entries)
        ]);
        assert_eq!(map_parallelism(par_three_layers()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_three_layers() {
        let expected = HashMap::from([
            (var!("i"), KernelGridState {
                blocks: Dim3::default().with_x(256).with_y(128),
                threads: Dim3::default().with_x(64),
                mapping: vec![
                    GridEntry {ty: GridType::Thread, dim: Dim::X, divisor: 1},
                    GridEntry {ty: GridType::Block, dim: Dim::X, divisor: 1},
                    GridEntry {ty: GridType::Block, dim: Dim::Y, divisor: 1}
                ]
            })
        ]);
        assert_eq!(kernel_dims(par_three_layers()), Ok(expected));
    }

    fn inner_seq() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("i"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("k"), ParSpec {kind: ParKind::GpuThreads(64)})
        ])
    }

    #[test]
    fn test_parallel_map_inner_seq() {
        let entries = vec![
            ParEntry {loop_var: var!("k"), nthreads: 64},
            ParEntry {loop_var: var!("i"), nthreads: 128}
        ];
        let expected = HashMap::from([
            (var!("i"), entries)
        ]);
        assert_eq!(map_parallelism(inner_seq()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_inner_seq() {
        let expected = HashMap::from([
            (var!("i"), KernelGridState {
                blocks: Dim3::default().with_x(128),
                threads: Dim3::default().with_x(64),
                mapping: vec![
                    GridEntry {ty: GridType::Thread, dim: Dim::X, divisor: 1},
                    GridEntry {ty: GridType::Block, dim: Dim::X, divisor: 1}
                ]
            })
        ]);
        assert_eq!(kernel_dims(inner_seq()), Ok(expected));
    }

    fn nested_par() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("j"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("l"), ParSpec {kind: ParKind::GpuThreads(256)})
        ])
    }

    #[test]
    fn test_parallel_map_nested_par() {
        let entries = vec![
            ParEntry {loop_var: var!("l"), nthreads: 256},
            ParEntry {loop_var: var!("j"), nthreads: 128}
        ];
        let expected = HashMap::from([
            (var!("j"), entries)
        ]);
        assert_eq!(map_parallelism(nested_par()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_nested_par() {
        let expected = HashMap::from([
            (var!("j"), KernelGridState {
                blocks: Dim3::default().with_x(128),
                threads: Dim3::default().with_x(256),
                mapping: vec![
                    GridEntry {ty: GridType::Thread, dim: Dim::X, divisor: 1},
                    GridEntry {ty: GridType::Block, dim: Dim::X, divisor: 1}
                ]
            })
        ]);
        assert_eq!(kernel_dims(nested_par()), Ok(expected));
    }

    fn par_many_threads() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("i"), ParSpec {kind: ParKind::GpuThreads(2048)})
        ])
    }

    #[test]
    fn test_parallel_map_many_threads() {
        let entries = vec![
            ParEntry {loop_var: var!("i"), nthreads: 2048}
        ];
        let expected = HashMap::from([(var!("i"), entries)]);
        assert_eq!(map_parallelism(par_many_threads()), Ok(expected));
    }

    #[test]
    fn test_kernel_dimensions_many_threads() {
        let expected = HashMap::from([
            (var!("i"), KernelGridState {
                blocks: Dim3::default().with_x(4),
                threads: Dim3::default().with_x(512),
                mapping: vec![
                    GridEntry {ty: GridType::Both, dim: Dim::X, divisor: 1}
                ]
            })
        ]);
        assert_eq!(kernel_dims(par_many_threads()), Ok(expected));
    }

    // The below example is considered inconsistent in the current implementation. In this case,
    // the for-loops over k and l are in the same level of nesting, yet they are mapped to distinct
    // numbers of parallel threads.
    fn inconsistent_par() -> HashMap<String, ParSpec> {
        HashMap::from([
            (var!("i"), ParSpec {kind: ParKind::GpuThreads(128)}),
            (var!("j"), ParSpec {kind: ParKind::GpuThreads(256)}),
            (var!("k"), ParSpec {kind: ParKind::GpuThreads(64)}),
            (var!("l"), ParSpec {kind: ParKind::GpuThreads(32)})
        ])
    }

    #[test]
    fn test_parallel_map_inconsistent() {
        assert!(map_parallelism(inconsistent_par()).is_err());
    }
}
