/// Functions defining how to map the parallel specification of for-loops in a Python function to a
/// GPU grid using a straightforward approach.

use super::ast::{Dim, LaunchArgs};
use crate::parir_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use itertools::Itertools;

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
        Stmt::Definition {i, ..} | Stmt::Assign {i, ..} |
        Stmt::SyncPoint {i, ..} | Stmt::Alloc {i, ..} |
        Stmt::Free {i, ..} => {
            Ok(Par {n: vec![], i: i.clone()})
        },
        Stmt::If {thn, els, i, ..} => {
            let thn = find_parallel_structure_stmts_par(thn)?;
            let els = find_parallel_structure_stmts_par(els)?;
            // For if-conditions, we require that both branches have exactly the same parallel
            // structure. That is, we do not allow parallelism only in one branch - this is due to
            // the performance implications this might have.
            if thn.n.eq(&els.n) {
                Ok(Par {n: thn.n, i: i.clone()})
            } else {
                let lhs = thn.n.iter().join(", ");
                let rhs = els.n.iter().join(", ");
                let msg = format!(
                    "Found branches with inconsistent parallel structure: \
                     [{lhs}] != [{rhs}].\n\
                     This is not supported because the compiller cannot generate \
                     efficient code for this."
                );
                parir_compile_error!(i, "{}", msg)
            }
        },
        Stmt::For {body, par, ..} => {
            let mut inner_par = find_parallel_structure_stmts_par(body)?;
            if par.is_parallel() {
                inner_par.n.insert(0, par.nthreads);
            };
            Ok(inner_par)
        },
        Stmt::While {body, ..} => {
            find_parallel_structure_stmts_par(body)
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
        Stmt::For {var, body, par, ..} if par.is_parallel() => {
            let mut p = find_parallel_structure_stmts_par(body)?;
            p.n.insert(0, par.nthreads);
            acc.insert(var.clone(), p.n);
            Ok(acc)
        },
        Stmt::For {body, ..} => find_parallel_structure_stmts_seq(acc, body),
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Alloc {..} | Stmt::Free {..} => {
            stmt.sfold_result(Ok(acc), find_parallel_structure_stmt_seq)
        }
    }
}

fn find_parallel_structure_stmts_seq(
    acc: BTreeMap<Name, Vec<i64>>,
    stmts: &Vec<Stmt>
) -> ParResult {
    stmts.sfold_result(Ok(acc), find_parallel_structure_stmt_seq)
}

fn find_parallel_structure_fun_def(fun_def: &FunDef) -> ParResult {
    let map = BTreeMap::new();
    find_parallel_structure_stmts_seq(map, &fun_def.body)
}

pub fn find_parallel_structure(ast: &Ast) -> ParResult {
    find_parallel_structure_fun_def(&ast.fun)
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuMap {
    Block {n: i64, mult: i64, dim: Dim},
    Thread {n: i64, mult: i64, dim: Dim},
    ThreadBlock {n: i64, nthreads: i64, nblocks: i64, dim: Dim}
}

pub const DEFAULT_TPB : i64 = 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMapping {
    pub grid: LaunchArgs,
    pub mapping: Vec<GpuMap>,
    pub tpb: i64
}

impl GpuMapping {
    pub fn add_parallelism(self, n: i64) -> Self {
        match self.mapping.last() {
            Some(_) => self.add_block_par(n),
            None => {
                if n > self.tpb {
                    let GpuMapping {grid, mut mapping, tpb} = self;
                    let nblocks = (n + tpb - 1) / tpb;
                    let grid = grid
                        .with_blocks_dim(&Dim::X, nblocks)
                        .with_threads_dim(&Dim::X, tpb);
                    mapping.push(GpuMap::ThreadBlock {n, nthreads: tpb, nblocks, dim: Dim::X});
                    GpuMapping {grid, mapping, tpb}
                } else {
                    self.with_thread_par(Dim::X, n)
                }
            },
        }
    }

    pub fn with_thread_par(self, dim: Dim, n: i64) -> Self {
        let GpuMapping {grid, mut mapping, tpb} = self;
        let old_dim_threads = grid.threads.get_dim(&dim);
        let new_dim_threads = old_dim_threads * n;
        let grid = grid.with_threads_dim(&dim, new_dim_threads);
        mapping.push(GpuMap::Thread {n, dim, mult: old_dim_threads});
        GpuMapping {grid, mapping, tpb}
    }

    pub fn add_block_par(self, n: i64) -> Self {
        if self.grid.blocks.x == 1 {
            self.with_block_par(Dim::X, n)
        } else if self.grid.blocks.y == 1 {
            self.with_block_par(Dim::Y, n)
        } else {
            self.with_block_par(Dim::Z, n)
        }
    }

    pub fn with_block_par(self, dim: Dim, n: i64) -> Self {
        let GpuMapping {grid, mut mapping, tpb} = self;
        let old_dim_blocks = grid.blocks.get_dim(&dim);
        let new_dim_blocks = old_dim_blocks * n;
        let grid = grid.with_blocks_dim(&dim, new_dim_blocks);
        mapping.push(GpuMap::Block {n, dim, mult: old_dim_blocks});
        GpuMapping {grid, mapping, tpb}
    }

    pub fn rev_mapping(mut self) -> Self {
        self.mapping.reverse();
        self
    }

    pub fn get_mapping(&self) -> Vec<GpuMap> {
        self.mapping.clone()
    }
}

impl Default for GpuMapping {
    fn default() -> Self {
        GpuMapping {
            grid: LaunchArgs::default(),
            mapping: vec![],
            tpb: DEFAULT_TPB}
    }
}

fn add_par_to_grid(m: GpuMapping, n: i64) -> GpuMapping {
    m.add_parallelism(n)
}

fn par_to_gpu_mapping(par: Vec<i64>) -> GpuMapping {
    par.into_iter()
        .rev()
        .fold(GpuMapping::default(), add_par_to_grid)
        .rev_mapping()
}

pub fn map_gpu_grid(
    par: BTreeMap<Name, Vec<i64>>
) -> BTreeMap<Name, GpuMapping> {
    par.into_iter()
        .map(|(id, par)| (id, par_to_gpu_mapping(par)))
        .collect::<BTreeMap<Name, GpuMapping>>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder;
    use crate::ir::ir_builder::*;

    fn id(s: &str) -> Name {
        ir_builder::id(s).with_new_sym()
    }

    fn find_structure(def: FunDef) -> BTreeMap<Name, Vec<i64>> {
        find_parallel_structure_fun_def(&def).unwrap()
    }

    fn to_map<T1: Ord, T2>(v: Vec<(T1, T2)>) -> BTreeMap<T1, T2> {
        v.into_iter().collect::<BTreeMap<T1, T2>>()
    }

    #[test]
    fn seq_loops() {
        let def = fun_def(vec![for_loop(id("x"), 0, vec![for_loop(id("y"), 0, vec![])])]);
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
            vec![for_loop(id("z"), 3, vec![for_loop(id("w"), 0, vec![])])],
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
            for_loop(x.clone(), 10, vec![for_loop(id("y"), 15, vec![for_loop(id("z"), 0, vec![])])])
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
                vec![for_loop(id("y"), 0, vec![for_loop(id("z"), 15, vec![])])]
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
                    vec![for_loop(id("y"), 0, vec![for_loop(id("z"), 6, vec![])])]
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

    fn assert_par_mapping(par: Vec<i64>, mapping: GpuMapping) {
        let x = id("x");
        let par = to_map(vec![(x.clone(), par)]);
        let expected = to_map(vec![(x, mapping)]);
        assert_eq!(map_gpu_grid(par), expected);
    }

    #[test]
    fn map_empty_grid() {
        assert_eq!(map_gpu_grid(BTreeMap::new()), BTreeMap::new());
    }

    #[test]
    fn map_threads_single_par() {
        let par = vec![128];
        let mapping = GpuMapping {
            grid: LaunchArgs::default().with_threads_dim(&Dim::X, 128),
            mapping: vec![GpuMap::Thread {n: 128, dim: Dim::X, mult: 1}],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_many_threads_to_grid() {
        let par = vec![2048];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2),
            mapping: vec![GpuMap::ThreadBlock {n: 2048, nthreads: 1024, nblocks: 2, dim: Dim::X}],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_par_threads_and_blocks() {
        let par = vec![64, 128];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 128)
                .with_blocks_dim(&Dim::X, 64),
            mapping: vec![
                GpuMap::Block {n: 64, dim: Dim::X, mult: 1},
                GpuMap::Thread {n: 128, dim: Dim::X, mult: 1}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_multi_blocks() {
        let par = vec![512, 128, 1682];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2)
                .with_blocks_dim(&Dim::Y, 128)
                .with_blocks_dim(&Dim::Z, 512),
            mapping: vec![
                GpuMap::Block {n: 512, dim: Dim::Z, mult: 1},
                GpuMap::Block {n: 128, dim: Dim::Y, mult: 1},
                GpuMap::ThreadBlock {n: 1682, nthreads: 1024, nblocks: 2, dim: Dim::X}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_overlapping_dim_blocks() {
        let par = vec![64, 512, 128, 1682];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2)
                .with_blocks_dim(&Dim::Y, 128)
                .with_blocks_dim(&Dim::Z, 512*64),
            mapping: vec![
                GpuMap::Block {n: 64, dim: Dim::Z, mult: 512},
                GpuMap::Block {n: 512, dim: Dim::Z, mult: 1},
                GpuMap::Block {n: 128, dim: Dim::Y, mult: 1},
                GpuMap::ThreadBlock {n: 1682, nthreads: 1024, nblocks: 2, dim: Dim::X}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }
}
