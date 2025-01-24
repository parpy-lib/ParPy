/// Functions defining how to map the parallel specification of for-loops in a Python function to a
/// GPU grid using a straightforward approach.

use super::ast::{Dim, LaunchArgs};
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
        Stmt::For {body, par, ..} => {
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
        Stmt::For {var, body, par, ..} => {
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

#[derive(Clone, Debug, PartialEq)]
pub enum GpuMap {
    Block {n: i64, dim: Dim},
    Thread {n: i64, dim: Dim},
    ThreadBlock {nthreads: i64, nblocks: i64, dim: Dim}
}

pub const DEFAULT_TPB : i64 = 512;

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMapping {
    pub grid: LaunchArgs,
    pub mapping: Vec<GpuMap>,
    pub tpb: i64
}

impl GpuMapping {
    pub fn add_parallelism(self, n: i64) -> Self {
        match self.mapping.last() {
            Some(GpuMap::Block {..}) | Some(GpuMap::ThreadBlock {..}) => {
                self.add_block_par(n)
            },
            Some(GpuMap::Thread {..}) => {
                if self.grid.threads.product() * n > self.tpb {
                    self.add_block_par(n)
                } else {
                    self.add_thread_par(n)
                }
            },
            None => {
                if n > self.tpb {
                    let GpuMapping {grid, mut mapping, tpb} = self;
                    let nblocks = (n + tpb - 1) / tpb;
                    let grid = grid
                        .with_blocks_dim(Dim::X, nblocks)
                        .with_threads_dim(Dim::X, tpb);
                    mapping.push(GpuMap::ThreadBlock {nthreads: tpb, nblocks, dim: Dim::X});
                    GpuMapping {grid, mapping, tpb}
                } else {
                    self.with_thread_par(Dim::X, n)
                }
            },
        }
    }

    pub fn add_thread_par(self, n: i64) -> Self {
        if self.grid.threads.x == 1 {
            self.with_thread_par(Dim::X, n)
        } else if self.grid.threads.y == 1 {
            self.with_thread_par(Dim::Y, n)
        } else {
            self.with_thread_par(Dim::Z, n)
        }
    }

    pub fn with_thread_par(self, dim: Dim, n: i64) -> Self {
        let GpuMapping {grid, mut mapping, tpb} = self;
        let new_dim_threads = grid.threads.get_dim(dim) * n;
        let grid = grid.with_threads_dim(dim, new_dim_threads);
        mapping.push(GpuMap::Thread {n, dim});
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
        let new_dim_blocks = grid.blocks.get_dim(dim) * n;
        let grid = grid.with_blocks_dim(dim, new_dim_blocks);
        mapping.push(GpuMap::Block {n, dim});
        GpuMapping {grid, mapping, tpb}
    }

    pub fn rev_mapping(mut self) -> Self {
        self.mapping.reverse();
        self
    }

    pub fn get_mapping(&self) -> Vec<GpuMap> {
        self.mapping.clone().into_iter().rev().collect::<Vec<GpuMap>>()
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

    fn to_map<T1: Ord, T2>(v: Vec<(T1, T2)>) -> BTreeMap<T1, T2> {
        v.into_iter().collect::<BTreeMap<T1, T2>>()
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
            grid: LaunchArgs::default().with_threads_dim(Dim::X, 128),
            mapping: vec![GpuMap::Thread {n: 128, dim: Dim::X}],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_many_threads_to_grid() {
        let par = vec![2048];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(Dim::X, 512)
                .with_blocks_dim(Dim::X, 4),
            mapping: vec![GpuMap::ThreadBlock {nthreads: 512, nblocks: 4, dim: Dim::X}],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_par_threads_and_blocks() {
        let par = vec![64, 128];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(Dim::X, 128)
                .with_blocks_dim(Dim::X, 64),
            mapping: vec![
                GpuMap::Block {n: 64, dim: Dim::X},
                GpuMap::Thread {n: 128, dim: Dim::X}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_multi_threads() {
        let par = vec![4, 8, 12];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(Dim::X, 12)
                .with_threads_dim(Dim::Y, 8)
                .with_threads_dim(Dim::Z, 4),
            mapping: vec![
                GpuMap::Thread {n: 4, dim: Dim::Z},
                GpuMap::Thread {n: 8, dim: Dim::Y},
                GpuMap::Thread {n: 12, dim: Dim::X}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_multi_blocks() {
        let par = vec![512, 128, 2048];
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(Dim::X, 512)
                .with_blocks_dim(Dim::X, 4)
                .with_blocks_dim(Dim::Y, 128)
                .with_blocks_dim(Dim::Z, 512),
            mapping: vec![
                GpuMap::Block {n: 512, dim: Dim::Z},
                GpuMap::Block {n: 128, dim: Dim::Y},
                GpuMap::ThreadBlock {nthreads: 512, nblocks: 4, dim: Dim::X}
            ],
            tpb: DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }
}
