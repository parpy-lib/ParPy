use crate::ir::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct ParNode {
    pub value: LoopPar,
    pub children: BTreeMap<Name, ParNode>
}

impl ParNode {
    pub fn new(value: LoopPar) -> ParNode {
        ParNode {value, children: BTreeMap::new()}
    }

    pub fn with_child(mut self, id: Name, node: ParNode) -> ParNode {
        self.children.insert(id, node);
        self
    }

    fn parallel_height(&self) -> i64 {
        let h = if self.value.is_parallel() { 1 } else { 0 };
        let inner_height = self.children.iter()
            .map(|(_, n)| n.parallel_height())
            .max()
            .unwrap_or(0);
        h + inner_height
    }

    fn inner_par_height(&self) -> i64 {
        let inner_height = self.children.iter()
            .map(|(_, n)| n.parallel_height())
            .max()
            .unwrap_or(0);
        inner_height
    }

    pub fn innermost_parallelism(&self) -> bool {
        self.inner_par_height() == 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParTree {
    pub roots: BTreeMap<Name, ParNode>
}

impl ParTree {
    pub fn new() -> ParTree {
        ParTree {roots: BTreeMap::new()}
    }

    pub fn with_root(mut self, id: Name, node: ParNode) -> ParTree {
        self.roots.insert(id, node);
        self
    }
}

fn build_tree_stmt_par(acc: ParNode, s: &Stmt) -> ParNode {
    match s {
        Stmt::For {var, body, par, ..} => {
            let node = body.iter()
                .fold(ParNode::new(par.clone()), build_tree_stmt_par);
            acc.with_child(var.clone(), node)
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Expr {..} | Stmt::Return {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.sfold(acc, build_tree_stmt_par)
        }
    }
}

fn build_tree_stmt(acc: ParTree, s: &Stmt) -> ParTree {
    match s {
        Stmt::For {var, body, par, ..} => {
            if par.is_parallel() {
                let node = body.iter()
                    .fold(ParNode::new(par.clone()), build_tree_stmt_par);
                acc.with_root(var.clone(), node)
            } else {
                body.sfold(acc, build_tree_stmt)
            }
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Expr {..} | Stmt::Return {..} |
        Stmt::Alloc {..} | Stmt::AllocShared {..} | Stmt::Free {..} => {
            s.sfold(acc, build_tree_stmt)
        }

    }
}

pub fn build_tree(body: &Vec<Stmt>) -> ParTree {
    body.iter().fold(ParTree::new(), build_tree_stmt)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ast_builder::*;

    fn assert_par_tree(body: Vec<Stmt>, expected: ParTree) {
        let t = build_tree(&body);
        assert_eq!(t, expected)
    }

    #[test]
    fn empty_stmts_par_tree() {
        assert_par_tree(vec![], ParTree::new());
    }

    #[test]
    fn seq_loops_par_tree() {
        let body = vec![for_loop(id("x"), 0, vec![for_loop(id("y"), 0, vec![])])];
        assert_par_tree(body, ParTree::new());
    }

    #[test]
    fn single_par_loop_par_tree() {
        let body = vec![for_loop(id("x"), 10, vec![])];
        let expected = ParTree::new()
            .with_root(id("x"), ParNode::new(loop_par(10)));
        assert_par_tree(body, expected);
    }

    #[test]
    fn nested_par_loops_par_tree() {
        let body = vec![
            for_loop(id("x"), 10, vec![
                for_loop(id("y"), 12, vec![])
            ])
        ];
        let x_root = ParNode::new(loop_par(10))
            .with_child(id("y"), ParNode::new(loop_par(12)));
        let expected = ParTree::new().with_root(id("x"), x_root);
        assert_par_tree(body, expected);
    }

    #[test]
    fn subsequent_par_loops_par_tree() {
        let body = vec![
            for_loop(id("x"), 10, vec![]),
            for_loop(id("y"), 10, vec![])
        ];
        let expected = ParTree::new()
            .with_root(id("x"), ParNode::new(loop_par(10)))
            .with_root(id("y"), ParNode::new(loop_par(10)));
        assert_par_tree(body, expected);
    }

    #[test]
    fn subsequent_nested_par_loops_par_tree() {
        let body = vec![
            for_loop(id("x"), 10, vec![
                for_loop(id("y"), 32, vec![]),
                for_loop(id("z"), 32, vec![])
            ])
        ];
        let x_root = ParNode::new(loop_par(10))
            .with_child(id("y"), ParNode::new(loop_par(32)))
            .with_child(id("z"), ParNode::new(loop_par(32)));
        let expected = ParTree::new().with_root(id("x"), x_root);
        assert_par_tree(body, expected)
    }

    #[test]
    fn par_seq_par_loops_par_tree() {
        let body = vec![
            for_loop(id("x"), 10, vec![
                for_loop(id("y"), 0, vec![
                    for_loop(id("z"), 14, vec![])
                ])
            ])
        ];
        let x_root = ParNode::new(loop_par(10))
            .with_child(
                id("y"),
                ParNode::new(loop_par(0))
                    .with_child(id("z"), ParNode::new(loop_par(14)))
            );
        let expected = ParTree::new().with_root(id("x"), x_root);
        assert_par_tree(body, expected);
    }

    #[test]
    fn par_seq_par_seq_par_loops_par_tree() {
        let body = vec![
            for_loop(id("x"), 10, vec![
                for_loop(id("y"), 0, vec![
                    for_loop(id("z"), 10, vec![
                        for_loop(id("w"), 0, vec![
                            for_loop(id("v"), 10, vec![])
                        ])
                    ])
                ])
            ])
        ];
        let x_root = ParNode::new(loop_par(10))
            .with_child(
                id("y"),
                ParNode::new(loop_par(0))
                    .with_child(
                        id("z"),
                        ParNode::new(loop_par(10))
                            .with_child(
                                id("w"),
                                ParNode::new(loop_par(0))
                                    .with_child(
                                        id("v"),
                                        ParNode::new(loop_par(10))))));
        let expected = ParTree::new().with_root(id("x"), x_root);
        assert_par_tree(body, expected);
    }
}
