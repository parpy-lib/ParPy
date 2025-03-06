use super::par_tree;
use crate::parir_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::{SFold, SMapAccum};

fn insert_synchronization_points_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} => {
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let is_par = par.is_parallel();
            let body = insert_synchronization_points(body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i: i.clone()});
            if is_par {
                acc.push(Stmt::SyncPoint {block_local: false, i});
            }
        },
        Stmt::While {cond, body, i} => {
            let body = insert_synchronization_points(body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = insert_synchronization_points(thn);
            let els = insert_synchronization_points(els);
            acc.push(Stmt::If {cond, thn, els, i});
        },
    }
    acc
}

fn insert_synchronization_points(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.into_iter()
        .fold(vec![], |acc, s| {
            insert_synchronization_points_stmt(acc, s)
        })
}

fn classify_synchronization_points_par_stmt(
    node: &par_tree::ParNode,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> Vec<Stmt> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} => {
            acc.push(s);
        },
        Stmt::SyncPoint {block_local, i} => {
            let prev_stmt_is_block_local_for = match acc.last() {
                Some(Stmt::For {par, ..}) => par.nthreads > 0 && par.nthreads <= 1024,
                _ => false
            };
            // When the synchronization statement is found in the innermost level of parallelism,
            // and the preceding for-loop is a parallel for-loop with at most 1024 threads, we
            // consider this synchronization point to be block-local. In this case, we can use a
            // CUDA intrinsic instead of splitting up the kernel.
            let s = if node.innermost_parallelism() && prev_stmt_is_block_local_for {
                Stmt::SyncPoint {block_local: true, i}
            } else {
                Stmt::SyncPoint {block_local, i}
            };
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let node = node.children.get(&var).unwrap_or(node);
            let body = classify_synchronization_points_par_stmts(node, body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = classify_synchronization_points_par_stmts(node, body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = classify_synchronization_points_par_stmts(node, thn);
            let els = classify_synchronization_points_par_stmts(node, els);
            acc.push(Stmt::If {cond, thn, els, i});
        }
    };
    acc
}

fn classify_synchronization_points_par_stmts(
    par: &par_tree::ParNode,
    stmts: Vec<Stmt>
) -> Vec<Stmt> {
    stmts.into_iter()
        .fold(vec![], |acc, s| classify_synchronization_points_par_stmt(par, acc, s))
}

fn classify_synchronization_points_stmt(
    t: &par_tree::ParTree,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = if let Some(node) = t.roots.get(&var) {
                classify_synchronization_points_par_stmts(&node, body)
            } else {
                body.smap(|s| classify_synchronization_points_stmt(t, s))
            };
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} => {
            s.smap(|s| classify_synchronization_points_stmt(t, s))
        }
    }
}

fn classify_synchronization_points(
    par: &par_tree::ParTree,
    body: Vec<Stmt>
) -> Vec<Stmt> {
    body.smap(|s| classify_synchronization_points_stmt(par, s))
}

fn is_inter_block_sync_point(s: &Stmt) -> bool {
    match s {
        Stmt::SyncPoint {block_local: false, ..} => true,
        _ => false
    }
}

fn contains_inter_block_sync_point(acc: bool, s: &Stmt) -> bool {
    s.sfold(acc || is_inter_block_sync_point(s), contains_inter_block_sync_point)
}

// Determines if the statement is a sequential for-loop containing an inter-block synchronization
// point.
fn is_seq_loop_with_inter_block_sync_point(s: &Stmt) -> bool {
    match s {
        Stmt::For {body, par, ..} if !par.is_parallel() => {
            body.sfold(false, contains_inter_block_sync_point)
        },
        _ => false
    }
}

fn hoist_chunk(
    var: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    par: LoopParallelism,
    i: Info,
    chunk: &[Stmt]
) -> CompileResult<Vec<Stmt>> {
    // As we perform an inclusive split, each part of the split will always contain at
    // least one element, so it is safe to unwrap this.
    let last_stmt = chunk.last().unwrap();

    if is_seq_loop_with_inter_block_sync_point(last_stmt) {
        // If the last statement of the chunk is a sequential loop with an inter-block
        // synchronization point, we extract the pre-statements and then process the
        // sequential loop afterward.
        let pre_stmts = hoist_inner_seq_loops_par_stmts(chunk[..chunk.len()-1].to_vec())?;
        let pre_stmt = Stmt::For {
            var: var.clone(),
            lo: lo.clone(),
            hi: hi.clone(),
            step,
            body: pre_stmts,
            par: par.clone(),
            i: i.clone()
        };
        let seq_loop_stmt = match last_stmt.clone() {
            Stmt::For {var: seq_var, lo: seq_lo, hi: seq_hi, step: seq_step,
                       body: seq_body, par: seq_par, i: seq_i} => {
                // Split up the body of the sequential for-loop such that each inter-block
                // synchronization point is at the end of a chunk. We place each chunk inside the
                // outer parallel for-loop.
                let inner_stmts = seq_body.split_inclusive(is_inter_block_sync_point)
                    .map(|chunk| {
                        let s = Stmt::For {
                            var: var.clone(),
                            lo: lo.clone(),
                            hi: hi.clone(),
                            step,
                            body: chunk.to_vec(),
                            par: par.clone(),
                            i: i.clone()
                        };
                        // Include a inter-block synchronization point after each parallel for-loop
                        // to ensure it is properly split up. Later, the remaining synchronization
                        // points can be eliminated as they will end up at the end of a parallel
                        // for-loop.
                        vec![s, Stmt::SyncPoint {block_local: false, i: i.clone()}]
                    })
                    .map(hoist_inner_seq_loops_par_stmts)
                    .collect::<CompileResult<Vec<Vec<Stmt>>>>()?
                    .concat();
                // Reconstruct the sequential for-loop outside of the parallel for-loops.
                Ok(Stmt::For {
                    var: seq_var, lo: seq_lo, hi: seq_hi, step: seq_step,
                    body: inner_stmts, par: seq_par, i: seq_i
                })
            },
            _ => parir_compile_error!(&i, "Internal error when hoisting \
                                           sequential loop")
        }?;
        Ok(vec![pre_stmt, seq_loop_stmt])
    } else {
        // Otherwise, if the chunk does not contain any applicable sequential loops, we recurse
        // into the body to produce the resulting body of the parallel for-loop.
        let body = hoist_inner_seq_loops_par_stmts(chunk.to_vec())?;

        // If the body contains an applicable loop after recursing down, we run the outer
        // transformation again to hoist it outside of this loop as well.
        if body.iter().any(is_seq_loop_with_inter_block_sync_point) {
            hoist_seq_loops(var, lo, hi, step, body, par, i)
        } else {
            Ok(vec![Stmt::For { var, lo, hi, step, body, par, i }])
        }
    }
}

fn hoist_seq_loops(
    var: Name,
    lo: Expr,
    hi: Expr,
    step: i64,
    body: Vec<Stmt>,
    par: LoopParallelism,
    i: Info
) -> CompileResult<Vec<Stmt>> {
    Ok(body.split_inclusive(is_seq_loop_with_inter_block_sync_point)
        .map(|chunk| {
            hoist_chunk(var.clone(), lo.clone(), hi.clone(), step, par.clone(),
                        i.clone(), chunk)
        })
        .collect::<CompileResult<Vec<Vec<Stmt>>>>()?
        .concat())
}

fn hoist_inner_seq_loops_par_stmt(
    acc: CompileResult<Vec<Stmt>>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    let mut acc = acc?;
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} => {
            acc.push(s);
        },
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            acc.append(&mut hoist_seq_loops(var, lo, hi, step, body, par, i)?);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = hoist_inner_seq_loops_par_stmts(body)?;
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = hoist_inner_seq_loops_par_stmts(body)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = hoist_inner_seq_loops_par_stmts(thn)?;
            let els = hoist_inner_seq_loops_par_stmts(els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        }
    };
    Ok(acc)
}

fn hoist_inner_seq_loops_par_stmts(stmts: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(vec![]), hoist_inner_seq_loops_par_stmt)
}

fn hoist_inner_sequential_loops_stmt(
    t: &par_tree::ParTree,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} => {
            acc.push(s);
        },
        Stmt::For {ref var, ..} if t.roots.contains_key(&var) => {
            let mut stmts = hoist_inner_seq_loops_par_stmt(Ok(vec![]), s)?;
            acc.append(&mut stmts);
        },
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let body = hoist_inner_sequential_loops(t, body)?;
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let body = hoist_inner_sequential_loops(t, body)?;
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = hoist_inner_sequential_loops(t, thn)?;
            let els = hoist_inner_sequential_loops(t, els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        },
    }
    Ok(acc)
}

fn hoist_inner_sequential_loops(
    t: &par_tree::ParTree,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    body.into_iter()
        .fold(Ok(vec![]), |acc, s| hoist_inner_sequential_loops_stmt(t, acc?, s))
}

#[derive(Clone, Debug)]
struct SyncPointEnv {
    in_parallel: bool,
    parallel_loop_body: bool
}

impl SyncPointEnv {
    fn new() -> Self {
        SyncPointEnv {in_parallel: false, parallel_loop_body: false}
    }

    fn enter_loop(&self, is_parallel: bool) -> Self {
        SyncPointEnv {
            in_parallel: self.in_parallel || is_parallel,
            parallel_loop_body: is_parallel
        }
    }
}

fn eliminate_unnecessary_synchronization_points_stmt(
    env: SyncPointEnv,
    mut acc: Vec<Stmt>,
    s: Stmt,
) -> Vec<Stmt> {
    match s {
        Stmt::SyncPoint {..} if !env.in_parallel => (),
        Stmt::For {var, lo, hi, step, body, par, i} => {
            let env = env.enter_loop(par.is_parallel());
            let body = eliminate_unnecessary_synchronization_points_stmts(env, body);
            acc.push(Stmt::For {var, lo, hi, step, body, par, i});
        },
        Stmt::While {cond, body, i} => {
            let env = env.enter_loop(false);
            let body = eliminate_unnecessary_synchronization_points_stmts(env, body);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = eliminate_unnecessary_synchronization_points_stmts(env.clone(), thn);
            let els = eliminate_unnecessary_synchronization_points_stmts(env, els);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} => {
            acc.push(s);
        },
    }
    acc
}

fn eliminate_unnecessary_synchronization_points_stmts(
    env: SyncPointEnv,
    mut stmts: Vec<Stmt>,
) -> Vec<Stmt> {
    if env.parallel_loop_body {
        if let Some(Stmt::SyncPoint {..}) = stmts.last() {
            stmts.pop();
        }
    }
    stmts.into_iter()
        .fold(vec![], |acc, s| {
            eliminate_unnecessary_synchronization_points_stmt(env.clone(), acc, s)
        })
}

fn eliminate_unnecessary_synchronization_points(body: Vec<Stmt>) -> Vec<Stmt> {
    let env = SyncPointEnv::new();
    eliminate_unnecessary_synchronization_points_stmts(env, body)
}

/// Adds explicit synchronization points in the AST where this is necessary, and eliminates the
/// ones that are unnecessary before returning the updated AST with synchronization.
///
/// The iterations of a parallel for-loop can run in arbitrary order, but all iterations must
/// complete before executing subsequent statements, as these may depend on the result of the
/// parallel for-loop. Therefore, we insert a synchronization point after every parallel for-loop.
///
/// We also insert a synchronization point after a statement which contains incompatible
/// parallelism with respect to subsequent statements. That is, we consider the statements on a
/// level of nesting in reverse order, and insert synchronization points when we find a statement
/// whose parallelism is incompatible with what we have seen so far. These synchronization points
/// are used to guide later transformations.
///
/// Finally, we eliminate synchronization points that are unnecessary. First, the outermost
/// parallel for-loop will be translated to a CUDA kernel entry. CUDA ensures that one kernel
/// completes before the next one starts, so we eliminate synchronization points of the outermost
/// parallel for-loops. Second, we eliminate synchronization points at the end of a parallel
/// for-loop (when the last statement is a synchronize node) because there is no need for the
/// iterations of a parallel for-loop to wait for one another.
pub fn add_synchronization_points(ast: Ast) -> CompileResult<Ast> {
    let Ast {fun: FunDef {id, params, body, i}, structs} = ast;
    let body = insert_synchronization_points(body);
    let par = par_tree::build_tree(&body);
    let body = classify_synchronization_points(&par, body);
    let body = hoist_inner_sequential_loops(&par, body)?;
    let body = eliminate_unnecessary_synchronization_points(body);
    Ok(Ast {fun: FunDef {id, params, body, i}, structs})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder::*;
    use crate::utils::pprint::*;

    fn for_(id: &str, n: i64, body: Vec<Stmt>) -> Stmt {
        let var = Name::new(id.to_string());
        for_loop(var, n, body)
    }

    fn stmts_str(stmts: &Vec<Stmt>) -> String {
        let (_, s) = pprint_iter(stmts.iter(), PrettyPrintEnv::new(), "\n");
        s
    }

    fn print_stmts(lhs: &Vec<Stmt>, rhs: &Vec<Stmt>) {
        let separator = str::repeat("=", 10);
        println!("{0}\n{1}\n{2}", stmts_str(&lhs), separator, stmts_str(&rhs));
    }

    fn assert_sync(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let body = insert_synchronization_points(body);
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    fn assert_classify(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let par = par_tree::build_tree(&body);
        let body = make_ast(classify_synchronization_points(&par, body));
        let expected = make_ast(expected);
        println!("{0}\n{1}", body.pprint_default(), expected.pprint_default());
        assert_eq!(body, expected);
    }

    fn assert_hoist(body: Vec<Stmt>, expected: Vec<Stmt>) {
        let par = par_tree::build_tree(&body);
        let body = hoist_inner_sequential_loops(&par, body).unwrap();
        let body = eliminate_unnecessary_synchronization_points(body);
        print_stmts(&body, &expected);
        assert_eq!(body, expected);
    }

    #[test]
    fn empty_sync_points() {
        assert_sync(vec![], vec![]);
    }

    #[test]
    fn single_par_loop_sync_points() {
        let s = vec![for_("x", 10, vec![])];
        let expected = vec![
            for_("x", 10, vec![]),
            sync_point(false)
        ];
        assert_sync(s, expected);
    }

    #[test]
    fn subsequent_par_loops_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 32, vec![]),
                for_("z", 32, vec![])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 32, vec![]),
                sync_point(false),
                for_("z", 32, vec![]),
                sync_point(false)
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_loops_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(false),
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_loops_classify() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(false),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![]),
                    sync_point(true),
                ])
            ])
        ];
        assert_classify(body, expected);
    }

    #[test]
    fn par_seq_par_loops_local_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(var("q"), int(3))]),
                    sync_point(true),
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 15, vec![assign(var("q"), int(3))]),
                    sync_point(true)
                ])
            ])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_seq_par_loops_inter_block_sync_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(4)),
                for_("y", 0, vec![
                    assign(var("b"), int(0)),
                    for_("z", 2048, vec![assign(var("c"), int(3))]),
                    sync_point(false),
                    assign(var("d"), int(1))
                ]),
                assign(var("e"), int(2))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(4))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(var("b"), int(0)),
                    for_("z", 2048, vec![assign(var("c"), int(3))])
                ]),
                for_("x", 10, vec![assign(var("d"), int(1))])
            ]),
            for_("x", 10, vec![assign(var("e"), int(2))])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_in_cond_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                if_cond(
                    vec![],
                    vec![for_("y", 10, vec![])]
                )
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                if_cond(
                    vec![],
                    vec![
                        for_("y", 10, vec![]),
                        sync_point(false)
                    ]
                )
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_in_while_sync_point() {
        let body = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(false)
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_in_while_classify() {
        let body = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                while_loop(vec![
                    for_("y", 10, vec![]),
                    sync_point(true)
                ])
            ])
        ];
        assert_classify(body, expected)
    }

    #[test]
    fn par_seq_par_seq_par_sync_points() {
        let body = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![])
                        ])
                    ])
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(false)
                        ])
                    ]),
                    sync_point(false)
                ])
            ]),
            sync_point(false)
        ];
        assert_sync(body, expected);
    }

    #[test]
    fn par_seq_par_seq_par_classify() {
        let identified_sync_points = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(false)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![]),
                            sync_point(true)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        assert_classify(identified_sync_points, expected);
    }

    #[test]
    fn par_seq_par_seq_par_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(1)),
                for_("y", 0, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(var("b"), int(2))]),
                            sync_point(true)
                        ])
                    ]),
                    sync_point(false)
                ])
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(1))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    for_("z", 10, vec![
                        for_("w", 0, vec![
                            for_("v", 10, vec![assign(var("b"), int(2))]),
                            sync_point(true)
                        ])
                    ])
                ])
            ])
        ];
        assert_hoist(body, expected)
    }

    #[test]
    fn par_seq_par_seq_par_double_inter_block_hoisting() {
        let body = vec![
            for_("x", 10, vec![
                assign(var("a"), int(1)),
                for_("y", 0, vec![
                    assign(var("b"), int(2)),
                    for_("z", 10, vec![
                        assign(var("c"), int(3)),
                        for_("w", 0, vec![
                            assign(var("d"), int(4)),
                            for_("v", 2048, vec![assign(var("e"), int(5))]),
                            sync_point(false),
                            assign(var("f"), int(6))
                        ]),
                        assign(var("g"), int(7))
                    ]),
                    sync_point(false),
                    assign(var("h"), int(8))
                ]),
                assign(var("i"), int(9))
            ])
        ];
        let expected = vec![
            for_("x", 10, vec![assign(var("a"), int(1))]),
            for_("y", 0, vec![
                for_("x", 10, vec![
                    assign(var("b"), int(2)),
                    for_("z", 10, vec![assign(var("c"), int(3))])
                ]),
                for_("w", 0, vec![
                    for_("x", 10, vec![
                        for_("z", 10, vec![
                            assign(var("d"), int(4)),
                            for_("v", 2048, vec![assign(var("e"), int(5))]),
                        ])
                    ]),
                    for_("x", 10, vec![
                        for_("z", 10, vec![assign(var("f"), int(6))]),
                    ]),
                ]),
                for_("x", 10, vec![
                    for_("z", 10, vec![assign(var("g"), int(7))]),
                ]),
                for_("x", 10, vec![assign(var("h"), int(8))]),
            ]),
            for_("x", 10, vec![assign(var("i"), int(9))])
        ];
        assert_hoist(body, expected)
    }
}
