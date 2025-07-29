use super::ast::*;
use crate::option;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

struct GraphEnv {
    inst_id: Name,
    exec_graph_id: Name,
    graph_id: Name,
    stream_id: Name,
    updated_id: Name,
}

fn new_err_id() -> Name {
    Name::sym_str("err")
}

fn generate_pre_kernel_body_statements(env: &GraphEnv, acc: &mut Vec<Stmt>) {
    // Initialize a temporary stream used when capturing the graph.
    acc.push(Stmt::Definition {
        ty: Type::Stream, id: env.stream_id.clone(), expr: None
    });
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::StreamCreate {
            id: env.stream_id.clone(), ty: Type::Stream, i: Info::default()
        }),
    });

    // Begin capturing the CUDA API calls on the temporary stream before running the original
    // function body.
    let stream = Stream::Id(env.stream_id.clone());
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::StreamBeginCapture {
            stream, ty: Type::Stream, i: Info::default()
        })
    });
}

fn generate_post_kernel_body_statements(env: &GraphEnv, acc: &mut Vec<Stmt>) {
    let inst_var = Expr::Var {
        id: env.inst_id.clone(),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: Info::default()
    };

    // End capture of the CUDA API calls after the original function body, and store the recorded
    // graph.
    let stream = Stream::Id(env.stream_id.clone());
    acc.push(Stmt::Definition {
        ty: Type::Graph, id: env.graph_id.clone(), expr: None
    });
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::StreamEndCapture {
            stream, graph: env.graph_id.clone(),
            ty: Type::Error, i: Info::default()
        })
    });

    // Delete the temporary stream used for capturing.
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::StreamDestroy {
            id: env.stream_id.clone(), ty: Type::Error, i: Info::default()
        })
    });

    // If the executable graph has not yet been instantiated, we instantiate it from the captured
    // graph.
    let inst_thn = vec![
        Stmt::Definition {
            ty: Type::Error, id: new_err_id(),
            expr: Some(Expr::GraphExecInstantiate {
                exec_graph: env.exec_graph_id.clone(), graph: env.graph_id.clone(),
                ty: Type::GraphExec, i: Info::default()
            })
        },
        Stmt::Assign {
            dst: inst_var.clone(),
            expr: Expr::Bool {
                v: true,
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: Info::default()
            }
        }
    ];
    
    // Otherwise, we update the existing executable graph based on the newly captured graph.
    let err_id = Name::sym_str("err");
    let err_is_success = Expr::BinOp {
        lhs: Box::new(Expr::Var {id: err_id.clone(), ty: Type::Error, i: Info::default()}),
        op: BinOp::Neq,
        rhs: Box::new(Expr::Error {e: Error::Success, ty: Type::Error, i: Info::default()}),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: Info::default()
    };
    let inst_els = vec![
        Stmt::Definition {
            ty: Type::GraphExecUpdateResultInfo, id: env.updated_id.clone(), expr: None
        },
        Stmt::Definition {
            ty: Type::Error, id: err_id.clone(), expr: Some(Expr::GraphExecUpdate {
                exec_graph: env.exec_graph_id.clone(), graph: env.graph_id.clone(),
                update: env.updated_id.clone(), ty: Type::Error, i: Info::default()
            })
        },
        Stmt::If {
            cond: err_is_success,
            thn: vec![
                Stmt::Definition {
                    ty: Type::Error, id: new_err_id(),
                    expr: Some(Expr::GraphExecDestroy {
                        id: env.exec_graph_id.clone(), ty: Type::Error,
                        i: Info::default()
                    })
                },
                Stmt::Definition {
                    ty: Type::Error, id: new_err_id(),
                    expr: Some(Expr::GraphExecInstantiate {
                        exec_graph: env.exec_graph_id.clone(),
                        graph: env.graph_id.clone(),
                        ty: Type::Error, i: Info::default()
                    })
                }
            ],
            els: vec![]
        }
    ];

    let graph_not_instantiated = Expr::UnOp {
        op: UnOp::Not,
        arg: Box::new(inst_var),
        ty: Type::Scalar {sz: ElemSize::Bool},
        i: Info::default()
    };
    acc.push(Stmt::If {
        cond: graph_not_instantiated, thn: inst_thn, els: inst_els
    });

    // Destroy the captured graph to avoid leaking memory.
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::GraphDestroy {
            id: env.graph_id.clone(), ty: Type::Error, i: Info::default()
        })
    });

    // Launch the executable graph based on the capturing of the body. Note that capturing is
    // performed on a separate stream.
    acc.push(Stmt::Definition {
        ty: Type::Error, id: new_err_id(),
        expr: Some(Expr::GraphExecLaunch {
            id: env.exec_graph_id.clone(), ty: Type::Error, i: Info::default()
        })
    });
}

fn use_env_stream_expr(stream: &Stream, e: Expr) -> Expr {
    match e {
        Expr::MallocAsync {id, elem_ty, sz, ty, i, ..} => {
            Expr::MallocAsync {id, elem_ty, sz, stream: stream.clone(), ty, i}
        },
        Expr::FreeAsync {id, ty, i, ..} => {
            Expr::FreeAsync {id, stream: stream.clone(), ty, i}
        },
        Expr::StreamBeginCapture {ty, i, ..} => {
            Expr::StreamBeginCapture {stream: stream.clone(), ty, i}
        },
        Expr::StreamEndCapture {graph, ty, i, ..} => {
            Expr::StreamEndCapture {stream: stream.clone(), graph, ty, i}
        },
        _ => e.smap(|e| use_env_stream_expr(stream, e))
    }
}

fn use_env_stream_stmt(stream: &Stream, s: Stmt) -> Stmt {
    match s {
        Stmt::KernelLaunch {id, blocks, threads, args, ..} => {
            Stmt::KernelLaunch {id, blocks, threads, stream: stream.clone(), args}
        },
        _ => {
            s.smap(|s| use_env_stream_stmt(stream, s))
                .smap(|e| use_env_stream_expr(stream, e))
        }
    }
}

fn use_env_stream_stmts(env: &GraphEnv, stmts: Vec<Stmt>) -> Vec<Stmt> {
    let stream = Stream::Id(env.stream_id.clone());
    stmts.smap(|s| use_env_stream_stmt(&stream, s))
}

fn use_cuda_graphs_kernel_body(mut body: Vec<Stmt>) -> (Vec<Top>, Vec<Stmt>) {
    let inst_id = Name::sym_str("instantiated");
    let exec_graph_id = Name::sym_str("exec_graph");
    let graph_id = Name::sym_str("graph");
    let stream_id = Name::sym_str("stream");
    let updated_id = Name::sym_str("updated");
    let env = GraphEnv {
        inst_id, exec_graph_id, graph_id, stream_id, updated_id
    };

    // Generate global variables representing the executable graph and a variable keeping track of
    // whether the executable graph has been instantiated. These are used in the updated function
    // body.
    let bool_ty = Type::Scalar {sz: ElemSize::Bool};
    let added_top_var_defs = vec![
        Top::VarDef {ty: Type::GraphExec, id: env.exec_graph_id.clone(), init: None},
        Top::VarDef {
            ty: bool_ty.clone(), id: env.inst_id.clone(),
            init: Some(Expr::Bool {v: false, ty: bool_ty, i: Info::default()})
        },
    ];

    // Generate an updated function body that includes code for capturing and launching a CUDA
    // graph representing the CUDA operations performed in the function. The purpose of such graphs
    // is to reduce the overhead of repeated CUDA API calls.
    let mut acc_body = vec![];
    generate_pre_kernel_body_statements(&env, &mut acc_body);
    let tail_ret = body.pop().unwrap();
    acc_body.append(&mut use_env_stream_stmts(&env, body));
    generate_post_kernel_body_statements(&env, &mut acc_body);
    acc_body.push(tail_ret);

    (added_top_var_defs, acc_body)
}

fn use_cuda_graphs_top(mut acc: Vec<Top>, t: Top) -> Vec<Top> {
    match t {
        Top::FunDef {dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body} => {
            let (mut tops, body) = use_cuda_graphs_kernel_body(body);
            acc.append(&mut tops);
            acc.push(Top::FunDef {
                dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body
            });
            acc
        },
        _ => {
            acc.push(t);
            acc
        }
    }
}

fn use_cuda_graphs_ast(ast: Ast) -> Ast {
    ast.into_iter().fold(vec![], use_cuda_graphs_top)
}

pub fn use_if_enabled(ast: Ast, opts: &option::CompileOptions) -> Ast {
    if opts.use_cuda_graphs {
        use_cuda_graphs_ast(ast)
    } else {
        ast
    }
}
