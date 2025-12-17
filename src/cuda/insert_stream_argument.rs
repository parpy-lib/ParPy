use super::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

fn use_stream_argument_expr(stream_id: &Name, e: Expr) -> Expr {
    let stream = Stream::Id(stream_id.clone());
    match e {
        Expr::MallocAsync {id, elem_ty, sz, stream: _, ty, i} => {
            Expr::MallocAsync {id, elem_ty, sz, stream, ty, i}
        },
        Expr::FreeAsync {id, stream: _, ty, i} => {
            Expr::FreeAsync {id, stream, ty, i}
        },
        _ => e.smap(|e| use_stream_argument_expr(&stream_id, e))
    }
}

fn use_stream_argument_stmt(stream_id: &Name, s: Stmt) -> Stmt {
    match s {
        Stmt::KernelLaunch {id, blocks, threads, smem, stream: _, args} => {
            let stream = Stream::Id(stream_id.clone());
            Stmt::KernelLaunch {id, blocks, threads, smem, stream, args}
        },
        _ => {
            s.smap(|s| use_stream_argument_stmt(&stream_id, s))
                .smap(|e| use_stream_argument_expr(&stream_id, e))
        }
    }
}

fn apply_top(t: Top) -> Top {
    match t {
        Top::FunDef {
            dev_attr: dev_attr @ Attribute::Entry, ret_ty, attrs, id,
            mut params, body
        } => {
            let stream_param_id = Name::sym_str("stream_param");
            // Updates any kernel launch or asynchronous memory operation to make it use the
            // provided stream instead of the default stream.
            let body = body.smap(|s| use_stream_argument_stmt(&stream_param_id, s));

            // Add the stream parameter to the generated AST. This parameter is provided
            // automatically by the wrapper function we use, so users of ParPy never have to
            // explicitly provide it.
            params.push(Param {
                id: stream_param_id,
                ty: Type::Stream
            });
            Top::FunDef {
                dev_attr, ret_ty, attrs, id, params, body
            }
        },
        _ => t
    }
}

pub fn apply(ast: Ast) -> Ast {
    ast.smap(apply_top)
}
