// Transforms the code by ensuring scalar arguments are passed as buffers. In particular, we
// transform kernel-side code to use pointers for scalar arguments, where all uses perform proper
// dereferencing. On the host-side, we allocate temporary buffers for each scalar argument passed
// in a kernel launch and deallocate them at the end of the host call.

use crate::gpu::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
struct TempBuffer {
    buf_id: Name,
    src_id: Name,
    elem_ty: Type,
    i: Info
}
type BufferEnv = BTreeMap<Name, TempBuffer>;

fn insert_temporary_buffers_arg(mut acc: BufferEnv, arg: Expr) -> (BufferEnv, Expr) {
    match arg {
        Expr::Var {id, ty, i} if ty.is_scalar() => {
            let buf_ty = Type::Pointer {ty: Box::new(ty.clone()), mem: MemSpace::Device};
            let o = acc.get(&id);
            match o {
                Some(TempBuffer {buf_id, ..}) => {
                    let new_arg = Expr::Var {id: buf_id.clone(), ty: buf_ty, i};
                    (acc, new_arg)
                },
                None => {
                    let buf_id = id.clone().with_new_sym();
                    let buf = TempBuffer {
                        buf_id: buf_id.clone(),
                        src_id: id.clone(),
                        elem_ty: ty.clone(),
                        i: i.clone()
                    };
                    acc.insert(id, buf);
                    (acc, Expr::Var {id: buf_id, ty: buf_ty, i})
                }
            }
        },
        _ => (acc, arg)
    }
}

fn alloc_temporary_buffer(buf: &TempBuffer) -> Vec<Stmt> {
    let TempBuffer {buf_id, src_id, elem_ty, i} = buf;
    let buf_ty = Type::Pointer {ty: Box::new(elem_ty.clone()), mem: MemSpace::Device};
    let def = Stmt::Definition {
        ty: buf_ty.clone(), id: buf_id.clone(),
        expr: Expr::Int {v: 0, ty: buf_ty.clone(), i: i.clone()},
        i: i.clone()
    };
    let alloc = Stmt::AllocDevice {
        elem_ty: elem_ty.clone(), id: buf_id.clone(), sz: 1, i: i.clone()
    };
    let copy = Stmt::CopyMemory {
        elem_ty: elem_ty.clone(),
        src: Expr::UnOp {
            op: UnOp::Addressof,
            arg: Box::new(Expr::Var {
                id: src_id.clone(), ty: elem_ty.clone(), i: i.clone()
            }),
            ty: buf_ty.clone(),
            i: i.clone()
        },
        src_mem: MemSpace::Host,
        dst: Expr::Var {id: buf_id.clone(), ty: buf_ty.clone(), i: i.clone()},
        dst_mem: MemSpace::Device,
        sz: 1,
        i: i.clone()
    };
    vec![def, alloc, copy]
}

fn free_temporary_buffer(buf: &TempBuffer) -> Stmt {
    let TempBuffer {buf_id, i, ..} = buf;
    Stmt::FreeDevice { id: buf_id.clone(), i: i.clone() }
}

fn with_temporary_buffer_management(acc: BufferEnv, s: Stmt) -> Vec<Stmt> {
    let alloc = acc.values().flat_map(alloc_temporary_buffer);
    let free = acc.values().map(free_temporary_buffer);
    alloc.into_iter()
        .chain(vec![s].into_iter())
        .chain(free.into_iter())
        .collect::<Vec<Stmt>>()
}

fn insert_temporary_buffers_stmt(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::KernelLaunch {id, args, grid, i} => {
            let (bufs, args) = args.smap_accum_l(BTreeMap::new(), insert_temporary_buffers_arg);
            let launch_stmt = Stmt::KernelLaunch {id, args, grid, i};
            let mut stmts = with_temporary_buffer_management(bufs, launch_stmt);
            acc.append(&mut stmts);
            acc
        },
        _ => s.sflatten(acc, insert_temporary_buffers_stmt),
    }
}

fn insert_temporary_buffers_host_body(body: Vec<Stmt>) -> Vec<Stmt> {
    body.sflatten(vec![], insert_temporary_buffers_stmt)
}

fn convert_scalar_param_to_pointer(
    mut conv_params: Vec<Name>, p: Param
) -> (Vec<Name>, Param) {
    let Param {id, ty, i} = p;
    let (conv_params, ty) = match &ty {
        _ if ty.is_scalar() => {
            conv_params.push(id.clone());
            let ty = Type::Pointer {ty: Box::new(ty.clone()), mem: MemSpace::Device};
            (conv_params, ty)
        },
        _ => (conv_params, ty)
    };
    (conv_params, Param {id, ty, i})
}

fn update_use_of_converted_scalar_params_expr(conv: &BTreeSet<Name>, e: Expr) -> Expr {
    match e {
        Expr::Var {ref id, ref i, ..} if conv.contains(&id) => {
            let i = i.clone();
            let ptr_ty = Type::Pointer {
                ty: Box::new(e.get_type().clone()),
                mem: MemSpace::Device
            };
            Expr::ArrayAccess {
                target: Box::new(e),
                idx: Box::new(Expr::Int {
                    v: 0, ty: Type::Scalar {sz: ElemSize::I32}, i: i.clone()
                }),
                ty: ptr_ty,
                i
            }
        },
        _ => e.smap(|e| update_use_of_converted_scalar_params_expr(conv, e))
    }
}

fn update_use_of_converted_scalar_params_stmt(conv: &BTreeSet<Name>, s: Stmt) -> Stmt {
    s.smap(|s| update_use_of_converted_scalar_params_stmt(conv, s))
        .smap(|e| update_use_of_converted_scalar_params_expr(conv, e))
}

fn transform_scalars_to_buffers_top(t: Top) -> Top {
    match t {
        Top::KernelFunDef {attrs, id, params, body} => {
            let (conv, params) = params.smap_accum_l(vec![], convert_scalar_param_to_pointer);
            let conv = conv.into_iter().collect::<BTreeSet<Name>>();
            let body = body.smap(|s| update_use_of_converted_scalar_params_stmt(&conv, s));
            Top::KernelFunDef {attrs, id, params, body}
        },
        Top::FunDef {ret_ty, id, params, body, target: Target::Host} => {
            let body = insert_temporary_buffers_host_body(body);
            Top::FunDef {ret_ty, id, params, body, target: Target::Host}
        },
        Top::FunDef {target: Target::Device, ..} => t,
        Top::StructDef {..} => t,
    }
}

pub fn transform_scalars_to_buffers(ast: Ast) -> Ast {
    ast.smap(transform_scalars_to_buffers_top)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gpu::ast_builder::*;

    #[test]
    fn insert_temp_buffer_scalar_var() {
        let env = BTreeMap::new();
        let e = var("x", scalar(ElemSize::I32));
        let (env, e) = insert_temporary_buffers_arg(env, e);
        assert_eq!(env.len(), 1);
        let TempBuffer {buf_id, ..} = env.get(&id("x")).unwrap();
        let buf_ty = Type::Pointer {ty: Box::new(scalar(ElemSize::I32)), mem: MemSpace::Device};
        let expected = Expr::Var {id: buf_id.clone(), ty: buf_ty, i: i()};
        assert_eq!(e, expected);
    }

    #[test]
    fn insert_existing_temp_buffer_var() {
        let mut env = BTreeMap::new();
        let buf_id = id("x").with_new_sym();
        let buf = TempBuffer {
            buf_id: buf_id.clone(),
            src_id: id("x"),
            elem_ty: scalar(ElemSize::F32),
            i: i()
        };
        env.insert(id("x"), buf);
        let e = var("x", scalar(ElemSize::F32));
        let (env_post, e) = insert_temporary_buffers_arg(env.clone(), e);
        assert_eq!(env, env_post);
        let buf_ty = Type::Pointer {ty: Box::new(scalar(ElemSize::I32)), mem: MemSpace::Device};
        let expected = Expr::Var {id: buf_id, ty: buf_ty, i: i()};
        assert_eq!(e, expected);
    }

    #[test]
    fn free_temp_buffer() {
        let buf = TempBuffer {
            buf_id: id("y"),
            src_id: id("x"),
            elem_ty: scalar(ElemSize::F32),
            i: i()
        };
        let s = Stmt::FreeDevice {id: id("y"), i: i()};
        assert_eq!(free_temporary_buffer(&buf), s);
    }

    #[test]
    fn convert_scalar_param() {
        let p = Param {id: id("x"), ty: scalar(ElemSize::F32), i: i()};
        let (conv, p) = convert_scalar_param_to_pointer(vec![], p);
        assert_eq!(conv, vec![id("x")]);
        let ty = Type::Pointer {ty: Box::new(scalar(ElemSize::F32)), mem: MemSpace::Device};
        assert_eq!(p, Param {id: id("x"), ty, i: i()});
    }

    #[test]
    fn convert_non_scalar_param() {
        let ty = Type::Pointer {ty: Box::new(scalar(ElemSize::F32)), mem: MemSpace::Device};
        let p = Param {id: id("x"), ty, i: i()};
        let (conv, p_post) = convert_scalar_param_to_pointer(vec![], p.clone());
        assert_eq!(conv, vec![]);
        assert_eq!(p_post, p);
    }
}
