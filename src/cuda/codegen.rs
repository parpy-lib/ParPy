use super::ast::*;
use crate::option;
use crate::prickle_internal_error;
use crate::prickle_type_error;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::pprint::PrettyPrint;

fn from_gpu_ir_type(ty: gpu_ast::Type) -> Type {
    match ty {
        gpu_ast::Type::Void => Type::Void,
        gpu_ast::Type::Boolean => Type::Boolean,
        gpu_ast::Type::Scalar {sz} => Type::Scalar {sz},
        gpu_ast::Type::Pointer {ty, ..} => {
            Type::Pointer {ty: Box::new(from_gpu_ir_type(*ty))}
        },
        gpu_ast::Type::Struct {id} => Type::Struct {id},
    }
}

fn validate_unary_operation(
    op: &UnOp, ty: &Type, i: &Info
) -> CompileResult<()> {
    match op {
        UnOp::Tanh => match ty.get_scalar_elem_size() {
            Some(ElemSize::F32 | ElemSize::F64) => Ok(()),
            Some(ElemSize::F16) => {
                prickle_type_error!(i, "Operation tanh not supported for \
                                      16-bit floats.")
            },
            Some(_) | None => {
                let ty = ty.pprint_default();
                prickle_type_error!(i, "Unexpected type {ty} of tanh \
                                      builtin (expected float).")
            }
        },
        _ => Ok(())
    }
}

fn validate_binary_operation(
    op: &BinOp, ty: &Type, i: &Info
) -> CompileResult<()> {
    match op {
        BinOp::Atan2 => match ty.get_scalar_elem_size() {
            Some(ElemSize::F64) => Ok(()),
            Some(ElemSize::F16 | ElemSize::F32) => {
                prickle_type_error!(i, "Operation atan2 is only supported \
                                      for 64-bit floats.")
            },
            Some(_) | None => {
                let ty = ty.pprint_default();
                prickle_type_error!(i, "Unexpected type {ty} of atan2 \
                                      builtin (expected float).")
            }
        },
        _ => Ok(())
    }
}

fn from_gpu_ir_expr(e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ir_type(e.get_type().clone());
    match e {
        gpu_ast::Expr::Var {id, i, ..} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, i, ..} => Ok(Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, i, ..} => Ok(Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, i, ..} => Ok(Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, i, ..} => {
            let arg = from_gpu_ir_expr(*arg)?;
            validate_unary_operation(&op, &ty, &i)?;
            Ok(Expr::UnOp {op, arg: Box::new(arg), ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = from_gpu_ir_expr(*lhs)?;
            let rhs = from_gpu_ir_expr(*rhs)?;
            validate_binary_operation(&op, &ty, &i)?;
            Ok(Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i})
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, i, ..} => {
            let cond = from_gpu_ir_expr(*cond)?;
            let thn = from_gpu_ir_expr(*thn)?;
            let els = from_gpu_ir_expr(*els)?;
            Ok(Expr::Ternary {
                cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
            })
        },
        gpu_ast::Expr::StructFieldAccess {target, label, i, ..} => {
            let target = from_gpu_ir_expr(*target)?;
            Ok(Expr::StructFieldAccess {target: Box::new(target), label, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {target, idx, i, ..} => {
            let target = from_gpu_ir_expr(*target)?;
            let idx = from_gpu_ir_expr(*idx)?;
            Ok(Expr::ArrayAccess {target: Box::new(target), idx: Box::new(idx), ty, i})
        },
        gpu_ast::Expr::Call {id, args, i, ..} => {
            let args = args.into_iter()
                .map(from_gpu_ir_expr)
                .collect::<CompileResult<Vec<Expr>>>()?;
            Ok(Expr::Call {id, args, ty, i})
        },
        gpu_ast::Expr::Convert {e, ..} => {
            let e = from_gpu_ir_expr(*e)?;
            Ok(Expr::Convert {e: Box::new(e), ty})
        },
        gpu_ast::Expr::Struct {id, fields, i, ..} => {
            let fields = fields.into_iter()
                .map(|(s, e)| Ok((s, from_gpu_ir_expr(e)?)))
                .collect::<CompileResult<Vec<(String, Expr)>>>()?;
            Ok(Expr::Struct {id, fields, ty, i})
        },
        gpu_ast::Expr::ThreadIdx {dim, i, ..} => Ok(Expr::ThreadIdx {dim, ty, i}),
        gpu_ast::Expr::BlockIdx {dim, i, ..} => Ok(Expr::BlockIdx {dim, ty, i}),
    }
}

fn from_gpu_ir_stmt(s: gpu_ast::Stmt) -> CompileResult<Stmt> {
    match s {
        gpu_ast::Stmt::Definition {ty, id, expr, ..} => {
            let ty = from_gpu_ir_type(ty);
            let expr = from_gpu_ir_expr(expr)?;
            Ok(Stmt::Definition {ty, id, expr})
        },
        gpu_ast::Stmt::Assign {dst, expr, ..} => {
            let dst = from_gpu_ir_expr(dst)?;
            let expr = from_gpu_ir_expr(expr)?;
            Ok(Stmt::Assign {dst, expr})
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, ..} => {
            let var_ty = from_gpu_ir_type(var_ty);
            let init = from_gpu_ir_expr(init)?;
            let cond = from_gpu_ir_expr(cond)?;
            let incr = from_gpu_ir_expr(incr)?;
            let body = from_gpu_ir_stmts(body)?;
            Ok(Stmt::For {var_ty, var, init, cond, incr, body})
        },
        gpu_ast::Stmt::If {cond, thn, els, ..} => {
            let cond = from_gpu_ir_expr(cond)?;
            let thn = from_gpu_ir_stmts(thn)?;
            let els = from_gpu_ir_stmts(els)?;
            Ok(Stmt::If {cond, thn, els})
        },
        gpu_ast::Stmt::While {cond, body, ..} => {
            let cond = from_gpu_ir_expr(cond)?;
            let body = from_gpu_ir_stmts(body)?;
            Ok(Stmt::While {cond, body})
        },
        gpu_ast::Stmt::Return {value, ..} => {
            let value = from_gpu_ir_expr(value)?;
            Ok(Stmt::Return {value})
        },
        gpu_ast::Stmt::Scope {i, ..} => {
            prickle_internal_error!(i, "Found scope statement that should have \
                                        been eliminated.")
        },
        gpu_ast::Stmt::ParallelReduction {i, ..} => {
            prickle_internal_error!(i, "Found parallel reduction statement \
                                        that should have been eliminated.")
        },
        gpu_ast::Stmt::Synchronize {scope, ..} => Ok(Stmt::Synchronize {scope}),
        gpu_ast::Stmt::WarpReduce {i, ..} => {
            prickle_internal_error!(i, "Found warp reduction statement that \
                                        should have been eliminated.")
        },
        gpu_ast::Stmt::ClusterReduce {i, ..} => {
            prickle_internal_error!(i, "Found cluster reduction statement that \
                                        should have been eliminated.")
        },
        gpu_ast::Stmt::KernelLaunch {id, args, grid, ..} => {
            let args = args.into_iter()
                .map(from_gpu_ir_expr)
                .collect::<CompileResult<Vec<Expr>>>()?;
            let gpu_ast::LaunchArgs {blocks, threads} = grid;
            Ok(Stmt::KernelLaunch {id, blocks, threads, args})
        },
        gpu_ast::Stmt::AllocDevice {elem_ty, id, sz, ..} => {
            let ty = from_gpu_ir_type(elem_ty);
            Ok(Stmt::MallocAsync {id, elem_ty: ty, sz})
        },
        gpu_ast::Stmt::AllocShared {elem_ty, id, sz, ..} => {
            let ty = from_gpu_ir_type(elem_ty);
            Ok(Stmt::AllocShared {ty, id, sz})
        },
        gpu_ast::Stmt::FreeDevice {id, ..} => {
            Ok(Stmt::FreeAsync {id})
        },
        gpu_ast::Stmt::CopyMemory {i, ..} => {
            prickle_internal_error!(i, "Memory copying not supported in CUDA backend")
        },
    }
}

fn from_gpu_ir_stmts(stmts: Vec<gpu_ast::Stmt>) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .map(from_gpu_ir_stmt)
        .collect::<CompileResult<Vec<Stmt>>>()
}

fn from_gpu_ir_param(p: gpu_ast::Param) -> Param {
    let gpu_ast::Param {id, ty, ..} = p;
    Param {id, ty: from_gpu_ir_type(ty)}
}

fn from_gpu_ir_field(f: gpu_ast::Field) -> Field {
    let gpu_ast::Field {id, ty, ..} = f;
    Field {id, ty: from_gpu_ir_type(ty)}
}

fn from_gpu_ir_attr(attr: gpu_ast::KernelAttribute) -> KernelAttribute {
    match attr {
        gpu_ast::KernelAttribute::LaunchBounds {threads} => {
            KernelAttribute::LaunchBounds {threads}
        },
        gpu_ast::KernelAttribute::ClusterDims {dims} => {
            KernelAttribute::ClusterDims {dims}
        },
    }
}

fn from_gpu_ir_top(t: gpu_ast::Top) -> CompileResult<Top> {
    match t {
        gpu_ast::Top::KernelFunDef {attrs, id, params, body} => {
            let attrs = attrs.into_iter()
                .map(from_gpu_ir_attr)
                .collect::<Vec<KernelAttribute>>();
            let params = params.into_iter()
                .map(from_gpu_ir_param)
                .collect::<Vec<Param>>();
            let body = from_gpu_ir_stmts(body)?;
            Ok(Top::FunDef {
                dev_attr: Attribute::Global, ret_ty: Type::Void,
                attrs, id, params, body
            })
        },
        gpu_ast::Top::FunDef {ret_ty, id, params, body, target} => {
            let ret_ty = from_gpu_ir_type(ret_ty);
            let params = params.into_iter()
                .map(from_gpu_ir_param)
                .collect::<Vec<Param>>();
            let body = from_gpu_ir_stmts(body)?;
            let dev_attr = match target {
                gpu_ast::Target::Host => Attribute::Entry,
                gpu_ast::Target::Device => Attribute::Device,
            };
            Ok(Top::FunDef {
                dev_attr, ret_ty, attrs: vec![],
                id, params, body
            })
        },
        gpu_ast::Top::StructDef {id, fields} => {
            let fields = fields.into_iter()
                .map(from_gpu_ir_field)
                .collect::<Vec<Field>>();
            Ok(Top::StructDef {id, fields})
        },
    }
}

pub fn from_gpu_ir(
    ast: gpu_ast::Ast,
    opts: &option::CompileOptions
) -> CompileResult<Ast> {
    let mut tops = vec![
        Top::Include {header: "<cmath>".to_string()},
        Top::Include {header: "<cstdint>".to_string()},
        Top::Include {header: "<cuda_fp16.h>".to_string()},
    ];
    if opts.use_cuda_thread_block_clusters {
        tops.push(Top::Include {header: "<cooperative_groups.h>".to_string()});
        tops.push(Top::Namespace {ns: "cooperative_groups".to_string(), alias: None});
    }
    let mut cu_ast = ast.into_iter()
        .map(from_gpu_ir_top)
        .collect::<CompileResult<Vec<Top>>>()?;
    tops.append(&mut cu_ast);
    Ok(tops)
}
