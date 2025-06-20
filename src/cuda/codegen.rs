use super::ast::*;
use crate::parir_compile_error;
use crate::parir_internal_error;
use crate::parir_type_error;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
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
                parir_type_error!(i, "Operation tanh not supported for \
                                      16-bit floats.")
            },
            Some(_) | None => {
                let ty = ty.pprint_default();
                parir_type_error!(i, "Unexpected type {ty} of tanh \
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
                parir_type_error!(i, "Operation atan2 is only supported \
                                      for 64-bit floats.")
            },
            Some(_) | None => {
                let ty = ty.pprint_default();
                parir_type_error!(i, "Unexpected type {ty} of atan2 \
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
        gpu_ast::Stmt::Scope {i, ..} => {
            parir_compile_error!(i, "Internal error: Found scope statement that \
                                     should have been eliminated")
        },
        gpu_ast::Stmt::SynchronizeBlock {..} => Ok(Stmt::Syncthreads {}),
        gpu_ast::Stmt::WarpReduce {value, op, ty, i} => {
            let iter_id = Name::sym_str("i");
            let i64_ty = Type::Scalar {sz: ElemSize::I64};
            let iter_var = Expr::Var {id: iter_id.clone(), ty: i64_ty.clone(), i: i.clone()};
            let value = from_gpu_ir_expr(value)?;
            let ty = from_gpu_ir_type(ty);
            let rhs = Expr::ShflXorSync {
                value: Box::new(value.clone()),
                idx: Box::new(iter_var.clone()), ty: value.get_type().clone(), i: i.clone(),
            };
            let sync_stmt = Stmt::Assign {
                dst: value.clone(),
                expr: Expr::BinOp {
                    lhs: Box::new(value),
                    op: op,
                    rhs: Box::new(rhs),
                    ty: ty,
                    i: i.clone()
                }
            };
            let int_lit = |v| {
                Expr::Int {v, ty: i64_ty.clone(), i: i.clone()}
            };
            let cond_expr = Expr::BinOp {
                lhs: Box::new(iter_var.clone()),
                op: BinOp::Gt,
                rhs: Box::new(int_lit(0)),
                ty: Type::Boolean,
                i: i.clone()
            };
            let incr_expr = Expr::BinOp {
                lhs: Box::new(iter_var),
                op: BinOp::Div,
                rhs: Box::new(int_lit(2)),
                ty: i64_ty.clone(),
                i: i.clone()
            };
            Ok(Stmt::For {
                var_ty: i64_ty.clone(), var: iter_id, init: int_lit(16),
                cond: cond_expr, incr: incr_expr, body: vec![sync_stmt]
            })
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
            parir_internal_error!(i, "Memory copying not supported in CUDA backend")
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

fn from_gpu_ir_top(t: gpu_ast::Top) -> CompileResult<Top> {
    match t {
        gpu_ast::Top::DeviceFunDef {threads, id, params, body} => {
            let params = params.into_iter()
                .map(from_gpu_ir_param)
                .collect::<Vec<Param>>();
            let body = from_gpu_ir_stmts(body)?;
            Ok(Top::FunDef {
                attr: Attribute::Global, ret_ty: Type::Void,
                bounds_attr: Some(threads), id, params, body
            })
        },
        gpu_ast::Top::HostFunDef {ret_ty, id, params, body} => {
            let ret_ty = from_gpu_ir_type(ret_ty);
            let params = params.into_iter()
                .map(from_gpu_ir_param)
                .collect::<Vec<Param>>();
            let body = from_gpu_ir_stmts(body)?;
            Ok(Top::FunDef {
                attr: Attribute::Entry, ret_ty, bounds_attr: None,
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

pub fn from_gpu_ir(ast: gpu_ast::Ast) -> CompileResult<Ast> {
    let mut tops = vec![
        Top::Include {header: "<cmath>".to_string()},
        Top::Include {header: "<cstdint>".to_string()},
        Top::Include {header: "<cuda_fp16.h>".to_string()},
    ];
    let mut cu_ast = ast.into_iter()
        .map(from_gpu_ir_top)
        .collect::<CompileResult<Vec<Top>>>()?;
    tops.append(&mut cu_ast);
    Ok(tops)
}
