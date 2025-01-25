use super::ast::*;
use super::free_vars::FreeVariables;
use super::par::{GpuMap, GpuMapping};
use crate::parir_compile_error;
use crate::ir::ast as ir_ast;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
struct CodegenEnv {
    gpu_mapping: BTreeMap<Name, GpuMapping>,
    sync: BTreeSet<Name>,
    struct_fields: BTreeMap<Name, Vec<Field>>,
    id: Name
}

fn from_ir_type(ty: ir_ast::Type) -> Type {
    match ty {
        ir_ast::Type::Boolean => Type::Boolean,
        ir_ast::Type::Tensor {sz, shape} => {
            if shape.len() == 0 {
                Type::Scalar {sz}
            } else {
                Type::Pointer {sz}
            }
        },
        ir_ast::Type::Struct {id} => Type::Struct {id}
    }
}

fn to_builtin(
    func: ir_ast::Builtin,
    args: Vec<ir_ast::Expr>,
    ty: Type,
    i: Info
) -> CompileResult<Expr> {
    let mut args = args.into_iter()
        .map(from_ir_expr)
        .collect::<CompileResult<Vec<Expr>>>()?;
    match func {
        ir_ast::Builtin::Exp if args.len() == 1 => {
            let arg = Box::new(args.remove(0));
            Ok(Expr::Exp {arg, ty, i})
        },
        ir_ast::Builtin::Inf if args.is_empty() => Ok(Expr::Inf {ty, i}),
        ir_ast::Builtin::Log if args.len() == 1 => {
            let arg = Box::new(args.remove(0));
            Ok(Expr::Log {arg, ty, i})
        },
        ir_ast::Builtin::Max if args.len() == 2 => {
            let lhs = Box::new(args.remove(0));
            let rhs = Box::new(args.remove(0));
            Ok(Expr::Max {lhs, rhs, ty, i})
        },
        ir_ast::Builtin::Min => {
            let lhs = Box::new(args.remove(0));
            let rhs = Box::new(args.remove(0));
            Ok(Expr::Min {lhs, rhs, ty, i})
        },
        _ => parir_compile_error!(i, "Invalid use of builtin")
    }
}

fn from_ir_expr(e: ir_ast::Expr) -> CompileResult<Expr> {
    let ty = from_ir_type(e.get_type().clone());
    match e {
        ir_ast::Expr::Var {id, i, ..} => Ok(Expr::Var {id, ty, i}),
        ir_ast::Expr::Int {v, i, ..} => Ok(Expr::Int {v, ty, i}),
        ir_ast::Expr::Float {v, i, ..} => Ok(Expr::Float {v, ty, i}),
        ir_ast::Expr::UnOp {op, arg, i, ..} => {
            let arg = Box::new(from_ir_expr(*arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        ir_ast::Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = Box::new(from_ir_expr(*lhs)?);
            let rhs = Box::new(from_ir_expr(*rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        ir_ast::Expr::StructFieldAccess {target, label, i, ..} => {
            let target = Box::new(from_ir_expr(*target)?);
            Ok(Expr::StructFieldAccess {target, label, ty, i})
        },
        ir_ast::Expr::TensorAccess {target, idx, i, ..} => {
            let target = Box::new(from_ir_expr(*target)?);
            let idx = Box::new(from_ir_expr(*idx)?);
            Ok(Expr::ArrayAccess {target, idx, ty, i})
        },
        ir_ast::Expr::Struct {id, fields, i, ..} => {
            let fields = fields.into_iter()
                .map(|(id, e)| Ok((id, from_ir_expr(e)?)))
                .collect::<CompileResult<Vec<(String, Expr)>>>()?;
            Ok(Expr::Struct {id, fields, ty, i})
        },
        ir_ast::Expr::Builtin {func, args, i, ..} => {
            to_builtin(func, args, ty, i)
        },
        ir_ast::Expr::Convert {e, ..} => {
            let e = Box::new(from_ir_expr(*e)?);
            Ok(Expr::Convert {e, ty})
        },
    }
}

fn generate_kernel_name(fun_id: &Name, loop_var_id: &Name) -> Name {
    let s = format!("{0}_{1}", fun_id.get_str(), loop_var_id.get_str());
    Name::new(s).with_new_sym()
}

fn remainder_if_shared_dimension(idx: Expr, dim_tot: i64, v: i64) -> Expr {
    if dim_tot > v {
        /*let ty = idx.get_type();
        Expr::BinOp {
            lhs: idx,
            op: BinOp::Mod,
            rhs: Expr::Int {v, ty: ty.clone(), i: idx.get_info()},
            i: idx.get_info()
        }*/
        // NOTE: When we have multiple for-loops sharing the same index, we may need to divide the
        // index by an offset to ensure they actually use different "parts" of the index...
        panic!("This case is not handled correctly yet")
    } else {
        idx
    }
}

fn determine_loop_bounds(
    grid: &LaunchArgs,
    lo: ir_ast::Expr,
    hi: ir_ast::Expr,
    par: Option<GpuMap>
) -> CompileResult<(Expr, Expr, i64)> {
    let init = from_ir_expr(lo)?;
    let cond = from_ir_expr(hi)?;
    let ty = init.get_type().clone();
    let i = init.get_info();
    match par {
        Some(GpuMap::Thread {n, dim}) => {
            let tot = grid.threads.get_dim(&dim);
            let idx = Expr::ThreadIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, n))
        },
        Some(GpuMap::Block {n, dim}) => {
            let tot = grid.blocks.get_dim(&dim);
            let idx = Expr::BlockIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, n))
        },
        Some(GpuMap::ThreadBlock {n, nthreads, nblocks, dim}) => {
            let tot_blocks = grid.blocks.get_dim(&dim);
            let idx = Expr::BinOp {
                lhs: Box::new(Expr::BinOp {
                    lhs: Box::new(Expr::BlockIdx {
                        dim, ty: ty.clone(), i: i.clone()
                    }),
                    op: BinOp::Mul,
                    rhs: Box::new(Expr::Int {
                        v: nthreads, ty: ty.clone(), i: i.clone()
                    }),
                    ty: ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Add,
                rhs: Box::new(Expr::ThreadIdx {dim, ty: ty.clone(), i: i.clone()}),
                ty: ty.clone(),
                i: i.clone()
            };
            let rhs = remainder_if_shared_dimension(idx.clone(), tot_blocks, nblocks);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            // If the total number of threads we are using is not evenly divisible by the number of
            // threads per block, we insert an additional condition to ensure only the intended
            // threads run the code inside the for-loop.
            let cond = if nthreads * nblocks != n {
                Expr::BinOp {
                    lhs: Box::new(cond),
                    op: BinOp::BoolAnd,
                    rhs: Box::new(Expr::BinOp {
                        lhs: Box::new(idx.clone()),
                        op: BinOp::Lt,
                        rhs: Box::new(Expr::Int {
                            v: n, ty: ty.clone(), i: i.clone()
                        }),
                        ty: ty.clone(), i: i.clone()
                    }),
                    ty: ty.clone(), i: i.clone()
                }
            } else {
                cond
            };
            Ok((init, cond, n))
        },
        None => Ok((init, cond, 1))
    }
}

fn subtract_from_grid(grid: LaunchArgs, m: &GpuMap) -> LaunchArgs {
    match m {
        GpuMap::Thread {n, dim} => {
            let threads = grid.threads.get_dim(&dim) / n;
            grid.with_threads_dim(dim, threads)
        },
        GpuMap::Block {n, dim} => {
            let blocks = grid.blocks.get_dim(&dim) / n;
            grid.with_blocks_dim(dim, blocks)
        },
        GpuMap::ThreadBlock {nthreads, nblocks, dim, ..} => {
            let threads = grid.threads.get_dim(&dim) / nblocks;
            let blocks = grid.blocks.get_dim(&dim) / nthreads;
            grid.with_blocks_dim(dim, blocks)
                .with_threads_dim(dim, threads)
        }
    }
}

fn generate_kernel_stmt(
    grid: LaunchArgs,
    map: &[GpuMap],
    sync: &BTreeSet<Name>,
    mut acc: Vec<Stmt>,
    stmt: ir_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match stmt {
        ir_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = from_ir_type(ty);
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Definition {ty, id, expr, i});
        },
        ir_ast::Stmt::Assign {dst, expr, i} => {
            let dst = from_ir_expr(dst)?;
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Assign {dst, expr, i});
        },
        ir_ast::Stmt::For {var, lo, hi, body, par, i} => {
            let (par, grid, map) = if par.is_some() {
                let m = map[0].clone();
                let grid = subtract_from_grid(grid, &m);
                (Some(m), grid, &map[1..])
            } else {
                (None, grid, map)
            };
            let var_ty = from_ir_type(lo.get_type().clone());
            let (init, cond, incr) = determine_loop_bounds(&grid, lo, hi, par)?;
            let body = generate_kernel_stmts(grid, map, sync, vec![], body)?;
            let should_sync = sync.contains(&var);
            acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i: i.clone()});
            if should_sync {
                acc.push(Stmt::Syncthreads {i});
            };
        },
        ir_ast::Stmt::If {cond, thn, els, i} => {
            let cond = from_ir_expr(cond)?;
            let thn = generate_kernel_stmts(grid.clone(), map, sync, vec![], thn)?;
            let els = generate_kernel_stmts(grid, map, sync, vec![], els)?;
            acc.push(Stmt::If {cond, thn, els, i});
        }
    };
    Ok(acc)
}

fn generate_kernel_stmts(
    grid: LaunchArgs,
    map: &[GpuMap],
    sync: &BTreeSet<Name>,
    acc: Vec<Stmt>,
    stmts: Vec<ir_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .fold(Ok(acc), |acc, stmt| generate_kernel_stmt(grid.clone(), map, sync, acc?, stmt))
}

fn from_ir_stmt(
    env: &CodegenEnv,
    mut host_body: Vec<Stmt>,
    mut kernels: Vec<Top>,
    stmt: ir_ast::Stmt
) -> CompileResult<(Vec<Stmt>, Vec<Top>)> {
    let kernels = match stmt {
        ir_ast::Stmt::Definition {i, ..} | ir_ast::Stmt::Assign {i, ..} => {
            parir_compile_error!(i, "Assignments are not allowed outside parallel code")
        },
        ir_ast::Stmt::For {var, lo, hi, body, par, i} => {
            let var_ty = from_ir_type(lo.get_type().clone());
            // If this for-loop has been marked for parallelization, we compile its contents to a
            // GPU kernel and insert a kernel launch into the host code. Otherwise, we generate a
            // sequential for-loop and add it to the host code.
            match &env.gpu_mapping.get(&var) {
                Some(m) => {
                    let kernel_id = generate_kernel_name(&env.id, &var);
                    let stmt = ir_ast::Stmt::For {var, lo, hi, body, par, i: i.clone()};
                    let kernel_body = generate_kernel_stmt(
                        m.grid.clone(), &m.get_mapping()[..], &env.sync, vec![], stmt
                    )?;
                    let fv = kernel_body.free_variables();
                    let kernel_params = fv.clone()
                        .into_iter()
                        .map(|(id, ty)| Param {id, ty, i: i.clone()})
                        .collect::<Vec<Param>>();
                    let kernel = Top::FunDef {
                        attr: Attribute::Global, ret_ty: Type::Void,
                        id: kernel_id.clone(), params: kernel_params,
                        body: kernel_body, i: i.clone()
                    };
                    kernels.push(kernel);
                    let args = fv.into_iter()
                        .map(|(id, ty)| Expr::Var {id, ty, i: i.clone()})
                        .collect::<Vec<Expr>>();
                    host_body.push(Stmt::KernelLaunch {
                        id: kernel_id, launch_args: m.grid.clone(), args, i
                    });
                    Ok(kernels)
                },
                None => {
                    let init = from_ir_expr(lo)?;
                    let cond = from_ir_expr(hi)?;
                    let incr = 1;
                    let (body, kernels) = from_ir_stmts(env, vec![], kernels, body)?;
                    host_body.push(Stmt::For {
                        var_ty, var, init, cond, incr, body, i
                    });
                    Ok(kernels)
                }
            }
        },
        ir_ast::Stmt::If {cond, thn, els, i} => {
            let cond = from_ir_expr(cond)?;
            let (thn, kernels) = from_ir_stmts(env, vec![], kernels, thn)?;
            let (els, kernels) = from_ir_stmts(env, vec![], kernels, els)?;
            host_body.push(Stmt::If {cond, thn, els, i});
            Ok(kernels)
        },
    }?;
    Ok((host_body, kernels))
}

fn from_ir_stmts(
    env: &CodegenEnv,
    host_body: Vec<Stmt>,
    kernels: Vec<Top>,
    stmts: Vec<ir_ast::Stmt>
) -> CompileResult<(Vec<Stmt>, Vec<Top>)> {
    stmts.into_iter()
        .fold(Ok((host_body, kernels)), |acc, stmt| {
            let (host_body, kernels) = acc?;
            from_ir_stmt(env, host_body, kernels, stmt)
        })
}

fn unwrap_params(
    env: &CodegenEnv,
    params: Vec<Param>
) -> CompileResult<(Vec<Stmt>, Vec<Param>)> {
    let mut params_init = vec![];
    let mut unwrapped_params = vec![];
    for p in params {
        if let Type::Struct {id} = &p.ty {
            if let Some(fields) = env.struct_fields.get(&id) {
                let field_names = fields.iter()
                    .map(|Field {id, ..}| Name::new(id.clone()).with_new_sym())
                    .collect::<Vec<Name>>();

                let field_exprs = fields.clone()
                    .into_iter()
                    .zip(field_names.clone().into_iter())
                    .map(|(Field {ty, id: label, i}, id)| {
                        (label, Expr::Var {id, ty, i})
                    })
                    .collect::<Vec<(String, Expr)>>();
                let init_expr = Expr::Struct {
                    id: id.clone(), fields: field_exprs, ty: p.ty.clone(),
                    i: p.i.clone()
                };
                params_init.push(Stmt::Definition {
                    ty: p.ty, id: p.id, expr: init_expr, i: p.i
                });

                let mut struct_params = fields.clone()
                    .into_iter()
                    .zip(field_names.into_iter())
                    .map(|(Field {ty, i, ..}, id)| Param {id, ty, i})
                    .collect::<Vec<Param>>();
                unwrapped_params.append(&mut struct_params);
            } else {
                parir_compile_error!(p.i, "Parameter refers to unknown struct type (internal compiler error)")?;
            }
        } else {
            unwrapped_params.push(p);
        }
    }
    Ok((params_init, unwrapped_params))
}

fn from_ir_fun_body(
    env: CodegenEnv,
    params: Vec<Param>,
    body: Vec<ir_ast::Stmt>,
    i: Info
) -> CompileResult<Vec<Top>> {
    let (mut params_init, unwrapped_params) = unwrap_params(&env, params)?;
    let (mut host_body, mut tops) = from_ir_stmts(&env, vec![], vec![], body)?;
    params_init.append(&mut host_body);
    tops.push(Top::FunDef {
        attr: Attribute::Entry, ret_ty: Type::Void, id: env.id,
        params: unwrapped_params, body: params_init, i
    });
    Ok(tops)
}

fn from_ir_param(p: ir_ast::Param) -> Param {
    let ir_ast::Param {id, ty, i} = p;
    Param {id, ty: from_ir_type(ty), i}
}

fn from_ir_fun_def(
    mut env: CodegenEnv,
    fun: ir_ast::FunDef
) -> CompileResult<Vec<Top>> {
    let ir_ast::FunDef {id, params, body, i} = fun;
    env.id = id;
    let params = params.into_iter()
        .map(|p| from_ir_param(p))
        .collect::<Vec<Param>>();
    from_ir_fun_body(env, params, body, i)
}

fn from_ir_field(f: ir_ast::Field) -> Field {
    let ir_ast::Field {id, ty, i} = f;
    Field {id, ty: from_ir_type(ty), i}
}

fn from_ir_struct(s: ir_ast::StructDef) -> Top {
    Top::StructDef {
        id: s.id,
        fields: s.fields.into_iter()
            .map(|f| from_ir_field(f))
            .collect::<Vec<Field>>(),
        i: s.i
    }
}

fn struct_top_fields(t: Top) -> (Name, Vec<Field>) {
    if let Top::StructDef {id, fields, ..} = t {
        (id, fields)
    } else {
        unreachable!()
    }
}

pub fn from_ir(
    ast: ir_ast::Ast,
    gpu_mapping: BTreeMap<Name, GpuMapping>,
    sync: BTreeSet<Name>
) -> CompileResult<Ast> {
    let mut structs = ast.structs
        .into_iter()
        .map(from_ir_struct)
        .collect::<Vec<Top>>();

    let struct_fields = structs.clone()
        .into_iter()
        .map(struct_top_fields)
        .collect::<BTreeMap<Name, Vec<Field>>>();
    let env = CodegenEnv {
        gpu_mapping, sync, struct_fields, id: Name::new(String::new())
    };

    let mut tops = vec![
        Top::Include {header: "<stdint.h>".to_string()}
    ];
    tops.append(&mut structs);
    tops.append(&mut from_ir_fun_def(env, ast.fun)?);
    Ok(tops)
}
