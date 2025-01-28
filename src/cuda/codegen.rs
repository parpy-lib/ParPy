use super::ast::*;
use super::free_vars::FreeVariables;
use super::par::{GpuMap, GpuMapping};
use super::pprint;
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
            Ok(Expr::UnOp {op: UnOp::Exp, arg, ty, i})
        },
        ir_ast::Builtin::Inf if args.is_empty() =>
            Ok(Expr::Float {v: f64::INFINITY, ty, i}),
        ir_ast::Builtin::Log if args.len() == 1 => {
            let arg = Box::new(args.remove(0));
            Ok(Expr::UnOp {op: UnOp::Log, arg, ty, i})
        },
        ir_ast::Builtin::Max if args.len() == 2 => {
            let lhs = Box::new(args.remove(0));
            let rhs = Box::new(args.remove(0));
            Ok(Expr::BinOp {lhs, op: BinOp::Max, rhs, ty, i})
        },
        ir_ast::Builtin::Min => {
            let lhs = Box::new(args.remove(0));
            let rhs = Box::new(args.remove(0));
            Ok(Expr::BinOp {lhs, op: BinOp::Min, rhs, ty, i})
        },
        _ => parir_compile_error!(i, "Invalid use of builtin")
    }
}

fn from_ir_unop(op: ir_ast::UnOp) -> UnOp {
    match op {
        ir_ast::UnOp::Sub => UnOp::Sub,
    }
}

fn from_ir_binop(op: ir_ast::BinOp) -> BinOp {
    match op {
        ir_ast::BinOp::Add => BinOp::Add,
        ir_ast::BinOp::Sub => BinOp::Sub,
        ir_ast::BinOp::Mul => BinOp::Mul,
        ir_ast::BinOp::FloorDiv | ir_ast::BinOp::Div => BinOp::Div,
        ir_ast::BinOp::Mod => BinOp::Rem,
        ir_ast::BinOp::BitAnd => BinOp::BitAnd,
        ir_ast::BinOp::Eq => BinOp::Eq,
        ir_ast::BinOp::Neq => BinOp::Neq,
        ir_ast::BinOp::Lt => BinOp::Lt,
        ir_ast::BinOp::Gt => BinOp::Gt,
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
            let op = from_ir_unop(op);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        ir_ast::Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = Box::new(from_ir_expr(*lhs)?);
            let op = from_ir_binop(op);
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
    var: Name,
    lo: ir_ast::Expr,
    hi: ir_ast::Expr,
    par: Option<GpuMap>
) -> CompileResult<(Expr, Expr, Expr)> {
    let init = from_ir_expr(lo)?;
    let ty = init.get_type().clone();
    let i = init.get_info();
    let var_e = Expr::Var {id: var, ty: ty.clone(), i: i.clone()};
    let cond = Expr::BinOp {
        lhs: Box::new(var_e.clone()),
        op: BinOp::Lt,
        rhs: Box::new(from_ir_expr(hi)?),
        ty: Type::Boolean,
        i: i.clone()
    };
    let fn_incr = |v| Expr::BinOp {
        lhs: Box::new(var_e), op: BinOp::Add,
        rhs: Box::new(Expr::Int {
            v, ty: Type::Scalar {sz: ElemSize::I64}, i: i.clone()
        }),
        ty: ty.clone(), i: i.clone()
    };
    match par {
        Some(GpuMap::Thread {n, dim}) => {
            let tot = grid.threads.get_dim(&dim);
            let idx = Expr::ThreadIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
        },
        Some(GpuMap::Block {n, dim}) => {
            let tot = grid.blocks.get_dim(&dim);
            let idx = Expr::BlockIdx {dim, ty: ty.clone(), i: i.clone()};
            let rhs = remainder_if_shared_dimension(idx, tot, n);
            let init = Expr::BinOp {
                lhs: Box::new(init), op: BinOp::Add, rhs: Box::new(rhs),
                ty: ty.clone(), i: i.clone()
            };
            Ok((init, cond, fn_incr(n)))
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
            Ok((init, cond, fn_incr(n)))
        },
        None => Ok((init, cond, fn_incr(1)))
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

fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> Expr {
    let ty = Type::Scalar {sz: sz.clone()};
    match sz {
        ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 =>
            Expr::Int {v: v as i64, ty, i},
        ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => Expr::Float {v, ty, i}
    }
}

fn reduction_op_neutral_element(
    op: &BinOp,
    ty: Type,
    i: Info
) -> CompileResult<Expr> {
    if let Type::Scalar {sz} = &ty {
        match op {
            BinOp::Add => Ok(generate_literal(0.0, sz, i)),
            BinOp::Mul => Ok(generate_literal(1.0, sz, i)),
            BinOp::Max => Ok(generate_literal(f64::NEG_INFINITY, sz, i)),
            BinOp::Min => Ok(generate_literal(f64::INFINITY, sz, i)),
            _ => {
                let op = pprint::print_binop(op, &ty);
                parir_compile_error!(i, "Parallel reductions not supported for operator {op}")
            }
        }
    } else {
        parir_compile_error!(i, "Parallel reductions not supported for non-scalar arguments")
    }
}

fn generate_warp_sync(
    acc: &mut Vec<Stmt>,
    opfn: impl Fn(Expr, Expr) -> Expr,
    temp_var: &Expr,
    i: &Info
) -> () {
    let iter_id = Name::sym_str("i");
    let i64_ty = Type::Scalar {sz: ElemSize::I64};
    let iter_var = Expr::Var {id: iter_id.clone(), ty: i64_ty.clone(), i: i.clone()};
    let rhs = Expr::ShflXorSync {
        value: Box::new(temp_var.clone()),
        idx: Box::new(iter_var.clone()), ty: temp_var.get_type().clone(), i: i.clone(),
    };
    let sync_stmt = Stmt::Assign {
        dst: temp_var.clone(),
        expr: opfn(temp_var.clone(), rhs)
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
    acc.push(Stmt::For {
        var_ty: i64_ty.clone(), var: iter_id, init: int_lit(16),
        cond: cond_expr, incr: incr_expr, body: vec![sync_stmt]
    });
}

fn parallel_reduction_error<T>(i: &Info) -> CompileResult<T> {
    let msg = concat!(
        "Parallel reduction are only supported on for-loops of the form\n",
        "  for i in range(a, b):\n",
        "    x = x + y\n",
        "where x is either a variable or an array access involving i, y is any ",
        "expression, and '+' is a known binary reduction operation.",
    );
    parir_compile_error!(i, "{msg}")
}

fn generate_parallel_reduction(
    var_ty: Type,
    var: Name,
    init: Expr,
    cond: Expr,
    incr: Expr,
    mut body: Vec<ir_ast::Stmt>,
    nthreads: i64,
    i: Info
) -> CompileResult<Stmt> {
    if body.len() == 1 {
        Ok(())
    } else {
        parallel_reduction_error(&i)
    }?;
    let mut acc = Vec::new();
    let fst = body.remove(0);
    let acc = if let ir_ast::Stmt::Assign {dst: dst @ ir_ast::Expr::Var {..}, expr, ..} = fst {
        let dst = from_ir_expr(dst)?;
        let expr = from_ir_expr(expr)?;
        match expr {
            Expr::BinOp {lhs, op, rhs, ty, i} => {
                let ne = reduction_op_neutral_element(&op, ty.clone(), i.clone())?;
                let opfn = |lhs, rhs| Expr::BinOp {
                    lhs: Box::new(lhs), op: op.clone(), rhs: Box::new(rhs),
                    ty: ty.clone(), i: i.clone()
                };
                if dst == *lhs {
                    let temp_id = Name::sym_str("t");
                    let temp_var = Expr::Var {id: temp_id.clone(), ty: ty.clone(), i: i.clone()};
                    // Generate code for a warp-local reduction
                    acc.push(Stmt::Definition {
                        ty: ty.clone(),
                        id: temp_id.clone(),
                        expr: ne
                    });
                    let temp_assign = Stmt::Assign {
                        dst: temp_var.clone(),
                        expr: Expr::BinOp {
                            lhs: Box::new(temp_var.clone()),
                            op: op.clone(), rhs, ty: ty.clone(), i: i.clone()
                        }
                    };
                    acc.push(Stmt::For {
                        var_ty, var, init, cond, incr, body: vec![temp_assign]
                    });
                    generate_warp_sync(&mut acc, opfn, &temp_var.clone(), &i);

                    // If the parallelism includes more than 32 threads, we are using more than one
                    // warp. In this case, we synchronize across all threads of the block.
                    if nthreads > 32 {
                        // Allocate shared memory and write to it
                        let shared_id = Name::sym_str("stemp");
                        let shared_var = Expr::Var {id: shared_id.clone(), ty: ty.clone(), i: i.clone()};
                        acc.push(Stmt::AllocShared {ty: ty.clone(), id: shared_id, sz: 32});
                        let i64_ty = Type::Scalar {sz: ElemSize::I64};
                        let thread_warp_idx = Expr::BinOp {
                            lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: i64_ty.clone(), i: i.clone()}),
                            op: BinOp::Rem,
                            rhs: Box::new(Expr::Int {v: 32, ty: i64_ty.clone(), i: i.clone()}),
                            ty: i64_ty.clone(),
                            i: i.clone()
                        };
                        acc.push(Stmt::If {
                            cond: Expr::BinOp {
                                lhs: Box::new(thread_warp_idx.clone()),
                                op: BinOp::Eq,
                                rhs: Box::new(Expr::Int {v: 0, ty: i64_ty.clone(), i: i.clone()}),
                                ty: Type::Boolean,
                                i: i.clone()
                            },
                            thn: vec![Stmt::Assign {
                                dst: Expr::ArrayAccess {
                                    target: Box::new(shared_var.clone()),
                                    idx: Box::new(Expr::BinOp {
                                        lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: i64_ty.clone(), i: i.clone()}),
                                        op: BinOp::Div,
                                        rhs: Box::new(Expr::Int {v: 32, ty: i64_ty.clone(), i: i.clone()}),
                                        ty: i64_ty.clone(),
                                        i: i.clone()
                                    }),
                                    ty: ty.clone(),
                                    i: i.clone()
                                },
                                expr: temp_var.clone()
                            }],
                            els: vec![]
                        });
                        acc.push(Stmt::Syncthreads {});

                        // Conditionally load data from shared memory, and synchronize across the
                        // threads of the warp.
                        acc.push(Stmt::If {
                            cond: Expr::BinOp {
                                lhs: Box::new(thread_warp_idx.clone()),
                                op: BinOp::Lt,
                                rhs: Box::new(Expr::Int {v: (nthreads + 31) / 32, ty: i64_ty.clone(), i: i.clone()}),
                                ty: Type::Boolean,
                                i: i.clone()
                            },
                            thn: vec![Stmt::Assign {
                                dst: temp_var.clone(),
                                expr: Expr::ArrayAccess {
                                    target: Box::new(shared_var),
                                    idx: Box::new(thread_warp_idx),
                                    ty: ty.clone(),
                                    i: i.clone()
                                }}
                            ],
                            els: vec![Stmt::Assign {
                                dst: temp_var.clone(),
                                expr: Expr::Int {v: 0, ty: i64_ty.clone(), i: i.clone()}
                            }]
                        });
                        generate_warp_sync(&mut acc, opfn, &temp_var.clone(), &i);

                        // Write the result stored in the temporary value to the original target.
                        // If the target is an array access, only the first thread writes to reduce
                        // global array contention.
                        let assign_stmt = Stmt::Assign {
                            dst: dst.clone(),
                            expr: Expr::BinOp {
                                lhs: Box::new(dst.clone()), op,
                                rhs: Box::new(temp_var.clone()),
                                ty: ty.clone(), i: i.clone()
                            },
                        };
                        match dst {
                            Expr::Var {..} => {
                                acc.push(assign_stmt);
                                Ok(())
                            },
                            Expr::ArrayAccess {..} => {
                                let is_first_thread = Expr::BinOp {
                                    lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: i64_ty.clone(), i: i.clone()}),
                                    op: BinOp::Eq,
                                    rhs: Box::new(Expr::Int {v: 0, ty: i64_ty.clone(), i: i.clone()}),
                                    ty: Type::Boolean,
                                    i: i.clone()
                                };
                                acc.push(Stmt::If {
                                    cond: is_first_thread,
                                    thn: vec![assign_stmt],
                                    els: vec![]
                                });
                                Ok(())
                            },
                            _ => parir_compile_error!(i, "Invalid destination expression in binary operation")
                        }?;
                    };
                    Ok(acc)
                } else {
                    parallel_reduction_error(&i)
                }
            },
            _ => parallel_reduction_error(&i)
        }
    } else {
        parallel_reduction_error(&i)
    }?;
    Ok(Stmt::Scope {body: acc})
}

fn generate_kernel_stmt(
    grid: LaunchArgs,
    map: &[GpuMap],
    sync: &BTreeSet<Name>,
    mut acc: Vec<Stmt>,
    stmt: ir_ast::Stmt
) -> CompileResult<Vec<Stmt>> {
    match stmt {
        ir_ast::Stmt::Definition {ty, id, expr, ..} => {
            let ty = from_ir_type(ty);
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Definition {ty, id, expr});
        },
        ir_ast::Stmt::Assign {dst, expr, ..} => {
            let dst = from_ir_expr(dst)?;
            let expr = from_ir_expr(expr)?;
            acc.push(Stmt::Assign {dst, expr});
        },
        ir_ast::Stmt::For {var, lo, hi, body, par, i} => {
            let (p, grid, map) = if par.is_parallel() {
                let m = map[0].clone();
                let grid = subtract_from_grid(grid, &m);
                (Some(m), grid, &map[1..])
            } else {
                (None, grid, map)
            };
            let var_ty = from_ir_type(lo.get_type().clone());
            let (init, cond, incr) = determine_loop_bounds(&grid, var.clone(), lo, hi, p)?;
            if par.is_parallel() && par.reduction {
                acc.push(generate_parallel_reduction(
                    var_ty, var, init, cond, incr, body, par.nthreads, i)?
                );
            } else {
                let body = generate_kernel_stmts(grid, map, sync, vec![], body)?;
                let should_sync = sync.contains(&var);
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body});
                if should_sync {
                    acc.push(Stmt::Syncthreads {});
                };
            };
        },
        ir_ast::Stmt::If {cond, thn, els, ..} => {
            let cond = from_ir_expr(cond)?;
            let thn = generate_kernel_stmts(grid.clone(), map, sync, vec![], thn)?;
            let els = generate_kernel_stmts(grid, map, sync, vec![], els)?;
            acc.push(Stmt::If {cond, thn, els});
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

fn generate_host_kernel_launch(
    id: Name,
    launch_args: LaunchArgs,
    args: Vec<Expr>
) -> Stmt {
    let threads_id = Name::new("threads".to_string()).with_new_sym();
    let blocks_id = Name::new("blocks".to_string()).with_new_sym();
    let body = vec![
        Stmt::Dim3Definition {id: threads_id.clone(), args: launch_args.threads},
        Stmt::Dim3Definition {id: blocks_id.clone(), args: launch_args.blocks},
        Stmt::KernelLaunch {id, blocks: blocks_id, threads: threads_id, args}
    ];
    Stmt::Scope { body }
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
                        body: kernel_body
                    };
                    kernels.push(kernel);
                    let args = fv.into_iter()
                        .map(|(id, ty)| Expr::Var {id, ty, i: i.clone()})
                        .collect::<Vec<Expr>>();
                    host_body.push(generate_host_kernel_launch(kernel_id, m.grid.clone(), args));
                    Ok(kernels)
                },
                None => {
                    let init = from_ir_expr(lo)?;
                    let cond = from_ir_expr(hi)?;
                    let i64_ty = Type::Scalar {sz: ElemSize::I64};
                    let incr = Expr::BinOp {
                        lhs: Box::new(Expr::Var {id: var.clone(), ty: i64_ty.clone(), i: i.clone()}),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::Int {v: 1, ty: i64_ty.clone(), i: i.clone()}),
                        ty: i64_ty, i: i.clone()
                    };
                    let (body, kernels) = from_ir_stmts(env, vec![], kernels, body)?;
                    host_body.push(Stmt::For {
                        var_ty, var, init, cond, incr, body
                    });
                    Ok(kernels)
                }
            }
        },
        ir_ast::Stmt::If {cond, thn, els, ..} => {
            let cond = from_ir_expr(cond)?;
            let (thn, kernels) = from_ir_stmts(env, vec![], kernels, thn)?;
            let (els, kernels) = from_ir_stmts(env, vec![], kernels, els)?;
            host_body.push(Stmt::If {cond, thn, els});
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
                    i: p.i
                };
                params_init.push(Stmt::Definition {
                    ty: p.ty, id: p.id, expr: init_expr
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
    body: Vec<ir_ast::Stmt>
) -> CompileResult<Vec<Top>> {
    let (mut params_init, unwrapped_params) = unwrap_params(&env, params)?;
    let (mut host_body, mut tops) = from_ir_stmts(&env, vec![], vec![], body)?;
    params_init.append(&mut host_body);
    tops.push(Top::FunDef {
        attr: Attribute::Entry, ret_ty: Type::Void, id: env.id,
        params: unwrapped_params, body: params_init
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
    let ir_ast::FunDef {id, params, body, ..} = fun;
    env.id = id;
    let params = params.into_iter()
        .map(|p| from_ir_param(p))
        .collect::<Vec<Param>>();
    from_ir_fun_body(env, params, body)
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
            .collect::<Vec<Field>>()
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
