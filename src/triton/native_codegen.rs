use super::native_ast::*;
use crate::parpy_internal_error;
use crate::gpu::ast as gpu_ast;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

#[derive(Clone, Debug)]
struct CodegenEnv {
    // Name of the stream parameter.
    stream_id: Name,

    // Name of the shared memory parameter, which is a pointer to integer values. each
    // integer represents the amount of shared memory used in a kernel.
    smem_id: Name,

    // Name of the argument count parameter, which is a pointer to integer values. Each integer
    // represents the number of arguments a GPU kernel expects.
    argc_id: Name,

    // Name of the cache path parameter, referring to the directory where the cached files
    // belonging to this particular file is stored.
    cache_path_id: Name,

    // Tracks the number of kernel launches visited so far. This is used to assign each kernel
    // launch a distinct integer identifier we use for indexing into the shared memory and argument
    // counter pointers.
    kernel_count: usize,
}

impl Default for CodegenEnv {
    fn default() -> Self {
        CodegenEnv {
            stream_id: Name::sym_str("stream"),
            smem_id: Name::sym_str("smem"),
            argc_id: Name::sym_str("argc"),
            cache_path_id: Name::sym_str("cache_path"),
            kernel_count: 0
        }
    }
}

fn get_include_headers() -> Vec<Top> {
    vec![
        Top::Include {header: "\"parpy_triton.h\"".to_string()}
    ]
}

fn from_gpu_ast_type(ty: &gpu_ast::Type) -> Type {
    match ty {
        gpu_ast::Type::Void => Type::Void,
        gpu_ast::Type::Scalar {sz} => Type::Scalar {sz: sz.clone()},
        gpu_ast::Type::Pointer {ty, mem: _} => {
            let ty = from_gpu_ast_type(ty);
            Type::Pointer {ty: Box::new(ty)}
        },
        gpu_ast::Type::Function {result, args} => {
            let result = from_gpu_ast_type(result);
            let args = args.iter()
                .map(from_gpu_ast_type)
                .collect::<Vec<Type>>();
            Type::Function {result: Box::new(result), args}
        }
    }
}

fn from_gpu_ast_expr(e: gpu_ast::Expr) -> CompileResult<Expr> {
    let ty = from_gpu_ast_type(e.get_type());
    match e {
        gpu_ast::Expr::Var {id, ty: _, i} => Ok(Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, ty: _, i} => Ok(Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, ty: _, i} => Ok(Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, ty: _, i} => Ok(Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, ty: _, i} => {
            let arg = Box::new(from_gpu_ast_expr(*arg)?);
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} => {
            let lhs = Box::new(from_gpu_ast_expr(*lhs)?);
            let rhs = Box::new(from_gpu_ast_expr(*rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::Assign {lhs, rhs, ty: _, i} => {
            let dst = Box::new(from_gpu_ast_expr(*lhs)?);
            let expr = Box::new(from_gpu_ast_expr(*rhs)?);
            Ok(Expr::Assign {dst, expr, ty, i})
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, ty: _, i} => {
            let cond = Box::new(from_gpu_ast_expr(*cond)?);
            let thn = Box::new(from_gpu_ast_expr(*thn)?);
            let els = Box::new(from_gpu_ast_expr(*els)?);
            Ok(Expr::Ternary {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i} => {
            let target = Box::new(from_gpu_ast_expr(*target)?);
            let idx = Box::new(from_gpu_ast_expr(*idx)?);
            Ok(Expr::ArrayAccess {target, idx, ty, i})
        },
        gpu_ast::Expr::Call {id, args, ty: _, i} => {
            let args = args.into_iter()
                .map(from_gpu_ast_expr)
                .collect::<CompileResult<Vec<Expr>>>()?;
            Ok(Expr::Call {id, args, ty, i})
        },
        gpu_ast::Expr::PyCallback {i, ..} => {
            parpy_internal_error!(i, "Found Python callback node in native Triton codegen.")
        },
        gpu_ast::Expr::Convert {e, ty: _} => {
            let i = e.get_info();
            let e = Box::new(from_gpu_ast_expr(*e)?);
            Ok(Expr::Convert {e, ty, i})
        },
        gpu_ast::Expr::ThreadIdx {i, ..} => {
            parpy_internal_error!(i, "Found thread index in native Triton codegen.")
        },
        gpu_ast::Expr::BlockIdx {i, ..} => {
            parpy_internal_error!(i, "Found block index in native Triton codegen.")
        },
    }
}

fn generate_kernel_launch(
    mut env: CodegenEnv,
    id: Name,
    args: Vec<Expr>,
    grid: LaunchArgs,
    i: Info
) -> CompileResult<(CodegenEnv, Stmt)> {
    let (arg_defs, arg_ids) = args.into_iter()
         .map(|arg| {
             let temp_id = Name::sym_str("t");
             let def = Stmt::Definition {
                 ty: arg.get_type().clone(),
                 dst: temp_id.clone(),
                 expr: Some(arg)
             };
             (def, temp_id)
         })
         .collect::<(Vec<Stmt>, Vec<Name>)>();
    let args_id = Name::sym_str("args");
    let argc_expr = Expr::ArrayAccess {
        target: Box::new(Expr::Var {
            id: env.argc_id.clone(),
            ty: Type::Pointer {ty: Box::new(Type::Scalar {sz: ElemSize::I32})},
            i: i.clone()
        }),
        idx: Box::new(Expr::Int {
            v: env.kernel_count as i128,
            ty: Type::Scalar {sz: ElemSize::I32},
            i: i.clone()
        }),
        ty: Type::Scalar {sz: ElemSize::I32},
        i: i.clone()
    };
    let args_ty = Type::Array {
        ty: Box::new(Type::Pointer {ty: Box::new(Type::Void)}),
        sz: Box::new(argc_expr.clone()),
    };
    let args_decl = Stmt::Definition {
        ty: args_ty.clone(),
        dst: args_id.clone(),
        expr: None
    };
    let args_access = |idx: Expr| Expr::ArrayAccess {
        target: Box::new(Expr::Var {
            id: args_id.clone(),
            ty: args_ty.clone(),
            i: i.clone()
        }),
        idx: Box::new(idx),
        ty: Type::Pointer {ty: Box::new(Type::Void)},
        i: i.clone()
    };
    let args_assigns = arg_ids.into_iter()
        .enumerate()
        .map(|(idx, arg_id)| Stmt::Expr {
            e: Expr::Assign {
                dst: Box::new(args_access(Expr::Int {
                    v: idx as i128,
                    ty: Type::Scalar {sz: ElemSize::I32},
                    i: i.clone()
                })),
                expr: Box::new(Expr::UnOp {
                    op: UnOp::Addressof,
                    arg: Box::new(Expr::Var {
                        id: arg_id,
                        ty: Type::Void,
                        i: i.clone()
                    }),
                    ty: Type::Pointer {ty: Box::new(Type::Void)},
                    i: i.clone()
                }),
                ty: Type::Pointer {ty: Box::new(Type::Void)},
                i: i.clone()
            }
        })
        .collect::<Vec<Stmt>>();
    let var = Name::sym_str("i");
    let dummy_id = Name::sym_str("dummy");
    let dummy_init_stmt = Stmt::Definition {
        ty: Type::Scalar {sz: ElemSize::I32},
        dst: dummy_id.clone(),
        expr: None
    };
    let fill_remaining_arg_slots_stmt = Stmt::For {
        var_ty: Type::Scalar {sz: ElemSize::I32},
        var: var.clone(),
        init: Expr::Int {
            v: args_assigns.len() as i128,
            ty: Type::Scalar {sz: ElemSize::I32},
            i: i.clone()
        },
        cond: Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: var.clone(),
                ty: Type::Scalar {sz: ElemSize::I32},
                i: i.clone()
            }),
            op: BinOp::Lt,
            rhs: Box::new(argc_expr),
            ty: Type::Scalar {sz: ElemSize::I32},
            i: i.clone()
        },
        incr: Expr::BinOp {
            lhs: Box::new(Expr::Var {
                id: var.clone(),
                ty: Type::Scalar {sz: ElemSize::I32},
                i: i.clone()
            }),
            op: BinOp::Add,
            rhs: Box::new(Expr::Int {
                v: 1,
                ty: Type::Scalar {sz: ElemSize::I32},
                i: i.clone()
            }),
            ty: Type::Scalar {sz: ElemSize::I32},
            i: i.clone()
        },
        body: vec![Stmt::Expr {
            e: Expr::Assign {
                dst: Box::new(args_access(Expr::Var {
                    id: var,
                    ty: Type::Scalar {sz: ElemSize::I32},
                    i: i.clone()
                })),
                expr: Box::new(Expr::UnOp {
                    op: UnOp::Addressof,
                    arg: Box::new(Expr::Var {
                        id: dummy_id.clone(),
                        ty: Type::Scalar {sz: ElemSize::I32},
                        i: i.clone()
                    }),
                    ty: Type::Pointer {ty: Box::new(Type::Void)},
                    i: i.clone()
                }),
                ty: Type::Pointer {ty: Box::new(Type::Void)},
                i: i.clone()
            }
        }]
    };
    let smem = Expr::ArrayAccess {
        target: Box::new(Expr::Var {
            id: env.smem_id.clone(),
            ty: Type::Pointer {ty: Box::new(Type::Scalar {sz: ElemSize::I32})},
            i: i.clone()
        }),
        idx: Box::new(Expr::Int {
            v: env.kernel_count as i128,
            ty: Type::Scalar {sz: ElemSize::I32},
            i: i.clone()
        }),
        ty: Type::Scalar {sz: ElemSize::I32},
        i: i.clone()
    };
    let stream = Expr::Var {
        id: env.stream_id.clone(),
        ty: Type::CudaStream,
        i: i.clone()
    };
    let args = Expr::Var {
        id: args_id,
        ty: args_ty,
        i: i.clone()
    };
    let launch_stmt = Stmt::CheckError {
        e: Expr::LaunchKernel {
            id,
            grid,
            smem: Box::new(smem),
            stream: Box::new(stream),
            args: Box::new(args),
            ty: Type::CudaResult,
            i
        }
    };
    let body = arg_defs.into_iter()
        .chain(vec![args_decl].into_iter())
        .chain(args_assigns.into_iter())
        .chain(vec![dummy_init_stmt, fill_remaining_arg_slots_stmt, launch_stmt].into_iter())
        .collect::<Vec<Stmt>>();
    env.kernel_count += 1;
    Ok((env, Stmt::Scope {body}))
}

fn from_gpu_ast_stmt(
    env: CodegenEnv,
    s: gpu_ast::Stmt
) -> CompileResult<(CodegenEnv, Stmt)> {
    match s {
        gpu_ast::Stmt::Definition {ty, id, expr, i: _} => {
            let ty = from_gpu_ast_type(&ty);
            let expr = from_gpu_ast_expr(expr)?;
            Ok((env, Stmt::Definition {ty, dst: id, expr: Some(expr)}))
        },
        gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, unroll: _, i: _} => {
            let var_ty = from_gpu_ast_type(&var_ty);
            let init = from_gpu_ast_expr(init)?;
            let cond = from_gpu_ast_expr(cond)?;
            let incr = from_gpu_ast_expr(incr)?;
            let (env, body) = from_gpu_ast_stmts(env, body)?;
            Ok((env, Stmt::For {var_ty, var, init, cond, incr, body}))
        },
        gpu_ast::Stmt::If {cond, thn, els, i: _} => {
            let cond = from_gpu_ast_expr(cond)?;
            let (env, thn) = from_gpu_ast_stmts(env, thn)?;
            let (env, els) = from_gpu_ast_stmts(env, els)?;
            Ok((env, Stmt::If {cond, thn, els}))
        },
        gpu_ast::Stmt::While {cond, body, i: _} => {
            let cond = from_gpu_ast_expr(cond)?;
            let (env, body) = from_gpu_ast_stmts(env, body)?;
            Ok((env, Stmt::While {cond, body}))
        },
        gpu_ast::Stmt::Return {value, i: _} => {
            let e = from_gpu_ast_expr(value)?;
            Ok((env, Stmt::Return {e}))
        },
        gpu_ast::Stmt::Scope {body, i: _} => {
            let (env, body) = from_gpu_ast_stmts(env, body)?;
            Ok((env, Stmt::Scope {body}))
        },
        gpu_ast::Stmt::Expr {e, i: _} => {
            let e = from_gpu_ast_expr(e)?;
            Ok((env, Stmt::Expr {e}))
        },
        gpu_ast::Stmt::KernelLaunch {id, args, grid, smem: _, i} => {
            let args = args.into_iter()
                .map(from_gpu_ast_expr)
                .collect::<CompileResult<Vec<Expr>>>()?;
            generate_kernel_launch(env, id, args, grid, i)
        },
        gpu_ast::Stmt::AllocDevice {elem_ty: _, id, sz, i} => {
            let alloc_expr = Expr::AllocDevice {
                id,
                sz,
                stream: env.stream_id.clone(),
                ty: Type::CudaResult,
                i
            };
            Ok((env, Stmt::CheckError {e: alloc_expr}))
        },
        gpu_ast::Stmt::FreeDevice {id, i} => {
            let free_expr = Expr::FreeDevice {
                id,
                stream: env.stream_id.clone(),
                ty: Type::CudaResult,
                i
            };
            Ok((env, Stmt::CheckError {e: free_expr}))
        },
        gpu_ast::Stmt::ParallelReduction {ref i, ..} |
        gpu_ast::Stmt::Synchronize {ref i, ..} |
        gpu_ast::Stmt::WarpReduce {ref i, ..} |
        gpu_ast::Stmt::ClusterReduce {ref i, ..} |
        gpu_ast::Stmt::AllocShared {ref i, ..} |
        gpu_ast::Stmt::CopyMemory {ref i, ..} => {
            parpy_internal_error!(i, "Found unsupported GPU statement {s:?} in \
                                      native Triton codegen.")
        },
    }
}

fn from_gpu_ast_stmts(
    env: CodegenEnv,
    stmts: Vec<gpu_ast::Stmt>
) -> CompileResult<(CodegenEnv, Vec<Stmt>)> {
    stmts.into_iter()
        .fold(Ok((env, vec![])), |acc, s| {
            let (env, mut stmts) = acc?;
            let (env, s) = from_gpu_ast_stmt(env, s)?;
            stmts.push(s);
            Ok((env, stmts))
        })
}

fn from_gpu_ast_param(p: &gpu_ast::Param) -> Param {
    let gpu_ast::Param {id, ty, ..} = p;
    Param {id: id.clone(), ty: from_gpu_ast_type(ty)}
}

fn from_gpu_ast_params(params: &Vec<gpu_ast::Param>) -> Vec<Param> {
    params.iter()
        .map(from_gpu_ast_param)
        .collect::<Vec<Param>>()
}

fn add_auxiliary_parameters(
    env: CodegenEnv,
    mut params: Vec<Param>
) -> Vec<Param> {
    params.append(&mut vec![
        Param {id: env.stream_id, ty: Type::CudaStream},
        Param {id: env.smem_id, ty: Type::Pointer {ty: Box::new(Type::Scalar {sz: ElemSize::I32})}},
        Param {id: env.argc_id, ty: Type::Pointer {ty: Box::new(Type::Scalar {sz: ElemSize::I32})}},
        Param {id: env.cache_path_id, ty: Type::String},
    ]);
    params
}

fn from_gpu_ast_top(
    env: &CodegenEnv,
    acc: (Vec<Top>, Vec<Top>),
    t: &gpu_ast::Top
) -> CompileResult<(Vec<Top>, Vec<Top>)> {
    let (mut includes, mut tops) = acc;
    match t {
        gpu_ast::Top::KernelFunDef {..} |
        gpu_ast::Top::ExtDecl {target: gpu_ast::Target::Device, ..} |
        gpu_ast::Top::FunDef {target: gpu_ast::Target::Device, ..} => {},
        gpu_ast::Top::ExtDecl {ret_ty: _, id, ext_id, params, header, target: _, i: _} => {
            let params = from_gpu_ast_params(params);
            if let Some(h) = header {
                includes.push(Top::Include {header: h.clone()});
            };
            tops.push(Top::ExtDecl {id: id.clone(), ext_id: ext_id.clone(), params});
        },
        gpu_ast::Top::FunDef {ret_ty, id, params, body, target: _, i: _} => {
            let ret_ty = from_gpu_ast_type(ret_ty);
            let params = from_gpu_ast_params(params);
            let env = env.clone();
            let (env, body) = from_gpu_ast_stmts(env, body.clone())?;
            let params = add_auxiliary_parameters(env, params);
            tops.push(Top::FunDef {ret_ty, id: id.clone(), params, body});
        },
    };
    Ok((includes, tops))
}

fn collect_launched_kernels_expr(mut acc: Vec<Name>, e: &Expr) -> Vec<Name> {
    match e {
        Expr::LaunchKernel {id, ..} => {
            acc.push(id.clone());
            acc
        },
        _ => e.sfold(acc, collect_launched_kernels_expr)
    }
}

fn collect_launched_kernels_stmt(acc: Vec<Name>, s: &Stmt) -> Vec<Name> {
    let acc = s.sfold(acc, collect_launched_kernels_stmt);
    s.sfold(acc, collect_launched_kernels_expr)
}

fn generate_kernel_initialization_top(
    env: &CodegenEnv,
    acc: (Vec<Top>, Vec<Top>),
    t: Top
) -> (Vec<Top>, Vec<Top>) {
    let (mut kernel_inits, mut tops) = acc;
    match t {
        Top::FunDef {ret_ty, id, params, mut body} => {
            let kernel_ids = body.sfold(vec![], collect_launched_kernels_stmt);
            let mut kernel_func_decls = kernel_ids.iter()
                .map(|id| Top::VarDef {
                    ty: Type::CudaFunction,
                    id: id.clone(),
                    value: None
                })
                .collect::<Vec<Top>>();
            kernel_inits.append(&mut kernel_func_decls);
            let init_id = Name::sym_str("init");
            let bool_ty = Type::Scalar {sz: ElemSize::Bool};
            let init_def = Top::VarDef {
                ty: bool_ty.clone(),
                id: init_id.clone(),
                value: Some(Expr::Bool {
                    v: false, ty: bool_ty.clone(), i: Info::default()
                })
            };
            kernel_inits.push(init_def);
            let kernel_load_stmts = kernel_ids.into_iter()
                .map(|id| vec![
                    Stmt::Expr {
                        e: Expr::Assign {
                            dst: Box::new(Expr::Var {
                                id: id.clone(),
                                ty: Type::CudaFunction,
                                i: Info::default()
                            }),
                            expr: Box::new(Expr::LoadKernel {
                                path: env.cache_path_id.clone(),
                                id: id.clone(),
                                ty: Type::CudaFunction,
                                i: Info::default()
                            }),
                            ty: Type::CudaFunction,
                            i: Info::default()
                        }
                    },
                    Stmt::CheckNonNull {
                        e: Expr::Var {
                            id: id,
                            ty: Type::CudaFunction,
                            i: Info::default()
                        }
                    }
                ]);
            let assign_init_true = Stmt::Expr {
                e: Expr::Assign {
                    dst: Box::new(Expr::Var {
                        id: init_id.clone(),
                        ty: bool_ty.clone(),
                        i: Info::default()
                    }),
                    expr: Box::new(Expr::Bool {
                        v: true, ty: bool_ty.clone(), i: Info::default()
                    }),
                    ty: bool_ty.clone(),
                    i: Info::default()
                }
            };
            let cond_stmts = kernel_load_stmts.flatten()
                .chain(vec![assign_init_true].into_iter())
                .collect::<Vec<Stmt>>();
            let init_cond = Expr::UnOp {
                op: UnOp::Not,
                arg: Box::new(Expr::Var {
                    id: init_id,
                    ty: bool_ty.clone(),
                    i: Info::default()
                }),
                ty: bool_ty.clone(),
                i: Info::default()
            };
            let if_stmt = Stmt::If {cond: init_cond, thn: cond_stmts, els: vec![]};
            body.insert(0, if_stmt);
            tops.push(Top::FunDef {ret_ty, id, params, body});
        },
        _ => tops.push(t),
    }
    (kernel_inits, tops)
}

fn generate_kernel_initialization(
    env: &CodegenEnv,
    tops: Vec<Top>
) -> (Vec<Top>, Vec<Top>) {
    tops.into_iter()
        .fold((vec![], vec![]), |acc, t| {
            generate_kernel_initialization_top(&env, acc, t)
        })
}

pub fn from_gpu_ast(ast: &gpu_ast::Ast) -> CompileResult<Ast> {
    let env = CodegenEnv::default();
    let includes = get_include_headers();
    let (includes, tops) = ast.sfold_result(Ok((includes, vec![])), |acc, t| {
        from_gpu_ast_top(&env, acc, t)
    })?;
    let (kernel_init_tops, tops) = generate_kernel_initialization(&env, tops);
    let stmts = includes.into_iter()
        .chain(kernel_init_tops.into_iter())
        .chain(tops.into_iter())
        .collect::<Ast>();
    Ok(stmts)
}
