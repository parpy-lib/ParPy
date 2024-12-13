use crate::ir::ast as ir_ast;
use crate::codegen::ast::*;
use crate::par::ParKind;

use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use rand::distributions::{Alphanumeric, DistString};

#[derive(Clone, Debug)]
pub struct CodegenError {
    msg : String
}

impl CodegenError {
    fn new(msg : String) -> Self {
        CodegenError {msg}
    }
}

impl error::Error for CodegenError {}
impl fmt::Display for CodegenError {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        let msg = &self.msg;
        write!(f, "Parir codegen error: {msg}")
    }
}

pub type CodegenResult<T> = Result<T, CodegenError>;

macro_rules! codegen_error {
    ($($t:tt)*) => {{
        Err(CodegenError::new(format!($($t)*)))
    }};
}

impl From<CodegenError> for PyErr {
    fn from(value : CodegenError) -> PyErr {
        PyRuntimeError::new_err(value.msg)
    }
}

//////////////////////////////
// PARALLEL CODE GENERATION //
//////////////////////////////

const DEFAULT_TPB : i64 = 256;

struct ParFunDef {
    id : String,
    params : Vec<TypedParam>,
    par : HashMap<String, Vec<ParKind>>
}

fn gen_kernel_id(
    ast : &Ast,
    fun_id : &str
) -> String {
    let kernel_ids = ast.kernels.iter()
        .map(|DeviceTop::KernelFunDef {id, ..}| id)
        .collect::<HashSet<&String>>();
    let rand_str = Alphanumeric.sample_string(&mut rand::thread_rng(), 8);
    let mut id = format!("{fun_id}{rand_str}");
    while kernel_ids.contains(&id) {
        let rand_str = Alphanumeric.sample_string(&mut rand::thread_rng(), 8);
        id = format!("{fun_id}{rand_str}");
    };
    id
}

fn find_inner_parallelism_stmt(
    stmt : &Stmt,
    par : &HashMap<String, Vec<ParKind>>
) -> CodegenResult<Option<i64>> {
    match stmt {
        Stmt::For {var, body, ..} => {
            let inner_threads = find_inner_parallelism_stmts(body, par)?;
            let empty = vec![];
            let local_par = match par.get(var).unwrap_or(&empty)[..] {
                [ParKind::GpuThreads(nthreads)] => Ok(Some(nthreads)),
                [ParKind::GpuBlocks(_)] => codegen_error!("Nested GpuBlocks are not supported"),
                [] => Ok(None),
                _ => codegen_error!("Unsupported parallelization arguments: {par:?}")
            }?;
            match (inner_threads, local_par) {
                (Some(a), Some(b)) if a == b => Ok(Some(a)),
                (Some(_), Some(_)) => codegen_error!("Variable amount of parallelism within parallel loops is not supported"),
                (Some(a), None) => Ok(Some(a)),
                (None, Some(b)) => Ok(Some(b)),
                _ => Ok(None)
            }
        },
        _ => Ok(None)
    }
}

fn find_inner_parallelism_stmts(
    stmts : &Vec<Stmt>,
    par : &HashMap<String, Vec<ParKind>>
) -> CodegenResult<Option<i64>> {
    stmts.iter()
        .fold(Ok(None), |acc, stmt| {
            if let Some(nthreads) = find_inner_parallelism_stmt(stmt, par)? {
                match acc? {
                    Some(n) => {
                        if n == nthreads {
                            Ok(Some(n))
                        } else {
                            codegen_error!("Variable amount of parallelism within parallel loops is not supported")
                        }
                    },
                    None => Ok(Some(nthreads))
                }
            } else {
                acc
            }
        })
}

fn codegen_kernel<'a>(
    ast : Ast,
    var : &'a String,
    init : &'a Expr,
    cond : &'a Expr,
    body : Vec<Stmt>,
    def : &'a ParFunDef,
    par_kind : &'a ParKind
) -> CodegenResult<(Ast, Stmt)> {
    let kernel_id = gen_kernel_id(&ast, &def.id);
    let kernel_args = def.params.clone()
        .into_iter()
        .map(|TypedParam {id, ty}| match ty {
            Type::IntTensor(_) | Type::FloatTensor(_) => {
                let elem_ty = match ty {
                    Type::IntTensor(sz) => Type::Int(sz),
                    Type::FloatTensor(sz) => Type::Float(sz),
                    _ => panic!("Impossible case")
                };
                Expr::Call {
                    target : Box::new(Expr::BinOp {
                        lhs : Box::new(Expr::Var {id, ty}),
                        op : BinOp::Proj,
                        rhs : Box::new(Expr::Str {v : "data_ptr".to_string()})
                    }),
                    ty_args : vec![elem_ty],
                    args : vec![]
                }
            },
            _ => Expr::Var {id, ty}
        })
        .collect::<Vec<Expr>>();
    let inner_threads = find_inner_parallelism_stmts(&body, &def.par)?.unwrap_or(1);
    let (tpb, xblocks, yblocks) = match par_kind {
        ParKind::GpuBlocks(nblocks) =>
            if inner_threads <= 1024 {
                Ok((inner_threads, 1, *nblocks))
            } else {
                let tpb = DEFAULT_TPB;
                let xblocks = (inner_threads + tpb - 1) / tpb;
                Ok((tpb, xblocks, *nblocks))
            },
        ParKind::GpuThreads(nthreads) =>
            if inner_threads != 1 {
                codegen_error!("Nested parallelism inside GpuThreads is not supported")
            } else {
                if *nthreads <= 1024 {
                    Ok((*nthreads, 1, 1))
                } else {
                    let tpb = DEFAULT_TPB;
                    let xblocks = (*nthreads + tpb - 1) / tpb;
                    Ok((tpb, xblocks, 1))
                }
            }
    }?;
    let kernel_launch = Stmt::KernelLaunch {
        threads : (tpb, 1, 1),
        blocks : (xblocks, yblocks, 1),
        id : kernel_id.clone(),
        args : kernel_args
    };
    let (mut ast, stmts) = codegen_stmts(ast, body, def, true)?;
    let (dev_init, dev_incr) = match par_kind {
        ParKind::GpuBlocks(nblocks) => {
            let dev_init = Expr::BinOp {
                lhs : Box::new(init.clone()),
                op : BinOp::Add,
                rhs : Box::new(Expr::BlockIdx(Dim::Y))
            };
            let dev_incr = Expr::Int {v : *nblocks, ty : Type::Int(IntSize::I64)};
            (dev_init, dev_incr)
        },
        ParKind::GpuThreads(_) => {
            let rhs = if xblocks == 1 {
                Expr::ThreadIdx(Dim::X)
            } else {
                Expr::BinOp {
                    lhs : Box::new(Expr::BinOp {
                        lhs : Box::new(Expr::BlockIdx(Dim::X)),
                        op : BinOp::Mul,
                        rhs : Box::new(Expr::Int {v : xblocks, ty : Type::Int(IntSize::I64)})
                    }),
                    op : BinOp::Add,
                    rhs : Box::new(Expr::ThreadIdx(Dim::X))
                }
            };
            let dev_init = Expr::BinOp {
                lhs : Box::new(init.clone()),
                op : BinOp::Add,
                rhs : Box::new(rhs)
            };
            let incr = tpb * xblocks;
            let dev_incr = Expr::Int {v : incr, ty : Type::Int(IntSize::I64)};
            (dev_init, dev_incr)
        }
    };
    let kernel_body = Stmt::For {
        var_ty : Type::Int(IntSize::I64),
        var : var.clone(),
        init : dev_init,
        cond : cond.clone(),
        incr : dev_incr,
        body : stmts
    };
    let kernel_def = DeviceTop::KernelFunDef {
        id : kernel_id,
        params : def.params.clone(),
        body : vec![kernel_body]
    };
    ast.kernels.push(kernel_def);
    Ok((ast, kernel_launch))
}

fn codegen_stmt<'a>(
    ast : Ast,
    stmt : Stmt,
    def : &'a ParFunDef,
    device : bool
) -> CodegenResult<(Ast, Stmt)> {
    match &stmt {
        Stmt::For {var_ty, var, init, cond, body, ..} => {
            let empty_vec = vec![];
            let par = def.par.get(var).unwrap_or(&empty_vec);
            match &par[..] {
                [] => Ok((ast, stmt)),
                [pk @ ParKind::GpuBlocks(_)] => {
                    if device {
                        codegen_error!("Nested GpuBlock parallelism is not supported")
                    } else {
                        codegen_kernel(ast, var, init, cond, body.clone(), def, pk)
                    }
                },
                [pk @ ParKind::GpuThreads(nthreads)] => {
                    if device {
                        // If we are already generating device code, we produce a for-loop running
                        // in parallel over the threads.
                        let rhs = if *nthreads >= 1024 {
                            let tpb = DEFAULT_TPB;
                            let nblocks = (*nthreads + tpb - 1) / tpb;
                            Expr::BinOp {
                                lhs : Box::new(Expr::BinOp {
                                    lhs : Box::new(Expr::Int {v : nblocks, ty : Type::Int(IntSize::I64)}),
                                    op : BinOp::Mul,
                                    rhs : Box::new(Expr::BlockIdx(Dim::X))
                                }),
                                op : BinOp::Add,
                                rhs : Box::new(Expr::ThreadIdx(Dim::X))
                            }
                        } else {
                            Expr::ThreadIdx(Dim::X)
                        };
                        let init_expr = Expr::BinOp {
                            lhs : Box::new(init.clone()),
                            op : BinOp::Add,
                            rhs : Box::new(rhs)
                        };
                        let (ast, stmts) = codegen_stmts(ast, body.clone(), def, true)?;
                        let stmt = Stmt::For {
                            var_ty : var_ty.clone(),
                            var : var.clone(),
                            init : init_expr,
                            cond : cond.clone(),
                            incr : Expr::Int {v : *nthreads, ty : Type::Int(IntSize::I64)},
                            body : stmts
                        };
                        Ok((ast, stmt))
                    } else {
                        codegen_kernel(ast, var, init, cond, body.clone(), def, pk)
                    }
                },
                _ => codegen_error!("Unsupported parallelism argument: {par:?}")
            }
        },
        Stmt::KernelLaunch {..} => {
            codegen_error!("Found kernel launch in host code generation")
        },
        _ => Ok((ast, stmt))
    }
}

fn codegen_stmts(
    ast : Ast,
    stmts : Vec<Stmt>,
    def : &ParFunDef,
    device : bool
) -> CodegenResult<(Ast, Vec<Stmt>)> {
    stmts.into_iter()
        .fold(Ok((ast, vec![])), |acc, stmt| {
            let (ast, mut stmts) = acc?;
            let (ast, stmt) = codegen_stmt(ast, stmt, &def, device)?;
            stmts.push(stmt);
            Ok((ast, stmts))
        })
}

fn add_host_entry_functions(
    mut host_entry : Vec<HostTop>,
    def : &ParFunDef
) -> Vec<HostTop> {
    let fun_decl = HostTop::FunDecl {
        id : def.id.clone(),
        params : def.params.clone()
    };
    host_entry.push(fun_decl);
    let mut body = def.params.iter()
        .filter_map(|TypedParam {id, ty}| match ty {
            Type::IntTensor(_) | Type::FloatTensor(_) => Some(Stmt::Expr {
                e : Expr::Call {
                    target : Box::new(Expr::Str {v : "CHECK_INPUT".to_string() }),
                    ty_args : vec![],
                    args : vec![Expr::Str {v : id.clone()}]
                }
            }),
            _ => None
        })
        .collect::<Vec<Stmt>>();
    let args = def.params.clone()
        .into_iter()
        .map(|TypedParam {id, ty}| Expr::Var {id, ty})
        .collect::<Vec<Expr>>();
    body.push(Stmt::Expr {e : Expr::Call {
        target : Box::new(Expr::Str {v : def.id.clone()}),
        ty_args : vec![],
        args
    }});
    let entry_def = HostTop::FunDef {
        id : format!("{0}_entry", def.id),
        params : def.params.clone(),
        body : body
    };
    host_entry.push(entry_def);
    host_entry
}

fn add_host_stage_function(
    mut host_stage : Vec<HostTop>,
    body : Vec<Stmt>,
    def : ParFunDef
) -> Vec<HostTop> {
    let fun_def = HostTop::FunDef {
        id : def.id,
        params : def.params,
        body : body
    };
    host_stage.push(fun_def);
    host_stage
}

fn instantiate_parallel_function(
    ast : Ast,
    body : Vec<Stmt>,
    def : ParFunDef,
) -> CodegenResult<Ast> {
    let (mut ast, body) = codegen_stmts(ast, body, &def, false)?;
    ast.host_entry = add_host_entry_functions(ast.host_entry, &def);
    ast.host_stage = add_host_stage_function(ast.host_stage, body, def);
    Ok(ast)
}

//////////////////////////////////////
// IR -> LOW-LEVEL CODE TRANSLATION //
//////////////////////////////////////

fn compile_ir_int_size(sz : ir_ast::IntSize) -> IntSize {
    match sz {
        ir_ast::IntSize::I8 => IntSize::I8,
        ir_ast::IntSize::I16 => IntSize::I16,
        ir_ast::IntSize::I32 => IntSize::I32,
        ir_ast::IntSize::I64 => IntSize::I64,
        ir_ast::IntSize::Any => IntSize::I64
    }
}

fn compile_ir_float_size(sz : ir_ast::FloatSize) -> FloatSize {
    match sz {
        ir_ast::FloatSize::F16 => FloatSize::F16,
        ir_ast::FloatSize::F32 => FloatSize::F32,
        ir_ast::FloatSize::F64 => FloatSize::F64,
        ir_ast::FloatSize::Any => FloatSize::F32
    }
}

fn compile_ir_type(
    ty : ir_ast::Type
) -> CodegenResult<Type> {
    match ty {
        ir_ast::Type::Int(sz) => {
            let sz = compile_ir_int_size(sz);
            Ok(Type::Int(sz))
        },
        ir_ast::Type::Float(sz) => {
            let sz = compile_ir_float_size(sz);
            Ok(Type::Float(sz))
        },
        ir_ast::Type::IntTensor(sz) => {
            let sz = compile_ir_int_size(sz);
            Ok(Type::IntTensor(sz))
        },
        ir_ast::Type::FloatTensor(sz) => {
            let sz = compile_ir_float_size(sz);
            Ok(Type::FloatTensor(sz))
        },
        ir_ast::Type::Unknown => {
            codegen_error!("Found unknown type in IR AST")
        }
    }
}

fn compile_ir_binop(
    bop : ir_ast::BinOp
) -> BinOp {
    match bop {
        ir_ast::BinOp::Add => BinOp::Add,
        ir_ast::BinOp::Sub => BinOp::Sub,
        ir_ast::BinOp::Mul => BinOp::Mul
    }
}

fn compile_ir_expr(
    e : ir_ast::Expr
) -> CodegenResult<Expr> {
    match e {
        ir_ast::Expr::Var {id, ty} => {
            let ty = compile_ir_type(ty)?;
            Ok(Expr::Var {id, ty})
        },
        ir_ast::Expr::Int {v, ty} => {
            let ty = compile_ir_type(ty)?;
            Ok(Expr::Int {v, ty})
        },
        ir_ast::Expr::Float {v, ty} => {
            let ty = compile_ir_type(ty)?;
            Ok(Expr::Float {v, ty})
        },
        ir_ast::Expr::BinOp {lhs, op, rhs, ..} => {
            let lhs = Box::new(compile_ir_expr(*lhs)?);
            let op = compile_ir_binop(op);
            let rhs = Box::new(compile_ir_expr(*rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs})
        },
        ir_ast::Expr::Subscript {target, idx, ..} => {
            let target = Box::new(compile_ir_expr(*target)?);
            let idx = Box::new(compile_ir_expr(*idx)?);
            Ok(Expr::Subscript {target, idx})
        }
    }
}

fn compile_ir_stmt(
    mut def_vars : HashSet<String>,
    stmt : ir_ast::Stmt
) -> CodegenResult<(HashSet<String>, Stmt)> {
    match stmt {
        ir_ast::Stmt::Assign {dst, e} => {
            let e = compile_ir_expr(e)?;
            let dst = compile_ir_expr(dst)?;
            match &dst {
                Expr::Var {id, ty} => {
                    if def_vars.contains(id) {
                        Ok((def_vars, Stmt::Assign {dst : dst.clone(), e}))
                    } else {
                        def_vars.insert(id.clone());
                        Ok((def_vars, Stmt::Defn {ty : ty.clone(), id : id.clone(), e}))
                    }
                },
                _ => Ok((def_vars, Stmt::Assign {dst, e}))
            }
        },
        ir_ast::Stmt::For {var, lo, hi, body} => {
            let var_ty = Type::Int(IntSize::I64);
            let init = compile_ir_expr(lo)?;
            let cond = compile_ir_expr(hi)?;
            let incr = Expr::Int {v : 1, ty : var_ty.clone()};
            let (def_vars, body) = compile_ir_stmts(def_vars, body)?;
            Ok((def_vars, Stmt::For {
                var_ty, var, init, cond, incr, body
            }))
        }
    }
}

fn compile_ir_stmts(
    def_vars : HashSet<String>,
    stmts : Vec<ir_ast::Stmt>
) -> CodegenResult<(HashSet<String>, Vec<Stmt>)> {
    stmts.into_iter()
        .fold(Ok((def_vars, vec![])), |acc, stmt| {
            let (def_vars, mut stmts) = acc?;
            let (def_vars, stmt) = compile_ir_stmt(def_vars, stmt)?;
            stmts.push(stmt);
            Ok((def_vars, stmts))
        })
}

fn compile_ir_params(
    def_vars : HashSet<String>,
    params : Vec<ir_ast::TypedParam>
) -> CodegenResult<(HashSet<String>, Vec<TypedParam>)> {
    params.into_iter()
        .fold(Ok((def_vars, vec![])), |acc, param| {
            let (mut def_vars, mut params) = acc?;
            let ir_ast::TypedParam {id, ty} = param;
            def_vars.insert(id.clone());
            let ty = compile_ir_type(ty)?;
            params.push(TypedParam {id, ty});
            Ok((def_vars, params))
        })
}

type CodegenEnv = HashMap<String, (Vec<TypedParam>, Vec<Stmt>)>;

fn compile_ir_def(
    mut env : CodegenEnv,
    ast : Ast,
    def : ir_ast::Def
) -> CodegenResult<(CodegenEnv, Ast)> {
    match def {
        ir_ast::Def::FunDef {id, params, body} => {
            let vars = HashSet::new();
            let (vars, params) = compile_ir_params(vars, params)?;
            let (_, body) = compile_ir_stmts(vars, body)?;
            env.insert(id, (params, body));
            Ok((env, ast))
        },
        ir_ast::Def::ParFunInst {id, par} => {
            if let Some((params, body)) = env.get(&id) {
                let def = ParFunDef {
                    id, params : params.clone(), par
                };
                let ast = instantiate_parallel_function(ast, body.clone(), def)?;
                Ok((env, ast))
            } else {
                codegen_error!("Parallel instantiation {id} refers to unknown function")
            }
        }
    }
}

pub fn ir_to_code(
    ir_ast : ir_ast::Ast
) -> CodegenResult<Ast> {
    let env = HashMap::new();
    let (_, ast) = ir_ast.into_iter()
        .fold(Ok((env, Ast::new())), |acc, ir_def| {
            let (env, ast) = acc?;
            compile_ir_def(env, ast, ir_def)
        })?;
    Ok(ast)
}
