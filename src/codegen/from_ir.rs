use crate::ir::ast as ir_ast;
use crate::codegen::ast::*;
use crate::par::ParKind;

use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

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

struct ParFunDef {
    id : String,
    params : Vec<TypedParam>,
    body : Vec<Stmt>,
    par : HashMap<String, Vec<ParKind>>
}

fn instantiate_parallel_function(
    ast : Ast,
    def : ParFunDef,
) -> CodegenResult<Ast> {
    // TODO: generate code for the parallel instantiation by picking out the parallel and the
    // sequential parts and adding appropriate definitions and declarations to the accumulated AST
    //
    // Think about how to do this in a rather extensible way, before diving into the details...
    Ok(ast)
}

type CodegenEnv = HashMap<String, (Vec<TypedParam>, Vec<Stmt>)>;

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
        ir_ast::Expr::BinOp {lhs, op, rhs, ty} => {
            let lhs = Box::new(compile_ir_expr(*lhs)?);
            let op = compile_ir_binop(op);
            let rhs = Box::new(compile_ir_expr(*rhs)?);
            let ty = compile_ir_type(ty)?;
            Ok(Expr::BinOp {lhs, op, rhs, ty})
        },
        ir_ast::Expr::Subscript {target, idx, ty} => {
            let target = Box::new(compile_ir_expr(*target)?);
            let idx = Box::new(compile_ir_expr(*idx)?);
            let ty = compile_ir_type(ty)?;
            Ok(Expr::Subscript {target, idx, ty})
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
    mut def_vars : HashSet<String>,
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
                    id, params : params.clone(), body : body.clone(), par
                };
                let ast = instantiate_parallel_function(ast, def)?;
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
