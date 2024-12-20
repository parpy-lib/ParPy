use crate::parir_runtime_error;
use crate::parir_type_error;
use crate::err::*;
use crate::info::*;
use crate::par::*;
use crate::ir::ast::*;
use crate::py_ir::ast as src_ast;

use std::collections::HashMap;

// Converting the Python IR AST to a lower-level IR, on which we perform the mapping of parallelism
// onto the GPU.

fn from_int_size(sz: src_ast::IntSize, i: &Info) -> CompileResult<IntSize> {
    match sz {
        src_ast::IntSize::I8 => Ok(IntSize::I8),
        src_ast::IntSize::I16 => Ok(IntSize::I16),
        src_ast::IntSize::I32 => Ok(IntSize::I32),
        src_ast::IntSize::I64 => Ok(IntSize::I64),
        src_ast::IntSize::Any =>
            parir_type_error!(i, "Found unspecified integer size type in intermediate AST")
    }
}

fn from_float_size(sz: src_ast::FloatSize, i: &Info) -> CompileResult<FloatSize> {
    match sz {
        src_ast::FloatSize::F16 => Ok(FloatSize::F16),
        src_ast::FloatSize::F32 => Ok(FloatSize::F32),
        src_ast::FloatSize::F64 => Ok(FloatSize::F64),
        src_ast::FloatSize::Any =>
            parir_type_error!(i, "Found unspecified floating-point size type in intermediate AST")
    }
}

fn from_type(ty: src_ast::Type, i: &Info) -> CompileResult<Type> {
    match ty {
        src_ast::Type::Int(sz) =>
            Ok(Type::Int(from_int_size(sz, &i)?)),
        src_ast::Type::Float(sz) =>
            Ok(Type::Float(from_float_size(sz, &i)?)),
        src_ast::Type::IntTensor(sz) =>
            Ok(Type::IntTensor(from_int_size(sz, &i)?)),
        src_ast::Type::FloatTensor(sz) =>
            Ok(Type::FloatTensor(from_float_size(sz, &i)?)),
        src_ast::Type::Unknown =>
            parir_type_error!(i, "Found unknown type in intermediate AST")
    }
}

impl TryFrom<src_ast::TypedParam> for TypedParam {
    type Error = CompileError;

    fn try_from(param: src_ast::TypedParam) -> CompileResult<TypedParam> {
        let src_ast::TypedParam {id, ty, i} = param;
        let ty = from_type(ty, &i)?;
        Ok(TypedParam {id, ty, i})
    }
}

fn from_params(params: Vec<src_ast::TypedParam>) -> CompileResult<Vec<TypedParam>> {
    params.into_iter()
        .map(|p| p.try_into())
        .collect()
}

impl TryFrom<src_ast::Expr> for Expr {
    type Error = CompileError;

    fn try_from(e: src_ast::Expr) -> CompileResult<Expr> {
        match e {
            src_ast::Expr::Var {id, ty, i} => {
                let ty = from_type(ty, &i)?;
                Ok(Expr::Var {id, ty, i})
            },
            src_ast::Expr::Int {v, ty, i} => {
                let ty = from_type(ty, &i)?;
                Ok(Expr::Int {v, ty, i})
            },
            src_ast::Expr::Float {v, ty, i} => {
                let ty = from_type(ty, &i)?;
                Ok(Expr::Float {v, ty, i})
            },
            src_ast::Expr::BinOp {lhs, op, rhs, ty, i} => {
                let lhs = lhs.try_into()?;
                let rhs = rhs.try_into()?;
                let ty = from_type(ty, &i)?;
                Ok(Expr::BinOp {lhs, op, rhs, ty, i})
            },
            src_ast::Expr::Subscript {target, idx, ty, i} => {
                let target = target.try_into()?;
                let idx = idx.try_into()?;
                let ty = from_type(ty, &i)?;
                Ok(Expr::Subscript {target, idx, ty, i})
            }
        }
    }
}

impl TryFrom<Box<src_ast::Expr>> for Box<Expr> {
    type Error = CompileError;

    fn try_from(e: Box<src_ast::Expr>) -> CompileResult<Box<Expr>> {
        Ok(Box::new((*e).try_into()?))
    }
}

fn from_stmt(stmt: src_ast::Stmt) -> CompileResult<Vec<Stmt>> {
    match stmt {
        src_ast::Stmt::Assign {dst, e, i} => {
            let dst = dst.try_into()?;
            let e = e.try_into()?;
            match dst {
                Expr::Var {id, ty, i: var_info} => {
                    Ok(vec![Stmt::AssignVar {var: id, e, i}])
                },
                Expr::Subscript {target, idx, ..} => match *target {
                    Expr::Var {id: var, ty, ..} => {
                        Ok(vec![Stmt::AssignArray {var, idx: *idx, e, i}])
                    },
                    _ => parir_runtime_error!(i, "Unsupported subscript assignment form")
                },
                _ => parir_runtime_error!(i, "Unsupported assignment expression form")
            }
        },
        src_ast::Stmt::For {var, lo, hi, body, i} => {
            let lo = lo.try_into()?;
            let hi = hi.try_into()?;
            let body = from_stmts(body)?;
            let properties = LoopProperties::default();
            let stmt = Stmt::For {var, lo, hi, body, properties, i};
            Ok(vec![stmt])
        }
    }
}

fn from_stmts(stmts: Vec<src_ast::Stmt>) -> CompileResult<Vec<Stmt>> {
    let mut s = vec![];
    for stmt in stmts.into_iter() {
        for inner_stmt in from_stmt(stmt)? {
            s.push(inner_stmt);
        }
    }
    Ok(s)
}

// Mapping of the parallelism available in for-loops to one or more concrete CUDA kernels (GPU) or
// to OpenMP annotated for-loops (CPU).

fn instantiate_parallel_function(
    ast: Ast,
    body: Vec<Stmt>,
    par: HashMap<String, ParSpec>,
    i: Info
) -> CompileResult<Ast> {
    parir_runtime_error!(i, "not done with instantiate_parallel_function yet...")
    /*let (on_device, loop_props) = analyze_parallelism_function(&body, &def)?;
    let (mut ast, host_body) = instantiate_stmts(ast, body, &def, false, true)?;
    ast.host = create_host_function(ast.host, host_body, def);
    Ok(ast)*/
}

/// Complete conversion from the Python IR AST to the lower-level AST. In this lower-level AST, we
/// explicitly represent kernel calls and include declarations of variables before their first use.
/// Translating from the resulting IR AST to the final generated code should be straightforward.
impl TryFrom<src_ast::Ast> for Ast {
    type Error = CompileError;

    fn try_from(ir_ast: src_ast::Ast) -> CompileResult<Self> {
        let env: HashMap<String, (Vec<TypedParam>, Vec<Stmt>)> = HashMap::new();
        let (_, ast) = ir_ast.into_iter()
            .fold(Ok((env, Ast::new())), |acc, ir_def| match ir_def {
                src_ast::Def::FunDef {id, params, body, ..} => {
                    let (mut env, ast) = acc?;
                    let params = from_params(params)?;
                    let body = from_stmts(body)?;
                    env.insert(id, (params, body));
                    Ok((env, ast))
                },
                src_ast::Def::ParFunInst {id, par, i, ..} => {
                    let (env, ast) = acc?;
                    if let Some((params, body)) = env.get(&id) {
                        let ast = instantiate_parallel_function(ast, body.clone(), par, i.clone())?;
                        Ok((env, ast))
                    } else {
                        parir_runtime_error!(i, "Parallel instantiation of {id} refers to unknown function")
                    }
                },
            })?;
        Ok(ast)
    }
}
