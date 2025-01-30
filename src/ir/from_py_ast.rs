use super::ast::*;

use crate::parir_compile_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::par::ParKind;
use crate::py::ast as py_ast;

use std::collections::BTreeMap;

pub struct IREnv {
    structs: BTreeMap<py_ast::Type, Name>,
    par: BTreeMap<String, Vec<ParKind>>,
}

impl IREnv {
    pub fn new(
        structs: BTreeMap<py_ast::Type, Name>,
        par: BTreeMap<String, Vec<ParKind>>
    ) -> Self {
        IREnv {structs, par}
    }
}

pub fn to_struct_def(
    env: &IREnv,
    id: Name,
    ty: py_ast::Type
) -> CompileResult<StructDef> {
    let i = Info::default();
    let mut fields = ty.get_dict_type_fields().into_iter()
        .map(|(id, ty)| Ok(Field {id, ty: to_ir_type(env, &i, ty)?, i: i.clone()}))
        .collect::<CompileResult<Vec<Field>>>()?;
    fields.sort_by(|Field {id: lid, ..}, Field {id: rid, ..}| lid.cmp(&rid));
    Ok(StructDef {id, fields, i: Info::default()})
}

fn to_ir_type(
    env: &IREnv,
    i: &Info,
    ty: py_ast::Type
) -> CompileResult<Type> {
    match ty {
        py_ast::Type::Boolean => Ok(Type::Boolean),
        py_ast::Type::String => {
            parir_compile_error!(i, "Encountered standalone string type when translating to IR AST")
        },
        py_ast::Type::Tensor {sz, shape} => Ok(Type::Tensor {sz, shape}),
        py_ast::Type::Tuple {..} => {
            parir_compile_error!(i, "Encountered standalone tuple type when translating to IR AST")
        },
        py_ast::Type::Dict {..} => {
            if let Some(id) = env.structs.get(&ty) {
                Ok(Type::Struct {id: id.clone()})
            } else {
                parir_compile_error!(i, "Encountered unknown dictionary type when translating to IR AST")
            }
        },
        py_ast::Type::Unknown => {
            parir_compile_error!(i, "Encountered unknown type when translating to IR AST")
        },
    }
}

fn to_ir_expr(
    env: &IREnv,
    e: py_ast::Expr
) -> CompileResult<Expr> {
    match e {
        py_ast::Expr::Var {id, ty, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Var {id, ty, i})
        },
        py_ast::Expr::String {i, ..} => {
            parir_compile_error!(i, "String literal may only be used for dict lookups")
        },
        py_ast::Expr::Bool {v, ty, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Bool {v, ty, i})
        },
        py_ast::Expr::Int {v, ty, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Int {v, ty, i})
        },
        py_ast::Expr::Float {v, ty, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Float {v, ty, i})
        },
        py_ast::Expr::UnOp {op, arg, ty, i} => {
            let arg = Box::new(to_ir_expr(env, *arg)?);
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        py_ast::Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = Box::new(to_ir_expr(env, *lhs)?);
            let rhs = Box::new(to_ir_expr(env, *rhs)?);
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        py_ast::Expr::Subscript {target, idx, ty, i} => {
            let target = Box::new(to_ir_expr(env, *target)?);
            let ty = to_ir_type(env, &i, ty)?;
            // Three possible uses of a subscript expression:
            // 1. We use a string to index in a dictionary.
            // 2. We use a tuple to index into a multi-dimensional tensor.
            // 3. We use an arbitrary expression (integer) to index into a tensor.
            if let py_ast::Expr::String {v: label, ..} = *idx {
                Ok(Expr::StructFieldAccess {target, label, ty, i})
            } else if let py_ast::Expr::Tuple {elems, i: i_tuple, ..} = *idx {
                if let Type::Tensor {sz, shape} = target.get_type() {
                    if elems.len() <= shape.len() {
                        let int_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
                        let zero = Expr::Int {v: 0, ty: int_ty.clone(), i: i_tuple.clone()};
                        // NOTE: We want to multiply each value by all lower dimensions, so we skip
                        // the most significant one and add a trailing one at the end of the
                        // sequence (to prevent the zip from excluding the final argument).
                        let idx = shape.clone()
                            .into_iter()
                            .skip(1)
                            .chain([1].into_iter())
                            .zip(elems.clone().into_iter())
                            .fold(Ok(zero), |acc, (n, idx)| {
                                let n = Expr::Int {v: n, ty: int_ty.clone(), i: i_tuple.clone()};
                                let idx = to_ir_expr(env, idx)?;
                                Ok(Expr::BinOp {
                                    lhs: Box::new(acc?),
                                    op: BinOp::Add,
                                    rhs: Box::new(Expr::BinOp {
                                        lhs: Box::new(n),
                                        op: BinOp::Mul,
                                        rhs: Box::new(idx),
                                        ty: int_ty.clone(),
                                        i: i_tuple.clone()
                                    }),
                                    ty: int_ty.clone(),
                                    i: i_tuple.clone()
                                })
                            })?;
                        let idx = Box::new(idx.clone());
                        let res_shape = shape.clone()
                            .into_iter()
                            .skip(elems.len())
                            .collect::<Vec<i64>>();
                        let res_ty = Type::Tensor {sz: sz.clone(), shape: res_shape};
                        Ok(Expr::TensorAccess {target, idx, ty: res_ty, i})
                    } else {
                        let dim = shape.len();
                        parir_compile_error!(i, "Too many indices for tensor of {dim} dimensions")
                    }
                } else {
                    parir_compile_error!(i, "Indexing into non-tensor target is not allowed")
                }
            } else {
                let idx = Box::new(to_ir_expr(env, *idx)?);
                Ok(Expr::TensorAccess {target, idx, ty, i})
            }
        },
        py_ast::Expr::Tuple {i, ..} => {
            parir_compile_error!(i, "Tuple literals are not supported outside of indexing")
        },
        py_ast::Expr::Dict {fields, ty, i} => {
            let fields = fields.into_iter()
                .map(|(id, e)| Ok((id, to_ir_expr(env, e)?)))
                .collect::<CompileResult<Vec<(String, Expr)>>>()?;
            if let Some(id) = env.structs.get(&ty) {
                let ty = Type::Struct {id: id.clone()};
                Ok(Expr::Struct {id: id.clone(), fields, ty, i})
            } else {
                parir_compile_error!(i, "Internal compiler error encountered when mapping dictionary to struct")
            }
        },
        py_ast::Expr::Builtin {func, args, ty, i} => {
            let args = args.into_iter()
                .map(|e| to_ir_expr(env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Builtin {func, args, ty, i})
        },
        py_ast::Expr::Convert {e, ty} => {
            let i = e.get_info();
            let e = Box::new(to_ir_expr(env, *e)?);
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Convert {e, ty})
        },
    }
}

fn convert_par_spec(p: Option<&Vec<ParKind>>) -> LoopParallelism {
    p.unwrap_or(&vec![])
        .clone()
        .into_iter()
        .fold(LoopParallelism::default(), |acc, kind| match kind {
            ParKind::GpuThreads(n) => acc.with_threads(n),
            ParKind::GpuReduction {} => acc.with_reduction()
        })
}

fn to_ir_stmt(
    env: &IREnv,
    stmt: py_ast::Stmt
) -> CompileResult<Stmt> {
    match stmt {
        py_ast::Stmt::Definition {ty, id, expr, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            let expr = to_ir_expr(env, expr)?;
            Ok(Stmt::Definition {ty, id, expr, i})
        },
        py_ast::Stmt::Assign {dst, expr, i} => {
            let dst = to_ir_expr(env, dst)?;
            let expr = to_ir_expr(env, expr)?;
            Ok(Stmt::Assign {dst, expr, i})
        },
        py_ast::Stmt::For {var, lo, hi, body, i} => {
            let lo = to_ir_expr(env, lo)?;
            let hi = to_ir_expr(env, hi)?;
            let body = to_ir_stmts(env, body)?;
            let par = convert_par_spec(env.par.get(var.get_str()));
            Ok(Stmt::For {var, lo, hi, body, par, i})
        },
        py_ast::Stmt::If {cond, thn, els, i} => {
            let cond = to_ir_expr(env, cond)?;
            let thn = to_ir_stmts(env, thn)?;
            let els = to_ir_stmts(env, els)?;
            Ok(Stmt::If {cond, thn, els, i})
        },
    }
}

fn to_ir_stmts(
    env: &IREnv,
    stmts: Vec<py_ast::Stmt>
) -> CompileResult<Vec<Stmt>> {
    stmts.into_iter()
        .map(|s| to_ir_stmt(env, s))
        .collect::<_>()
}

fn to_ir_param(
    env: &IREnv,
    p: py_ast::Param
) -> CompileResult<Param> {
    let py_ast::Param {id, ty, i} = p;
    let ty = to_ir_type(env, &i, ty)?;
    Ok(Param {id, ty, i})
}

pub fn to_ir_def(
    env: &IREnv,
    def: py_ast::FunDef
) -> CompileResult<FunDef> {
    let py_ast::FunDef {id, params, body, i} = def;
    let params = params.into_iter()
        .map(|p| to_ir_param(env, p))
        .collect::<CompileResult<Vec<Param>>>()?;
    let body = to_ir_stmts(env, body)?;
    Ok(FunDef {id, params, body, i})
}
