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
        py_ast::Type::String => {
            parir_compile_error!(i, "Encountered standalone string type when translating to IR AST")
        },
        py_ast::Type::Tensor {sz, shape} if shape.is_empty() => Ok(Type::Scalar {sz}),
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

fn to_float_literal_value(func: py_ast::Builtin, i: &Info) -> CompileResult<f64> {
    match func {
        py_ast::Builtin::Inf => Ok(f64::INFINITY),
        _ => parir_compile_error!(i, "Invalid builtin literal value: {func}")
    }
}

fn to_unary_op(func: py_ast::Builtin, i: &Info) -> CompileResult<UnOp> {
    match func {
        py_ast::Builtin::Exp => Ok(UnOp::Exp),
        py_ast::Builtin::Log => Ok(UnOp::Log),
        py_ast::Builtin::Cos => Ok(UnOp::Cos),
        py_ast::Builtin::Sin => Ok(UnOp::Sin),
        py_ast::Builtin::Sqrt => Ok(UnOp::Sqrt),
        py_ast::Builtin::Tanh => Ok(UnOp::Tanh),
        py_ast::Builtin::Abs => Ok(UnOp::Abs),
        py_ast::Builtin::Inf | py_ast::Builtin::Max | py_ast::Builtin::Min |
        py_ast::Builtin::Atan2 | py_ast::Builtin::Convert {..} => {
            parir_compile_error!(i, "Invalid builtin unary operator: {func}")
        }
    }
}

fn to_binary_op(func: py_ast::Builtin, i: &Info) -> CompileResult<BinOp> {
    match func {
        py_ast::Builtin::Max => Ok(BinOp::Max),
        py_ast::Builtin::Min => Ok(BinOp::Min),
        py_ast::Builtin::Atan2 => Ok(BinOp::Atan2),
        py_ast::Builtin::Exp | py_ast::Builtin::Inf | py_ast::Builtin::Log |
        py_ast::Builtin::Cos | py_ast::Builtin::Sin | py_ast::Builtin::Sqrt |
        py_ast::Builtin::Tanh | py_ast::Builtin::Abs |
        py_ast::Builtin::Convert {..} => {
            parir_compile_error!(i, "Invalid builtin binary operator: {func}")
        }
    }
}

fn to_builtin(
    func: py_ast::Builtin,
    mut args: Vec<Expr>,
    ty: Type,
    i: Info
) -> CompileResult<Expr> {
    match args.len() {
        0 => {
            let v = to_float_literal_value(func, &i)?;
            Ok(Expr::Float {v, ty, i})
        },
        1 => {
            let op = to_unary_op(func, &i)?;
            let arg = Box::new(args.remove(0));
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        2 => {
            let op = to_binary_op(func, &i)?;
            let lhs = Box::new(args.remove(0));
            let rhs = Box::new(args.remove(0));
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        n => parir_compile_error!(i, "Builtin {func} does not expect {n} arguments")
    }
}

fn to_ir_unop(unop: py_ast::UnOp) -> UnOp {
    match unop {
        py_ast::UnOp::Sub => UnOp::Sub,
    }
}

fn to_ir_binop(binop: py_ast::BinOp) -> BinOp {
    match binop {
        py_ast::BinOp::Add => BinOp::Add,
        py_ast::BinOp::Sub => BinOp::Sub,
        py_ast::BinOp::Mul => BinOp::Mul,
        py_ast::BinOp::FloorDiv | py_ast::BinOp::Div => BinOp::Div,
        py_ast::BinOp::Mod => BinOp::Rem,
        py_ast::BinOp::Pow => BinOp::Pow,
        py_ast::BinOp::And => BinOp::And,
        py_ast::BinOp::Or => BinOp::Or,
        py_ast::BinOp::BitAnd => BinOp::BitAnd,
        py_ast::BinOp::Eq => BinOp::Eq,
        py_ast::BinOp::Neq => BinOp::Neq,
        py_ast::BinOp::Leq => BinOp::Leq,
        py_ast::BinOp::Geq => BinOp::Geq,
        py_ast::BinOp::Lt => BinOp::Lt,
        py_ast::BinOp::Gt => BinOp::Gt,
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
            let op = to_ir_unop(op);
            let arg = Box::new(to_ir_expr(env, *arg)?);
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        py_ast::Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = Box::new(to_ir_expr(env, *lhs)?);
            let op = to_ir_binop(op);
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
                        let (idx, _) = shape.clone()
                            .into_iter()
                            .rev()
                            .zip(elems.clone().into_iter().rev())
                            .fold(Ok((zero, 1)), |acc, (n, idx)| {
                                let (expr, mult) = acc?;
                                let nexpr = Expr::Int {
                                    v: mult, ty: int_ty.clone(), i: i_tuple.clone()
                                };
                                let idx = to_ir_expr(env, idx)?;
                                let idx_expr = Expr::BinOp {
                                    lhs: Box::new(Expr::BinOp {
                                        lhs: Box::new(nexpr),
                                        op: BinOp::Mul,
                                        rhs: Box::new(idx),
                                        ty: int_ty.clone(),
                                        i: i_tuple.clone()
                                    }),
                                    op: BinOp::Add,
                                    rhs: Box::new(expr),
                                    ty: int_ty.clone(),
                                    i: i_tuple.clone()
                                };
                                Ok((idx_expr, mult * n))
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
            to_builtin(func, args, ty, i)
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
        py_ast::Stmt::While {cond, body, i} => {
            let cond = to_ir_expr(env, cond)?;
            let body = to_ir_stmts(env, body)?;
            Ok(Stmt::While {cond, body, i})
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
