use super::ast::*;

use crate::prickle_compile_error;
use crate::prickle_internal_error;
use crate::option::CompileOptions;
use crate::utils::ast::ScalarSizes;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::py::ast as py_ast;

use itertools::Itertools;

use std::collections::BTreeMap;

pub struct IREnv {
    structs: BTreeMap<py_ast::Type, Name>,
    par: BTreeMap<String, LoopPar>,
    scalar_sizes: ScalarSizes
}

impl IREnv {
    pub fn new(
        structs: BTreeMap<py_ast::Type, Name>,
        par: BTreeMap<String, LoopPar>,
        opts: &CompileOptions
    ) -> Self {
        IREnv {structs, par, scalar_sizes: ScalarSizes::from_opts(opts)}
    }
}

pub fn to_struct_def(
    env: &IREnv,
    id: Name,
    ty: py_ast::Type
) -> CompileResult<Top> {
    let i = Info::default();
    let mut fields = ty.get_dict_type_fields().into_iter()
        .map(|(id, ty)| Ok(Field {id, ty: to_ir_type(env, &i, ty)?, i: i.clone()}))
        .collect::<CompileResult<Vec<Field>>>()?;
    fields.sort_by(|Field {id: lid, ..}, Field {id: rid, ..}| lid.cmp(&rid));
    Ok(Top::StructDef {id, fields, i: Info::default()})
}

fn to_ir_type(
    env: &IREnv,
    i: &Info,
    ty: py_ast::Type
) -> CompileResult<Type> {
    match ty {
        py_ast::Type::String => {
            prickle_compile_error!(i, "Encountered standalone string type when translating to IR AST")
        },
        py_ast::Type::Tensor {sz, shape} => Ok(Type::Tensor {sz, shape}),
        py_ast::Type::Pointer {sz} => {
            Ok(Type::Pointer {ty: Box::new(Type::Tensor {sz, shape: vec![]})})
        },
        py_ast::Type::Tuple {..} => {
            prickle_compile_error!(i, "Encountered standalone tuple type when translating to IR AST")
        },
        py_ast::Type::Dict {..} => {
            if let Some(id) = env.structs.get(&ty) {
                Ok(Type::Struct {id: id.clone()})
            } else {
                prickle_compile_error!(i, "Encountered unknown dictionary type when translating to IR AST")
            }
        },
        py_ast::Type::Void => Ok(Type::Void),
        py_ast::Type::Unknown => {
            prickle_compile_error!(i, "Encountered unknown type when translating to IR AST")
        },
    }
}

fn to_float_literal_value(func: py_ast::Builtin, i: &Info) -> CompileResult<f64> {
    match func {
        py_ast::Builtin::Inf => Ok(f64::INFINITY),
        _ => prickle_compile_error!(i, "Invalid builtin literal value: {func}")
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
        py_ast::Builtin::Atan2 | py_ast::Builtin::Sum | py_ast::Builtin::Prod |
        py_ast::Builtin::Convert {..} | py_ast::Builtin::Label |
        py_ast::Builtin::GpuContext => {
            prickle_compile_error!(i, "Invalid builtin unary operator: {func}")
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
        py_ast::Builtin::Tanh | py_ast::Builtin::Abs | py_ast::Builtin::Sum |
        py_ast::Builtin::Prod | py_ast::Builtin::Convert {..} |
        py_ast::Builtin::Label | py_ast::Builtin::GpuContext => {
            prickle_compile_error!(i, "Invalid builtin binary operator: {func}")
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
        n => prickle_compile_error!(i, "Builtin {func} does not expect {n} arguments")
    }
}

fn unwrap_tensor_indices(
    env: &IREnv,
    target: py_ast::Expr,
    idx: py_ast::Expr
) -> CompileResult<(Expr, Vec<Expr>)> {
    let mut idx = match idx {
        py_ast::Expr::Tuple {elems, ..} => {
            elems.into_iter()
                .map(|elem| to_ir_expr(env, elem))
                .collect::<CompileResult<Vec<Expr>>>()
        },
        _ => {
            Ok(vec![to_ir_expr(env, idx)?])
        }
    }?;
    let is_string = |e: &py_ast::Expr| match e {
        py_ast::Expr::String {..} => true,
        _ => false
    };
    match target {
        py_ast::Expr::Subscript {target: itarget, idx: iidx, ..} if !is_string(iidx.as_ref()) => {
            let (target, mut inner_indices) = unwrap_tensor_indices(env, *itarget, *iidx)?;
            inner_indices.append(&mut idx);
            Ok((target, inner_indices))
        },
        _ => {
            Ok((to_ir_expr(env, target)?, idx))
        }
    }
}

fn flatten_indices(
    env: &IREnv,
    mut shape: Vec<i64>,
    indices: Vec<Expr>,
    i: &Info
) -> CompileResult<Expr> {
    let int_ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
    let zero = Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()};
    let nindices = indices.len();
    let tail = shape.split_off(nindices);
    let init = tail.into_iter().product::<i64>() as i128;
    let (idx, _) = shape.clone()
        .into_iter()
        .rev()
        .zip(indices.into_iter().rev())
        .fold(Ok((zero, init)), |acc, (n, idx)| {
            let (expr, mult) = acc?;
            let i = idx.get_info();
            let nexpr = Expr::Int {v: mult, ty: int_ty.clone(), i: i.clone()};
            let idx_expr = Expr::BinOp {
                lhs: Box::new(Expr::BinOp {
                    lhs: Box::new(idx),
                    op: BinOp::Mul,
                    rhs: Box::new(nexpr),
                    ty: int_ty.clone(),
                    i: i.clone()
                }),
                op: BinOp::Add,
                rhs: Box::new(expr),
                ty: int_ty.clone(),
                i: i.clone()
            };
            Ok((idx_expr, mult * n as i128))
        })?;
    Ok(idx)
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
            prickle_compile_error!(i, "String literal may only be used in dict lookups")
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
        py_ast::Expr::IfExpr {cond, thn, els, ty, i} => {
            let cond = Box::new(to_ir_expr(env, *cond)?);
            let thn = Box::new(to_ir_expr(env, *thn)?);
            let els = Box::new(to_ir_expr(env, *els)?);
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::IfExpr {cond, thn, els, ty, i})
        },
        py_ast::Expr::Subscript {target, idx, ty, i} => {
            let ty = to_ir_type(env, &i, ty)?;
            // We can use a subscript to index in a dictionary using a string key, or we can use it
            // as an integer index into a tensor. In the latter case, we collect nested uses of
            // indexing, as "a[1,2]" should be considered equivalent to "a[1][2]".
            if let py_ast::Expr::String {v: label, ..} = *idx {
                let target = Box::new(to_ir_expr(env, *target)?);
                Ok(Expr::StructFieldAccess {target, label, ty, i})
            } else {
                let (target, indices) = unwrap_tensor_indices(env, *target, *idx)?;
                if let Type::Tensor {sz, shape} = target.get_type() {
                    let n = indices.len();
                    if n <= shape.len() {
                        let idx = Box::new(flatten_indices(&env, shape.clone(), indices, &i)?);
                        let res_shape = shape.clone()
                            .into_iter()
                            .skip(n)
                            .collect::<Vec<i64>>();
                        let res_ty = Type::Tensor {sz: sz.clone(), shape: res_shape};
                        let target = Box::new(target);
                        Ok(Expr::TensorAccess {target, idx, ty: res_ty, i})
                    } else {
                        let msg = format!(
                            "Invalid dimensions. Indexing into tensor of shape \
                             {shape:?} using {n} indices"
                        );
                        prickle_compile_error!(i, "{msg}")
                    }
                } else {
                    let msg = "Indexing into non-tensor target {target} is not supported";
                    prickle_compile_error!(i, "{msg}")
                }
            }
        },
        py_ast::Expr::Slice {i, ..} => {
            prickle_compile_error!(i, "Slices are not allowed outside of indexing")
        },
        py_ast::Expr::Tuple {i, ..} => {
            prickle_compile_error!(i, "Tuples are not allowed outside of indexing")
        },
        py_ast::Expr::Call {id, args, ty, i} => {
            let args = args.into_iter()
                .map(|e| to_ir_expr(env, e))
                .collect::<CompileResult<Vec<Expr>>>()?;
            let ty = to_ir_type(env, &i, ty)?;
            Ok(Expr::Call {id, args, ty, i})
        },
        py_ast::Expr::NeutralElement {i, ..} => {
            prickle_internal_error!(i, "Intermediate reduction node remaining during IR translation")
        },
        py_ast::Expr::Builtin {func, args, ty, i, ..} => {
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

fn lookup_labels(
    par: &BTreeMap<String, LoopPar>,
    labels: Vec<String>,
    i: &Info
) -> CompileResult<LoopPar> {
    let par = labels.clone()
        .into_iter()
        .map(|l| par.get(&l))
        .fold(Some(LoopPar::default()), |acc, p| acc?.try_merge(p));
    match par {
        Some(p) => Ok(p),
        None => {
            let labels = labels.into_iter().join(", ");
            let msg = format!(
                "The labels associated with this statement specify conflicting \
                 number of threads in parallelization.\n\
                 Labels of this statement: {labels}"
            );
            prickle_compile_error!(i, "{}", msg)
        }
    }
}

fn to_ir_stmt(
    env: &IREnv,
    stmt: py_ast::Stmt
) -> CompileResult<Stmt> {
    match stmt {
        py_ast::Stmt::Definition {ty, id, expr, i, ..} => {
            let ty = to_ir_type(env, &i, ty)?;
            let expr = to_ir_expr(env, expr)?;
            Ok(Stmt::Definition {ty, id, expr, i})
        },
        py_ast::Stmt::Assign {dst, expr, i, ..} => {
            let dst = to_ir_expr(env, dst)?;
            let expr = to_ir_expr(env, expr)?;
            Ok(Stmt::Assign {dst, expr, i})
        },
        py_ast::Stmt::For {var, lo, hi, step, body, labels, i} => {
            let lo = to_ir_expr(env, lo)?;
            let hi = to_ir_expr(env, hi)?;
            let body = to_ir_stmts(env, body)?;
            let par = lookup_labels(&env.par, labels, &i)?;
            Ok(Stmt::For {var, lo, hi, step, body, par, i})
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
        py_ast::Stmt::Return {value, i} => {
            let value = to_ir_expr(env, value)?;
            Ok(Stmt::Return {value, i})
        },
        py_ast::Stmt::WithGpuContext {body, i} => {
            // NOTE: To ensure code within a GPU context actually runs on the GPU, we generate a
            // for-loop with a single iteration annotated to run in parallel on 1 thread.
            let var = Name::sym_str("_gpu_context");
            let int64_ty = Type::Tensor {sz: env.scalar_sizes.int.clone(), shape: vec![]};
            let lo = Expr::Int {v: 0, ty: int64_ty.clone(), i: i.clone()};
            let hi = Expr::Int {v: 1, ty: int64_ty.clone(), i: i.clone()};
            let body = to_ir_stmts(env, body)?;
            let par = LoopPar::default().threads(1).unwrap();
            Ok(Stmt::For {var, lo, hi, step: 1, body, par, i})
        },
        py_ast::Stmt::Scope {i, ..} => {
            prickle_compile_error!(i,
                "Found intermediate scope statement that should have been \
                 removed by the compiler"
            )
        },
        py_ast::Stmt::Call {func, i, ..} => {
            prickle_compile_error!(i, "Found unsupported function call to {func}")
        },
        py_ast::Stmt::Label {i, ..} => {
            prickle_compile_error!(i, "Found label without associated statement")
        }
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

fn to_ir_params(
    env: &IREnv,
    params: Vec<py_ast::Param>
) -> CompileResult<Vec<Param>> {
    params.into_iter()
        .map(|p| to_ir_param(env, p))
        .collect::<CompileResult<Vec<Param>>>()
}

fn to_ir_def(
    env: &IREnv,
    def: py_ast::FunDef
) -> CompileResult<FunDef> {
    let py_ast::FunDef {id, params, body, res_ty, i} = def;
    let params = to_ir_params(env, params)?;
    let body = to_ir_stmts(env, body)?;
    let res_ty = to_ir_type(env, &i, res_ty)?;
    Ok(FunDef {id, params, body, res_ty, i})
}

fn to_ir_top(
    env: &IREnv,
    t: py_ast::Top
) -> CompileResult<Top> {
    match t {
        py_ast::Top::ExtDecl {id, ext_id, params, res_ty, header, target, par, i} => {
            let params = to_ir_params(env, params)?;
            let res_ty = to_ir_type(env, &i, res_ty)?;
            Ok(Top::ExtDecl {id, ext_id, params, res_ty, header, target, par, i})
        },
        py_ast::Top::FunDef {v} => Ok(Top::FunDef {v: to_ir_def(env, v)?}),
    }
}

pub fn to_ir_ast(
    env: &IREnv,
    ast: py_ast::Ast,
    structs: Vec<Top>
) -> CompileResult<Ast> {
    let tops = structs.into_iter()
        .map(|t| Ok(t))
        .chain(ast.tops.into_iter().map(|t| to_ir_top(env, t)))
        .collect::<CompileResult<Vec<Top>>>()?;
    let main = to_ir_def(env, ast.main)?;
    Ok(Ast {tops, main})
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ast_builder::*;
    use crate::ir::constant_fold;
    use crate::test::*;
    use crate::py::ast_builder as py;

    use std::collections::BTreeMap;

    fn ir_env() -> IREnv {
        IREnv::new(BTreeMap::new(), BTreeMap::new(), &CompileOptions::default())
    }

    fn conv_ir_type(ty: py_ast::Type) -> CompileResult<Type> {
        to_ir_type(&ir_env(), &i(), ty)
    }

    #[test]
    fn convert_struct_def() {
        let env = ir_env();
        let id = id("x");
        let ty = py::dict_ty(vec![("y", py::scalar(ElemSize::I32))]);
        let expected = Top::StructDef {
            id: id.clone(),
            fields: vec![Field {
                id: "y".to_string(), ty: scalar(ElemSize::I32), i: i()
            }],
            i: i()
        };
        assert_eq!(to_struct_def(&env, id, ty).unwrap(), expected);
    }

    #[test]
    fn string_ir_type() {
        let r = conv_ir_type(py_ast::Type::String);
        assert_error_matches(r, r"standalone string type");
    }

    #[test]
    fn scalar_ir_type() {
        let r = conv_ir_type(py::scalar(ElemSize::I32));
        assert_eq!(r.unwrap(), scalar(ElemSize::I32));
    }

    #[test]
    fn tensor_vector_ir_type() {
        let r = conv_ir_type(py::shape(vec![10]));
        assert_eq!(r.unwrap(), shape(vec![10]));
    }

    #[test]
    fn tuple_ir_type() {
        let r = conv_ir_type(py_ast::Type::Tuple {elems: vec![]});
        assert_error_matches(r, r"standalone tuple type");
    }

    #[test]
    fn unknown_dict_type() {
        let r = conv_ir_type(py_ast::Type::Dict {fields: BTreeMap::new()});
        assert_error_matches(r, r"unknown dictionary type");
    }

    #[test]
    fn known_dict_type() {
        let dty = py::dict_ty(vec![("x", py::scalar(ElemSize::I32))]);
        let mut structs = BTreeMap::new();
        let id = id("y");
        structs.insert(dty.clone(), id.clone());
        let env = IREnv::new(structs, BTreeMap::new(), &CompileOptions::default());
        assert_eq!(to_ir_type(&env, &i(), dty).unwrap(), Type::Struct {id});
    }

    #[test]
    fn void_type() {
        assert_eq!(conv_ir_type(py_ast::Type::Void).unwrap(), Type::Void);
    }

    #[test]
    fn unknown_type() {
        let r = conv_ir_type(py_ast::Type::Unknown);
        assert_error_matches(r, r"unknown type when translating to IR");
    }

    #[test]
    fn inf_float_literal_builtin() {
        let r = to_float_literal_value(py_ast::Builtin::Inf, &i());
        assert_eq!(r.unwrap(), f64::INFINITY);
    }

    #[test]
    fn non_float_literal_builtin() {
        let r = to_float_literal_value(py_ast::Builtin::Max, &i());
        assert_error_matches(r, r"Invalid builtin literal value");
    }

    #[test]
    fn invalid_unop_builtin_max() {
        let r = to_unary_op(py_ast::Builtin::Max, &i());
        assert_error_matches(r, r"Invalid builtin unary");
    }

    #[test]
    fn invalid_binop_builtin_exp() {
        let r = to_binary_op(py_ast::Builtin::Exp, &i());
        assert_error_matches(r, r"Invalid builtin binary");
    }

    #[test]
    fn to_float_inf_expr() {
        let r = to_builtin(py_ast::Builtin::Inf, vec![], scalar(ElemSize::F32), i());
        assert_eq!(r.unwrap(), float(f64::INFINITY, None));
    }

    #[test]
    fn builtin_to_unop_expr() {
        let r = to_builtin(
            py_ast::Builtin::Log, vec![float(1.5, None)], scalar(ElemSize::F32), i()
        );
        assert_eq!(r.unwrap(), unop(UnOp::Log, float(1.5, None)));
    }

    #[test]
    fn builtin_to_binop_expr() {
        let ty = scalar(ElemSize::F32);
        let r = to_builtin(
            py_ast::Builtin::Min, vec![float(1.0, None), float(2.0, None)], ty.clone(), i()
        );
        let expected = binop(float(1.0, None), BinOp::Min, float(2.0, None), Some(ty));
        assert_eq!(r.unwrap(), expected);
    }

    #[test]
    fn builtin_too_many_args() {
        let args = vec![float(1.0, None), float(2.0, None), float(3.0, None)];
        let r = to_builtin(py_ast::Builtin::Sum, args, scalar(ElemSize::F32), i());
        assert_error_matches(r, r"does not expect 3 arguments");
    }

    #[test]
    fn flatten_1d_index() {
        let shape = vec![10];
        let indices = vec![int(2, None)];
        let e = flatten_indices(&ir_env(), shape, indices, &i()).unwrap();
        assert_eq!(e, binop(
            binop(int(2, None), BinOp::Mul, int(1, None), None),
            BinOp::Add,
            int(0, None),
            None
        ));
    }

    #[test]
    fn flatten_2d_index() {
        let shape = vec![10, 10];
        let indices = vec![int(1, None), int(2, None)];
        let e = flatten_indices(&ir_env(), shape, indices, &i()).unwrap();
        assert_eq!(constant_fold::fold_expr(e), int(12, None));
    }

    #[test]
    fn flatten_3d_index() {
        let shape = vec![10, 20, 30];
        let indices = vec![int(3, None), int(2, None), int(1, None)];
        let e = flatten_indices(&ir_env(), shape, indices, &i()).unwrap();
        assert_eq!(constant_fold::fold_expr(e), int(1861, None));
    }

    #[test]
    fn var_expr_to_ir() {
        let r = to_ir_expr(&ir_env(), py::var("x", py::scalar(ElemSize::I32)));
        assert_eq!(r.unwrap(), var("x", scalar(ElemSize::I32)));
    }

    #[test]
    fn string_expr_to_ir() {
        let s = py_ast::Expr::String {v: "x".to_string(), ty: py_ast::Type::String, i: i()};
        assert_error_matches(to_ir_expr(&ir_env(), s), r"literal may only be used in");
    }

    #[test]
    fn binop_to_ir() {
        let e = py::binop(
            py::int(1, Some(ElemSize::I64)),
            BinOp::Add,
            py::int(2, Some(ElemSize::I64)),
            py::scalar(ElemSize::I64)
        );
        let expected = binop(
            int(1, Some(ElemSize::I64)),
            BinOp::Add,
            int(2, Some(ElemSize::I64)),
            None
        );
        assert_eq!(to_ir_expr(&ir_env(), e).unwrap(), expected);
    }

    #[test]
    fn dict_subscript_expr_to_ir() {
        let dty = py::dict_ty(vec![("y", py::scalar(ElemSize::I64))]);
        let mut structs = BTreeMap::new();
        let id = id("z");
        structs.insert(dty.clone(), id.clone());
        let env = IREnv::new(structs, BTreeMap::new(), &CompileOptions::default());
        let subscript = py::subscript(
            py::var("x", dty), py::string("y"), py::scalar(ElemSize::I64)
        );
        let expected = Expr::StructFieldAccess {
            target: Box::new(var("x", Type::Struct {id})),
            label: "y".to_string(),
            ty: scalar(ElemSize::I64),
            i: i()
        };
        assert_eq!(to_ir_expr(&env, subscript).unwrap(), expected);
    }

    #[test]
    fn tensor_subscript_non_tensor_target() {
        let dty = py::dict_ty(vec![]);
        let e = py::subscript(
            py::var("x", dty.clone()),
            py::int(1, Some(ElemSize::I32)),
            py::scalar(ElemSize::I32)
        );
        let mut structs = BTreeMap::new();
        structs.insert(dty, id("?"));
        let env = IREnv::new(structs, BTreeMap::new(), &CompileOptions::default());
        assert_error_matches(to_ir_expr(&env, e), r"non-tensor target.*not supported");
    }

    #[test]
    fn tensor_subscript_invalid_dims() {
        let e = py::subscript(
            py::var("x", py::scalar(ElemSize::I32)),
            py::int(1, Some(ElemSize::I32)),
            py::scalar(ElemSize::I32)
        );
        let pat = r"Indexing into tensor of shape \[\] using 1 indices";
        assert_error_matches(to_ir_expr(&ir_env(), e), pat);
    }

    #[test]
    fn tensor_subscript_expr_to_ir() {
        let e = py::subscript(
            py::var("x", py::shape(vec![10, 20])),
            py::tuple(vec![py::int(1, Some(ElemSize::I64)), py::int(2, Some(ElemSize::I64))]),
            py::scalar(ElemSize::I64)
        );
        let r = to_ir_expr(&ir_env(), e);
        let expected = tensor_access(
            var("x", shape(vec![200])),
            int(22, None),
            scalar(ElemSize::I64)
        );
        assert_eq!(constant_fold::fold_expr(r.unwrap()), expected);
    }

    #[test]
    fn slice_expr_to_ir() {
        let r = to_ir_expr(&ir_env(), py::slice(None, None));
        assert_error_matches(r, r"compile error.*Slices are not allowed outside");
    }

    #[test]
    fn tuple_expr_to_ir() {
        let r = to_ir_expr(&ir_env(), py::tuple(vec![]));
        assert_error_matches(r, r"compile error.*Tuples are not allowed outside");
    }

    #[test]
    fn neutral_element_expr_to_ir() {
        let ne = py_ast::Expr::NeutralElement {
            op: BinOp::Add, tyof: Box::new(py::int(1, Some(ElemSize::I64))),
            i: Info::default()
        };
        let r = to_ir_expr(&ir_env(), ne);
        assert_error_matches(r, r"Internal.*Intermediate reduction node");
    }

    fn make_par_map(v: Vec<(&str, LoopPar)>) -> BTreeMap<String, LoopPar> {
        v.into_iter()
            .map(|(id, p)| (id.to_string(), p))
            .collect::<BTreeMap<String, LoopPar>>()
    }

    #[test]
    fn lookup_unknown_labels() {
        let par = make_par_map(vec![]);
        let labels = vec!["x".to_string()];
        assert_eq!(lookup_labels(&par, labels, &i()).unwrap(), LoopPar::default());
    }

    #[test]
    fn lookup_known_label() {
        let p = LoopPar::default().threads(2).unwrap();
        let par = make_par_map(vec![
            ("x", p.clone()),
        ]);
        let labels = vec!["x".to_string()];
        assert_eq!(lookup_labels(&par, labels, &i()).unwrap(), p);
    }

    #[test]
    fn lookup_inconsistent_labels() {
        let par = make_par_map(vec![
            ("x", LoopPar::default().threads(2).unwrap()),
            ("y", LoopPar::default().threads(4).unwrap()),
        ]);
        let labels = vec!["x".to_string(), "y".to_string()];
        assert_error_matches(lookup_labels(&par, labels, &i()), r"conflicting number of threads");
    }

    #[test]
    fn lookup_consistent_labels() {
        let par = make_par_map(vec![
            ("x", LoopPar::default().threads(4).unwrap()),
            ("y", LoopPar::default().tpb(512).unwrap()),
            ("z", LoopPar::default().threads(4).unwrap()),
        ]);
        let labels = ["x", "y", "z"].into_iter().map(|x| x.to_string()).collect::<Vec<String>>();
        let expected = LoopPar::default()
            .threads(4).unwrap()
            .tpb(512).unwrap();
        assert_eq!(lookup_labels(&par, labels, &i()).unwrap(), expected);
    }

    #[test]
    fn with_gpu_context_stmt_to_ir() {
        let s = py_ast::Stmt::WithGpuContext {body: vec![], i: i()};
        let s = to_ir_stmt(&ir_env(), s).unwrap();
        assert!(matches!(s, Stmt::For {..}));
    }
}
