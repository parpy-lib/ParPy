use super::ast::*;

use crate::parir_compile_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::par::ParKind;
use crate::py::ast as py_ast;

use itertools::Itertools;

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
        py_ast::Builtin::Atan2 | py_ast::Builtin::Sum | py_ast::Builtin::Prod |
        py_ast::Builtin::Any | py_ast::Builtin::All |
        py_ast::Builtin::Convert {..} | py_ast::Builtin::Label |
        py_ast::Builtin::GpuContext | py_ast::Builtin::Ext {..} => {
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
        py_ast::Builtin::Tanh | py_ast::Builtin::Abs | py_ast::Builtin::Sum |
        py_ast::Builtin::Prod | py_ast::Builtin::Any | py_ast::Builtin::All |
        py_ast::Builtin::Convert {..} | py_ast::Builtin::Label |
        py_ast::Builtin::GpuContext | py_ast::Builtin::Ext {..} => {
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
    mut shape: Vec<i64>,
    indices: Vec<Expr>,
    i: &Info
) -> CompileResult<Expr> {
    let int_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
    let zero = Expr::Int {v: 0, ty: int_ty.clone(), i: i.clone()};
    let nindices = indices.len();
    let tail = shape.split_off(nindices);
    let init = tail.into_iter().product();
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
            Ok((idx_expr, mult * n))
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
                        let idx = Box::new(flatten_indices(shape.clone(), indices, &i)?);
                        let res_shape = shape.clone()
                            .into_iter()
                            .skip(n)
                            .collect::<Vec<i64>>();
                        let res_ty = Type::Tensor {sz: sz.clone(), shape: res_shape};
                        let target = Box::new(target);
                        Ok(Expr::TensorAccess {target, idx, ty: res_ty, i})
                    } else {
                        let msg = concat!(
                            "Invalid dimensions. Indexing into tensor of shape ",
                            "{shape:?} using {n} indices"
                        );
                        parir_compile_error!(i, "{msg}")
                    }
                } else {
                    let msg = "Indexing into non-tensor target {target} is not supported";
                    parir_compile_error!(i, "{msg}")
                }
            }
        },
        py_ast::Expr::Slice {i, ..} => {
            parir_compile_error!(i, "Slices are not allowed outside of indexing")
        },
        py_ast::Expr::Tuple {i, ..} => {
            parir_compile_error!(i, "Tuple literals are not supported outside of indexing")
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

fn convert_par_spec(
    acc: Option<LoopParallelism>,
    p: Option<&Vec<ParKind>>
) -> Option<LoopParallelism> {
    p.unwrap_or(&vec![])
        .clone()
        .into_iter()
        .fold(acc, |acc, kind| match kind {
            ParKind::GpuThreads(n) => acc?.with_threads(n),
            ParKind::GpuReduction {} => Some(acc?.with_reduction())
        })
}

fn lookup_labels(
    par: &BTreeMap<String, Vec<ParKind>>,
    labels: Vec<String>,
    i: &Info
) -> CompileResult<LoopParallelism> {
    let par = labels.clone()
        .into_iter()
        .map(|l| par.get(&l))
        .fold(Some(LoopParallelism::default()), convert_par_spec);
    match par {
        Some(p) => Ok(p),
        None => {
            let labels = labels.into_iter().join(", ");
            let msg = format!(
                "The labels associated with this statement specify conflicting \
                 number of threads in parallelization.\n\
                 Labels of this statement: {labels}"
            );
            parir_compile_error!(i, "{}", msg)
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
        py_ast::Stmt::WithGpuContext {body, i} => {
            // NOTE: To ensure code within a GPU context actually runs on the GPU, we generate a
            // for-loop with a single iteration annotated to run in parallel on 1 thread.
            let var = Name::sym_str("_gpu_context");
            let int64_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
            let lo = Expr::Int {v: 0, ty: int64_ty.clone(), i: i.clone()};
            let hi = Expr::Int {v: 1, ty: int64_ty.clone(), i: i.clone()};
            let body = to_ir_stmts(env, body)?;
            let par = LoopParallelism::default().with_threads(1).unwrap();
            Ok(Stmt::For {var, lo, hi, step: 1, body, par, i})
        },
        py_ast::Stmt::Scope {i, ..} => {
            parir_compile_error!(i,
                "Found intermediate scope statement that should have been \
                 removed by the compiler"
            )
        },
        py_ast::Stmt::Call {func, i, ..} => {
            parir_compile_error!(i, "Found unsupported function call to {func}")
        },
        py_ast::Stmt::Label {i, ..} => {
            parir_compile_error!(i, "Found label without associated statement")
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

#[cfg(test)]
mod test {
    use super::*;

    fn i() -> Info {
        Info::default()
    }

    #[test]
    fn dict_subscript_to_ir() {
        let x = "x".to_string();
        let y = Name::sym_str("y");
        let elem_ty = py_ast::Type::Tensor {sz: ElemSize::I64, shape: vec![]};
        let mut fields = BTreeMap::new();
        fields.insert(x.clone(), elem_ty.clone());
        let ty = py_ast::Type::Dict {fields};
        let py_expr = py_ast::Expr::Subscript {
            target: Box::new(py_ast::Expr::Var {id: y.clone(), ty: ty.clone(), i: i()}),
            idx: Box::new(py_ast::Expr::String {
                v: x.clone(), ty: py_ast::Type::String, i: i()
            }),
            ty: elem_ty.clone(), i: i()
        };
        let y_struct = Name::sym_str("dict_y");
        let mut structs = BTreeMap::new();
        structs.insert(ty, y_struct.clone());
        let env = IREnv::new(structs, BTreeMap::new());
        let res = to_ir_expr(&env, py_expr).unwrap();
        assert_eq!(res, Expr::StructFieldAccess {
            target: Box::new(Expr::Var {
                id: y, ty: Type::Struct {id: y_struct.clone()}, i: i()
            }),
            label: x,
            ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]},
            i: i()
        });
    }

    fn py_tensor_ty(shape: Vec<i64>) -> py_ast::Type {
        py_ast::Type::Tensor {sz: ElemSize::I64, shape}
    }

    fn py_tuple_ty(nelems: usize) -> py_ast::Type {
        let elems = (0..nelems).map(|_| py_tensor_ty(vec![]))
            .collect::<Vec<py_ast::Type>>();
        py_ast::Type::Tuple {elems}
    }

    fn py_int(v: i64) -> py_ast::Expr {
        py_ast::Expr::Int {v, ty: py_tensor_ty(vec![]), i: i()}
    }

    fn ir_tensor_ty(shape: Vec<i64>) -> Type {
        Type::Tensor {sz: ElemSize::I64, shape}
    }

    fn ir_int_ty() -> Type {
        ir_tensor_ty(vec![])
    }

    fn ir_int(v: i64) -> Expr {
        Expr::Int {v, ty: ir_int_ty(), i: i()}
    }

    fn ir_binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type) -> Expr {
        Expr::BinOp {
            lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: i()
        }
    }

    #[test]
    fn nested_tensor_index_to_ir() {
        let x = Name::new("x".to_string());
        let py_expr = py_ast::Expr::Subscript {
            target: Box::new(py_ast::Expr::Subscript {
                target: Box::new(py_ast::Expr::Var {
                    id: x.clone(), ty: py_tensor_ty(vec![3,4,5,6]), i: i()
                }),
                idx: Box::new(py_ast::Expr::Tuple {
                    elems: vec![py_int(1), py_int(2)],
                    ty: py_tuple_ty(2),
                    i: i()
                }),
                ty: py_tensor_ty(vec![5,6]),
                i: i()
            }),
            idx: Box::new(py_int(3)),
            ty: py_tensor_ty(vec![6]),
            i: i()
        };
        let env = IREnv::new(BTreeMap::new(), BTreeMap::new());
        let res = to_ir_expr(&env, py_expr).unwrap();
        let idx_expr = ir_binop(
            ir_binop(ir_int(1), BinOp::Mul, ir_int(120), ir_int_ty()),
            BinOp::Add,
            ir_binop(
                ir_binop(ir_int(2), BinOp::Mul, ir_int(30), ir_int_ty()),
                BinOp::Add,
                ir_binop(
                    ir_binop(ir_int(3), BinOp::Mul, ir_int(6), ir_int_ty()),
                    BinOp::Add,
                    ir_int(0),
                    ir_int_ty()
                ),
                ir_int_ty()
            ),
            ir_int_ty()
        );
        assert_eq!(res, Expr::TensorAccess {
            target: Box::new(Expr::Var {
                id: x, ty: ir_tensor_ty(vec![3,4,5,6]), i: i()
            }),
            idx: Box::new(idx_expr),
            ty: Type::Tensor {sz: ElemSize::I64, shape: vec![6]},
            i: i()
        });
    }
}
