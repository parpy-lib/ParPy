use super::extract_callbacks::Callback;
use crate::parpy_internal_error;
use crate::gpu::ast as gpu_ast;
use crate::option::{CompileBackend, CompileOptions};
use crate::py::ast as py_ast;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

fn gpu_ast_to_py_type(ty: gpu_ast::Type) -> py_ast::Type {
    match ty {
        gpu_ast::Type::Void => py_ast::Type::Void,
        gpu_ast::Type::Scalar {sz} => {
            let sz = py_ast::TensorElemSize::Fixed {sz};
            py_ast::Type::Tensor {sz, shape: vec![]}
        },
        _ => py_ast::Type::Unknown
    }
}

fn gpu_ast_to_py_expr(e: gpu_ast::Expr) -> CompileResult<py_ast::Expr> {
    let ty = gpu_ast_to_py_type(e.get_type().clone());
    match e {
        gpu_ast::Expr::Var {id, ty: _, i} => Ok(py_ast::Expr::Var {id, ty, i}),
        gpu_ast::Expr::Bool {v, ty: _, i} => Ok(py_ast::Expr::Bool {v, ty, i}),
        gpu_ast::Expr::Int {v, ty: _, i} => Ok(py_ast::Expr::Int {v, ty, i}),
        gpu_ast::Expr::Float {v, ty: _, i} => Ok(py_ast::Expr::Float {v, ty, i}),
        gpu_ast::Expr::UnOp {op, arg, ty: _, i} => {
            let arg = Box::new(gpu_ast_to_py_expr(*arg)?);
            Ok(py_ast::Expr::UnOp {op, arg, ty, i})
        },
        gpu_ast::Expr::BinOp {lhs, op, rhs, ty: _, i} => {
            let lhs = Box::new(gpu_ast_to_py_expr(*lhs)?);
            let rhs = Box::new(gpu_ast_to_py_expr(*rhs)?);
            Ok(py_ast::Expr::BinOp {lhs, op, rhs, ty, i})
        },
        gpu_ast::Expr::IfExpr {cond, thn, els, ty: _, i} => {
            let cond = Box::new(gpu_ast_to_py_expr(*cond)?);
            let thn = Box::new(gpu_ast_to_py_expr(*thn)?);
            let els = Box::new(gpu_ast_to_py_expr(*els)?);
            Ok(py_ast::Expr::IfExpr {cond, thn, els, ty, i})
        },
        gpu_ast::Expr::ArrayAccess {target, idx, ty: _, i} => {
            let target = Box::new(gpu_ast_to_py_expr(*target)?);
            let idx = Box::new(gpu_ast_to_py_expr(*idx)?);
            Ok(py_ast::Expr::Subscript {target, idx, ty, i})
        },
        _ => {
            let i = e.get_info();
            parpy_internal_error!(i, "Failed to convert expression in callback generation")
        }
    }
}

fn extract_upper_bound(cond: gpu_ast::Expr) -> CompileResult<py_ast::Expr> {
    let i = cond.get_info();
    if let gpu_ast::Expr::BinOp {op: gpu_ast::BinOp::Lt, rhs, ..} = cond {
        gpu_ast_to_py_expr(*rhs)
    } else {
        parpy_internal_error!(i, "Failed to extract upper-bound in callback generation")
    }
}

fn extract_step_size(incr: gpu_ast::Expr) -> CompileResult<i64> {
    let i = incr.get_info();
    if let gpu_ast::Expr::BinOp {op: gpu_ast::BinOp::Add, rhs, ..} = incr {
        if let gpu_ast::Expr::Int {v, ..} = *rhs {
            Ok(v as i64)
        } else {
            parpy_internal_error!(i, "Failed to extract RHS of loop step size \
                                      in callback generation")
        }
    } else {
        parpy_internal_error!(i, "Failed to extract loop step size in callback generation")
    }
}

fn extract_gpu_ast_loop_bounds(
    init: gpu_ast::Expr,
    cond: gpu_ast::Expr,
    incr: gpu_ast::Expr
) -> CompileResult<(py_ast::Expr, py_ast::Expr, i64)> {
    let lo = gpu_ast_to_py_expr(init)?;
    let hi = extract_upper_bound(cond)?;
    let step = extract_step_size(incr)?;
    Ok((lo, hi, step))
}

fn gpu_ast_to_py_stmt(s: gpu_ast::Stmt) -> CompileResult<py_ast::Stmt> {
    match s {
        gpu_ast::Stmt::For {var_ty: _, var, init, cond, incr, body, unroll: _, i} => {
            let (lo, hi, step) = extract_gpu_ast_loop_bounds(init, cond, incr)?;
            let body = body.into_iter()
                .map(|s| gpu_ast_to_py_stmt(s))
                .collect::<CompileResult<Vec<py_ast::Stmt>>>()?;
            Ok(py_ast::Stmt::For {
                var,
                lo,
                hi,
                step: py_ast::Expr::Int {
                    v: step as i128,
                    ty: py_ast::Type::Unknown,
                    i: i.clone()
                },
                body,
                labels: vec![],
                i
            })
        },
        gpu_ast::Stmt::Expr {e: gpu_ast::Expr::PyCallback {id, args, i, ..}, i: stmt_i} => {
            Ok(py_ast::Stmt::Expr {
                e: py_ast::Expr::Call {id, args, ty: py_ast::Type::Void, i},
                i: stmt_i
            })
        },
        _ => {
            parpy_internal_error!(
                Info::default(),
                "Failed to generate Python-side callback wrapper."
            )
        }
    }
}

fn is_non_scalar_tensor_param(p: &py_ast::Param) -> bool {
    match &p.ty {
        py_ast::Type::Tensor {shape, ..} => !shape.is_empty(),
        _ => false
    }
}

fn generate_shape_expr(sh: &py_ast::TensorShape) -> CompileResult<py_ast::Expr> {
    if let py_ast::TensorShape::Num {n} = sh {
        Ok(py_ast::Expr::Int {v: *n as i128, ty: py_ast::Type::Unknown, i: Info::default()})
    } else {
        parpy_internal_error!(Info::default(), "Found unresolved shape dimension \
                                                in callback wrapper codegen.")
    }
}

fn generate_shape_list(ty: &py_ast::Type) -> CompileResult<py_ast::Expr> {
    if let py_ast::Type::Tensor {shape, ..} = ty {
        let shape_exprs = shape.iter()
            .map(generate_shape_expr)
            .collect::<CompileResult<Vec<py_ast::Expr>>>()?;
        Ok(py_ast::Expr::List {
            elems: shape_exprs, ty: py_ast::Type::Unknown, i: Info::default()
        })
    } else {
        parpy_internal_error!(Info::default(), "Found invalid parameter type {ty} \
                                                in callback wrapper codegen.")
    }
}

fn generate_param_element_type(ty: py_ast::Type) -> CompileResult<py_ast::Expr> {
    if let py_ast::Type::Tensor {sz: py_ast::TensorElemSize::Fixed {sz}, ..} = ty {
        let id = Name::new(format!("parpy.types.{sz:?}"));
        Ok(py_ast::Expr::Var {id, ty: py_ast::Type::Unknown, i: Info::default()})
    } else {
        parpy_internal_error!(Info::default(), "Found invalid parameter type {ty} \
                                                in callback wrapper codegen.")
    }
}

fn generate_backend_var(backend: &CompileBackend) -> py_ast::Expr {
    let id = match backend {
        CompileBackend::Cuda => "parpy.CompileBackend.Cuda",
        CompileBackend::Metal => "parpy.CompileBackend.Metal",
        CompileBackend::Triton => "parpy.CompileBackend.Triton",
        _ => "None"
    };
    py_ast::Expr::Var {
        id: Name::new(id.to_string()), ty: py_ast::Type::Unknown, i: Info::default()
    }
}

fn generate_parameter_construction(
    opts: &CompileOptions,
    p: py_ast::Param
) -> CompileResult<py_ast::Stmt> {
    let var = py_ast::Expr::Var {
        id: p.id.clone(), ty: p.ty.clone(), i: p.i.clone()
    };
    let shape_list = generate_shape_list(&p.ty)?;
    let parpy_elem_type = generate_param_element_type(p.ty.clone())?;
    let backend = generate_backend_var(&opts.backend);
    let from_raw_args = vec![
        var.clone(),
        shape_list,
        parpy_elem_type,
        backend
    ];
    let parpy_buffer_from_raw = py_ast::Expr::Call {
        id: Name::new(format!("parpy.buffer.from_raw")),
        args: from_raw_args,
        ty: p.ty,
        i: p.i.clone()
    };
    Ok(py_ast::Stmt::Assign {
        dst: var,
        expr: parpy_buffer_from_raw,
        labels: vec![],
        i: Info::default()
    })
}

fn generate_callback_body(
    opts: &CompileOptions,
    params: Vec<py_ast::Param>,
    body: gpu_ast::Stmt
) -> CompileResult<Vec<py_ast::Stmt>> {
    params.into_iter()
        .filter(is_non_scalar_tensor_param)
        .map(|p| generate_parameter_construction(&opts, p))
        .chain(vec![gpu_ast_to_py_stmt(body)].into_iter())
        .collect::<CompileResult<Vec<py_ast::Stmt>>>()
}

fn strip_parameter_type(p: py_ast::Param) -> py_ast::Param {
    py_ast::Param {ty: py_ast::Type::Unknown, ..p}
}

fn strip_parameter_types(params: Vec<py_ast::Param>) -> Vec<py_ast::Param> {
    params.into_iter()
        .map(strip_parameter_type)
        .collect::<Vec<py_ast::Param>>()
}

fn generate_callback_function(
    opts: &CompileOptions,
    callback: Callback
) -> CompileResult<py_ast::Ast> {
    let callback_id = Name::new(format!("{}_wrapper", callback.id)).with_new_sym();
    let callback_body = generate_callback_body(&opts, callback.params.clone(), callback.body)?;
    // NOTE(larshum 2025-10-08): This generated callback function receives raw C pointers and
    // converts them into objects usable from Python. To avoid confusion, we strip the annotated
    // types of the parameters as they are not accurate with this in mind.
    let callback_params = strip_parameter_types(callback.params);
    let main = py_ast::FunDef {
        id: callback_id,
        params: callback_params,
        body: callback_body,
        res_ty: py_ast::Type::Void,
        i: Info::default()
    };
    Ok(py_ast::Ast {tops: vec![], main})
}

pub fn generate_callbacks(
    opts: &CompileOptions,
    callbacks: Vec<Callback>
) -> CompileResult<Vec<py_ast::Ast>> {
    callbacks.into_iter()
        .map(|cb| generate_callback_function(&opts, cb))
        .collect::<CompileResult<Vec<py_ast::Ast>>>()
}
