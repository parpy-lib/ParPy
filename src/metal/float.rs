// Converts uses of floating-point scalar values in a Metal kernel to 32-bit values as opposed to
// the 64-bit values used by default. This is required as Metal does not support 64-bit floats.
//
// Note that we do not convert pointers of 64-bit floats to 32-bits, as this would require
// reallocating and copying data. In this case, the compiler will report an error.

use crate::gpu::ast::*;
use crate::utils::smap::SMapAccum;

fn convert_floats_to_32bit_type(ty: Type) -> Type {
    match ty {
        Type::Scalar {sz: ElemSize::F64} => Type::Scalar {sz: ElemSize::F32},
        _ => ty
    }
}

fn convert_floats_to_32bit_kernel_expr(e: Expr) -> Expr {
    let ty = convert_floats_to_32bit_type(e.get_type().clone());
    let e = e.with_type(ty);
    e.smap(convert_floats_to_32bit_kernel_expr)
}

fn convert_floats_to_32bit_kernel_stmt(s: Stmt) -> Stmt {
    s.smap(convert_floats_to_32bit_kernel_stmt)
        .smap(convert_floats_to_32bit_kernel_expr)
}

fn convert_floats_to_32bit_launch_arg(arg: Expr) -> Expr {
    // If an argument passed to a kernel launch is a 64-bit floating-point value, we insert a cast
    // to a 32-bit value before passing it to the kernel.
    let is_64bit_scalar_arg = match arg.get_type() {
        Type::Scalar {sz: ElemSize::F64} => true,
        _ => false
    };
    if is_64bit_scalar_arg {
        Expr::Convert {e: Box::new(arg), ty: Type::Scalar {sz: ElemSize::F32}}
    } else {
        arg
    }
}

fn convert_floats_to_32bit_host_stmt(s: Stmt) -> Stmt {
    match s {
        Stmt::KernelLaunch {id, args, grid, i} => {
            let args = args.smap(convert_floats_to_32bit_launch_arg);
            Stmt::KernelLaunch {id, args, grid, i}
        },
        _ => s
    }
}

fn convert_floats_to_32bit_param(p: Param) -> Param {
    let Param {id, ty, i} = p;
    let ty = convert_floats_to_32bit_type(ty);
    Param {id, ty, i}
}

fn convert_floats_to_32bit_top(t: Top) -> Top {
    match t {
        Top::DeviceFunDef {threads, id, params, body} => {
            let params = params.smap(convert_floats_to_32bit_param);
            let body = body.smap(convert_floats_to_32bit_kernel_stmt);
            Top::DeviceFunDef {threads, id, params, body}
        },
        Top::HostFunDef {ret_ty, id, params, body} => {
            let body = body.smap(convert_floats_to_32bit_host_stmt);
            Top::HostFunDef {ret_ty, id, params, body}
        },
        Top::StructDef {..} => t
    }
}

pub fn convert_floats_to_32bit(ast: Ast) -> Ast {
    ast.smap(convert_floats_to_32bit_top)
}
