pub mod ast;
mod buffers;
mod codegen;
mod error;
mod pprint;
mod threadgroup;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::*;

pub fn codegen(gpu_ast: gpu_ast::Ast) -> CompileResult<Ast> {
    // Transforms the code such that scalar parameters of kernels are passed via temporary buffers
    // and treated as pointers inside kernel code.
    let gpu_ast = buffers::transform_scalars_to_buffers(gpu_ast);

    // Convert the GPU AST to a Metal AST.
    let metal_ast = codegen::from_gpu_ir(gpu_ast)?;

    // Insert parameters containing threadgroup local memory for applicable kernels.
    let metal_ast = threadgroup::configure_ast(metal_ast);

    Ok(error::add_error_handling(metal_ast))
}
