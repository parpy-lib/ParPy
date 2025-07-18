pub mod ast;
mod buffers;
mod codegen;
mod pprint;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::gpu;
use crate::option;
use crate::gpu::flatten_structs;
use crate::utils::debug::*;
use crate::utils::err::*;

pub fn codegen(
    ir_ast: ir_ast::Ast,
    opts: &option::CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Convert the IR AST to a general GPU AST.
    let gpu_ast = gpu::from_general_ir(ir_ast, opts, debug_env)?;

    // Flatten struct types by replacing them by the individual fields, as the Metal backend does
    // not support the use of structs.
    let gpu_ast = flatten_structs::flatten_structs(gpu_ast)?;

    // Transforms the code such that scalar parameters of kernels are passed via temporary buffers
    // and treated as pointers inside kernel code.
    let gpu_ast = buffers::transform_scalars_to_buffers(gpu_ast);

    // Convert the GPU AST to a Metal AST.
    codegen::from_gpu_ir(gpu_ast)
}
