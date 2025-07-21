pub mod ast;
mod clusters;
mod codegen;
mod error;
mod graphs;
mod memory;
mod pprint;
mod reduce;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::gpu;
use crate::option;
use crate::utils::debug::*;
use crate::utils::err::*;

pub fn codegen(
    ir_ast: ir_ast::Ast,
    opts: &option::CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Convert the IR AST to a general GPU AST.
    let gpu_ast = gpu::from_general_ir(ir_ast, opts, debug_env)?;

    // Expand the abstract representations of warp and cluster reductions. We do this separately
    // from this codegen to avoid making it unnecessarily complex.
    let gpu_ast = reduce::expand_parallel_reductions(gpu_ast);

    // Ensure we perform no accesses into GPU memory from the host code.
    memory::validate_gpu_memory_access(&gpu_ast)?;

    // Convert the GPU AST to a CUDA C++ AST.
    let cuda_ast = codegen::from_gpu_ir(gpu_ast, opts)?;

    // Update all kernel entry points to make use of CUDA graphs.
    let cuda_ast = graphs::use_if_enabled(cuda_ast, opts);

    // Add an attribute to the kernels for which we use a non-standard amount of thread blocks per
    // cluster (currently, only up to 8 blocks are supported by default).
    let cuda_ast = clusters::insert_attribute_for_nonstandard_blocks_per_cluster(cuda_ast, opts);

    Ok(error::add_error_handling(cuda_ast))
}
