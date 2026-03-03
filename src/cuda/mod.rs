pub mod ast;
mod clusters;
mod codegen;
mod constant_fold;
mod error;
mod escape_function_names;
mod graphs;
mod insert_stream_argument;
mod memory;
mod pprint;
mod power;
mod reduce;
mod shared_memory;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::gpu::ast as gpu_ast;
use crate::option;
use crate::utils::err::*;

pub fn codegen(
    gpu_ast: gpu_ast::Ast,
    opts: &option::CompileOptions
) -> CompileResult<Ast> {
    // Expand the abstract representations of warp and cluster reductions. We do this separately
    // from this codegen to avoid making it unnecessarily complex.
    let gpu_ast = reduce::expand_parallel_reductions(gpu_ast);

    // Ensure we perform no accesses into GPU memory from the host code.
    memory::validate_gpu_memory_access(&gpu_ast)?;

    // Convert the GPU AST to a CUDA C++ AST.
    let cuda_ast = codegen::from_gpu_ir(gpu_ast, opts)?;

    // Insert validation and proper configuration of shared memory.
    let cuda_ast = shared_memory::configure_ast(cuda_ast);

    // Adds a prefix to the names of all called functions and their definitions to avoid naming
    // collisions with existing functions.
    let cuda_ast = escape_function_names::apply(cuda_ast);

    // Inserts a stream argument to all entry points, and uses this stream in place of the default
    // stream.
    let cuda_ast = insert_stream_argument::apply(cuda_ast);

    // Replaces applicable uses of the power operator (pow) in CUDA with other operators. We do
    // this specifically for CUDA because its compiler does not optimize for special cases of the
    // operator, such as translating 'pow(x, 2.0)' to 'x * x' or 'pow(x, 0.5)' to 'sqrt(x)'.
    let cuda_ast = power::simplify_power_operator(cuda_ast);

    // Update all kernel entry points to make use of CUDA graphs.
    let cuda_ast = graphs::use_if_enabled(cuda_ast, opts);

    // Add an attribute to the kernels for which we use a non-standard amount of thread blocks per
    // cluster (currently, only up to 8 blocks are supported by default).
    let cuda_ast = clusters::insert_attribute_for_nonstandard_blocks_per_cluster(cuda_ast, opts);

    Ok(error::add_error_handling(cuda_ast))
}
