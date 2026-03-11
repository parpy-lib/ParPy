pub mod ast;
mod blocking;
mod codegen;
mod constant_fold;
mod eliminate_thread_for_loops;
mod fuse_memory;
mod inline;
pub mod native_ast;
mod native_codegen;
mod power;
mod pprint;
mod remove_sync;
mod rewrite_reductions;
mod shapes;
mod tuning;
mod utils;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::option::CompileOptions;
use crate::gpu::ast as gpu_ast;
use crate::utils::debug::*;
use crate::utils::err::CompileResult;

pub fn codegen(
    gpu_ast: gpu_ast::Ast,
    opts: &CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Remove synchronization points after reductions, as Triton will ensure threads are
    // synchronized after such operations.
    let gpu_ast = remove_sync::apply(gpu_ast)?;

    // Rewrite reductions such that the intermediate result is always stored in a (fresh) temporary
    // variable, and written once to the left-hand side of the original reduction.
    let gpu_ast = rewrite_reductions::apply(gpu_ast)?;

    // Convert the GPU AST to an AST representing the Triton code.
    let ast = codegen::from_gpu_ast(gpu_ast, &opts)?;
    debug_env.print("Initial Triton AST", &ast);

    // Apply constant folding to eliminate unnecessary expressions.
    let ast = constant_fold::apply(ast);

    // Performs inlining within the GPU code such that all GPU kernels consist of one function
    // without performing any function calls.
    let ast = inline::apply(ast)?;
    debug_env.print("Triton AST after inlining", &ast);

    // Attempts to unify the block-wide shapes of all expressions in each GPU kernel, to ensure
    // proper tracking of block-wide operations.
    let ast = shapes::unify(ast)?;

    // Transforms the code within each GPU kernel to use a blocking structure. In particular, we
    // rewrite control-flow statements whose condition depends on a block-wide value to a format
    // supported by Triton.
    let ast = blocking::transform(ast)?;
    debug_env.print("Triton AST after inserting blocking", &ast);

    // Simplifies uses of the power operator where the exponent is a known value by rewriting it to
    // use multiplications and square root.
    let ast = power::simplify_power_operator(ast);

    // Eliminates for-loops over threads with known bounds by rewriting them as a single blocked
    // operation. Rather than iteratively construct several smaller blocks, the updated code
    // constructs one big block on which it performs all updates. This has a significant impact on
    // performance.
    let ast = eliminate_thread_for_loops::apply(ast)?;
    debug_env.print("Triton AST after eliminating thread-loops", &ast);

    // Apply another round of constant folding to simplify the later analyses.
    let ast = constant_fold::apply(ast);

    // Apply a fusion of memory operations, where excess loads and stores to the same static memory
    // location are replaced by storage of data in temporary variables. This reduces the number of
    // reads and writes to memory, which can help Triton generate significantly more efficient
    // code.
    let ast = fuse_memory::apply(ast)?;

    // Run a final round of constant folding before returning the resulting AST.
    let ast = constant_fold::apply(ast);

    // Adds use of autotuning via Triton into the AST, so that it automatically selects between a
    // few different block sizes rather than using a one-to-one mapping between block size and
    // thread count.
    let ast = tuning::apply(ast, &opts);

    Ok(ast)
}

pub fn generate_native_entry_point(ast: &gpu_ast::Ast) -> CompileResult<native_ast::Ast> {
    native_codegen::from_gpu_ast(ast)
}
