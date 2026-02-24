pub mod ast;
mod blocking;
mod codegen;
mod constant_fold;
mod inline;
mod pprint;
mod shapes;
mod rewrite_reductions;
mod utils;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::CompileResult;

pub fn codegen(gpu_ast: gpu_ast::Ast) -> CompileResult<Ast> {
    // Rewrite reductions such that the intermediate result is always stored in a (fresh) temporary
    // variable, and written once to the left-hand side of the original reduction.
    let gpu_ast = rewrite_reductions::apply(gpu_ast)?;

    // Convert the GPU AST to an AST representing the Triton code.
    let ast = codegen::from_gpu_ast(gpu_ast)?;

    // Apply constant folding to eliminate unnecessary expressions.
    let ast = constant_fold::apply(ast);

    // Performs inlining within the GPU code such that all GPU kernels consist of one function
    // without performing any function calls.
    let ast = inline::apply(ast)?;

    // Attempts to unify the block-wide shapes of all expressions in each GPU kernel, to ensure
    // proper tracking of block-wide operations.
    let ast = shapes::unify(ast)?;

    // Transforms the code within each GPU kernel to use a blocking structure. In particular, we
    // rewrite control-flow statements whose condition depends on a block-wide value to a format
    // supported by Triton.
    let ast = blocking::transform(ast)?;

    Ok(ast)
}
