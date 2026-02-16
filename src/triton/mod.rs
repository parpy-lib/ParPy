pub mod ast;
mod codegen;
mod inline;
mod pprint;
//mod shapes;
mod rewrite_reductions;

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

    // Performs inlining within the GPU code such that all GPU kernels consist of one function
    // without performing any function calls.
    let ast = inline::apply(ast)?;

    Ok(ast)
}
