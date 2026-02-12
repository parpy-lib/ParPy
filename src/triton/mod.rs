mod ast;
mod codegen;
mod pprint;
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

    codegen::from_gpu_ast(gpu_ast)
}
