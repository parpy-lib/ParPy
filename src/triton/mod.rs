mod ast;
mod codegen;
mod pprint;

#[cfg(test)]
mod ast_builder;

use ast::*;
use crate::gpu::ast as gpu_ast;
use crate::utils::err::CompileResult;

pub fn codegen(ast: gpu_ast::Ast) -> CompileResult<Ast> {
    codegen::from_gpu_ast(ast)
}
