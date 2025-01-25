mod ast;
mod codegen;
mod free_vars;
mod pprint;
mod par;
mod sync;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::utils::err::*;

pub fn codegen(ast: ir_ast::Ast) -> CompileResult<Ast> {
    // Identify the parallel structure in the IR AST and use this to determine how to map each
    // outermost parallel for-loop to a GPU kernel, and specifically, how to parallelize each
    // for-loop with respect to the threads and blocks of the GPU.
    let par = par::find_parallel_structure(&ast)?;
    let gpu_mapping = par::map_gpu_grid(par);

    // Record which for-loops require synchronization and ensure that this does not result in an
    // inter-block synchronization (which is not supported).
    let sync = sync::identify_sync_points(&ast, &gpu_mapping)?;

    codegen::from_ir(ast, gpu_mapping, sync)
}
