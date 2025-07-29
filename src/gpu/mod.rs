pub mod ast;
mod codegen;
mod constant_fold;
pub mod flatten_structs;
mod free_vars;
mod global_mem;
mod inter_block;
mod par;
mod par_tree;
mod pprint;
mod reduce;

#[cfg(test)]
pub mod ast_builder;

use ast::*;
use crate::option;
use crate::ir::ast as ir_ast;
use crate::utils::debug::*;
use crate::utils::err::*;

pub fn from_general_ir(
    ast: ir_ast::Ast,
    opts: &option::CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    let ast = inter_block::restructure_inter_block_synchronization(opts, ast)?;
    debug_env.print("IR AST after GPU inter-block transformation", &ast);

    // Identify the parallel structure in the IR AST and use this to determine how to map each
    // outermost parallel for-loop to the blocks and threads of a GPU kernel.
    let par = par::find_parallel_structure(&ast)?;
    let gpu_mapping = par::map_gpu_grid(par);

    // Translate the general IR AST to a representation used for all GPU targets.
    let ast = codegen::from_general_ir(opts, ast, gpu_mapping)?;
    debug_env.print("GPU AST", &ast);

    // Expand intermediate parallel reductions node to proper for-loops in the GPU IR AST.
    let ast = reduce::expand_parallel_reductions(opts, ast)?;

    // Transform memory writes where multiple threads write to the same location so that only one
    // thread writes and the threads are synchronized afterward.
    let ast = global_mem::eliminate_block_wide_memory_writes(ast)?;

    // Apply constant folding on the resulting AST before returning it.
    Ok(constant_fold::fold(ast))
}
