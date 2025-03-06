pub mod ast;
mod codegen;
mod constant_fold;
mod free_vars;
mod global_mem;
mod pprint;
mod par;
mod sync;

mod new_sync;
mod par_tree;

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

    // Translate the IR AST to a CUDA C++ AST based on the information gathered above.
    let ast = codegen::from_ir(ast, gpu_mapping, sync)?;

    // Eliminates all block-wide memory writes to the same memory address by having only one thread
    // write the value, and then synchronizing to ensure all threads have a consistent view of the
    // memory afterward.
    let ast = global_mem::eliminate_block_wide_memory_writes(ast)?;

    // Perform simple constant folding to produce more readable output.
    Ok(constant_fold::fold(ast))
}
