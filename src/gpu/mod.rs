pub mod ast;
mod codegen;
mod collect_shared_memory;
mod constant_fold;
mod free_vars;
mod fuse_memory;
mod global_mem;
mod par;
mod pprint;
pub mod reduce;
mod split_function_targets;
mod sync_elim;
mod unroll_loops;

#[cfg(test)]
pub mod ast_builder;
#[cfg(test)]
pub mod unsymbolize;

use ast::*;
use crate::option::CompileOptions;
use crate::ir::ast as ir_ast;
use crate::ir::TargetClass;
use crate::utils::debug::*;
use crate::utils::err::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub fn from_general_ir(
    ast: ir_ast::Ast,
    classification: BTreeMap<Name, TargetClass>,
    opts: &CompileOptions
) -> CompileResult<Ast> {
    // Identify the parallel structure in the IR AST and use this to determine how to map each
    // outermost parallel for-loop to the blocks and threads of a GPU kernel.
    let par = par::find_parallel_structure(&ast)?;
    let gpu_mapping = par::map_gpu_grid(par);

    // Functions assigned TargetClass::Both are split into two separate versions, depending on
    // whether they are called from the host or from the device.
    let (ast, classification) = split_function_targets::apply(ast, classification)?;

    // Translate the general IR AST to a representation used for all GPU targets.
    let ast = codegen::from_general_ir(ast, classification, gpu_mapping, opts)?;
    let ast = constant_fold::fold(ast);

    // Eliminate redundant uses of synchronization. This includes repeated uses of synchronization
    // on the same scope and trailing synchronization at the end of a kernel.
    Ok(sync_elim::remove_redundant_synchronization(ast))
}

pub fn transform(
    ast: Ast,
    opts: &CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Expand intermediate parallel reductions node to proper for-loops in the GPU IR AST.
    let ast = reduce::expand_parallel_reductions(opts, ast)?;
    debug_env.print("GPU AST after expanding reductions", &ast);

    // Unroll simple for-loops that contain a single statement and that do not perform more than a
    // fixed number of iterations in total, as determined via a compiler option.
    let ast = unroll_loops::apply(ast, opts);
    debug_env.print("GPU AST after unrolling for-loops", &ast);

    // Attempts to fuse memory operations performed in the AST to reduce the memory bandwidth usage
    // of a kernel.
    let ast = fuse_memory::apply(ast)?;
    debug_env.print("GPU AST after fusing memory operations", &ast);

    // Collect use of shared memory within each kernel or device function. Based on this
    // information, we encode the amount of required dynamic shared memory in each kernel node, and
    // determine the offset of each shared memory use. Also, this pass ensures shared memory is
    // never used outside device code.
    let ast = collect_shared_memory::apply(ast)?;
    debug_env.print("GPU AST after collecting uses of shared memory in kernels", &ast);

    // Transform memory writes where multiple threads write to the same location so that only one
    // thread writes and the threads are synchronized afterward.
    let ast = global_mem::eliminate_block_wide_memory_writes(ast)?;
    debug_env.print("GPU AST after eliminating block-wide memory writes", &ast);

    Ok(constant_fold::fold(ast))
}
