pub mod ast;
pub mod pprint;
mod par;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::utils::err::*;

pub fn codegen(ast: ir_ast::Ast) -> CompileResult<Ast> {
    let par = par::find_parallel_structure(&ast)?;
    let gpu_mapping = par::map_gpu_grid(par);

    // Remaining parts for codegen:
    // * Ensure no barriers in code mapped to blocks (synchronization is hard there); in tandem
    //   with this, we also need to record where to insert __syncthreads in the output program
    // * Use the mapping to translate the AST to CUDA
    // * Pretty-print the CUDA AST (this part is already implemented)
    Ok(vec![])
}
