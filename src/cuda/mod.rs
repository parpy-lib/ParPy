pub mod ast;
pub mod pprint;
mod par;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::utils::err::*;

pub fn codegen(ast: ir_ast::Ast) -> CompileResult<Ast> {
    let par = par::find_parallel_structure(&ast);

    // Remaining parts for codegen:
    // * Map each entry of the parallel structure to threads or blocks (sth like Vec<(i64, ThreadBlock)>)
    // * Ensure no barriers in code mapped to blocks (synchronization is hard there)
    // * Use the mapping to translate the AST to CUDA
    // * Pretty-print the CUDA AST (this part is already implemented)
    Ok(vec![])
}
