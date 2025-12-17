pub mod ast;
mod classify_functions;
mod constant_fold;
mod eliminate_gpu_context;
mod from_py_ast;
mod inter_block;
mod par_tree;
mod pprint;
mod target_constraints;
mod tpb;

#[cfg(test)]
pub mod ast_builder;

use ast::*;
pub use classify_functions::TargetClass;
use crate::option::CompileOptions;
use crate::par::LoopPar;
use crate::par::REDUCE_PAR_LABEL;
use crate::py::ast as py_ast;
use crate::utils::debug::*;
use crate::utils::err::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub fn from_python(
    ast: py_ast::Ast,
    opts: &CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<(Ast, BTreeMap<Name, TargetClass>)> {
    // Insert the special label associated with a reduction into the parallelization mapping. This
    // is used in slicing involving reduction operations.
    let mut par = opts.parallelize.clone();
    par.insert(REDUCE_PAR_LABEL.to_string(), LoopPar::default().par_reduction());
    let env = from_py_ast::IREnv::new(par, &opts);
    let ast = from_py_ast::to_ir_ast(env, ast)?;
    debug_env.print("Initial IR AST", &ast);

    let ast = eliminate_gpu_context::apply(ast)?;
    let ast = constant_fold::fold(ast);
    debug_env.print("IR AST after eliminating excessive for-loops", &ast);

    let mapping = target_constraints::collect(&ast)?;
    let ast = inter_block::restructure_inter_block_synchronization(ast, &mapping, &opts)?;
    let ast = tpb::propagate_configuration(ast)?;
    debug_env.print("IR AST after GPU inter-block transformation", &ast);

    let classification = classify_functions::apply(&ast, &mapping)?;

    Ok((ast, classification))
}
