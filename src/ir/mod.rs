pub mod ast;
mod constant_fold;
mod from_py_ast;
mod pprint;
mod struct_types;
mod tpb;

#[cfg(test)]
pub mod ast_builder;

use ast::*;
use crate::option::CompileOptions;
use crate::par::LoopPar;
use crate::par::REDUCE_PAR_LABEL;
use crate::py::ast as py_ast;
use crate::utils::debug::*;
use crate::utils::err::*;

pub fn from_python(
    ast: py_ast::Ast,
    opts: &CompileOptions,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Insert the special label associated with a reduction into the parallelization mapping. This
    // is used in slicing involving reduction operations.
    let mut par = opts.parallelize.clone();
    par.insert(REDUCE_PAR_LABEL.to_string(), LoopPar::default().reduce());
    let structs = struct_types::find_dict_types(&ast).to_named_structs();
    let env = from_py_ast::IREnv::new(structs.clone(), par, &opts);
    let structs = structs.into_iter()
        .map(|(ty, id)| from_py_ast::to_struct_def(&env, id, ty))
        .collect::<CompileResult<Vec<Top>>>()?;
    let ast = from_py_ast::to_ir_ast(env, ast, structs)?;
    debug_env.print("Initial IR AST", &ast);
    let ast = tpb::propagate_configuration(ast)?;
    Ok(constant_fold::fold(ast))
}
