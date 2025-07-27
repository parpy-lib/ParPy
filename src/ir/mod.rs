pub mod ast;
mod constant_fold;
mod from_py_ast;
mod pprint;
mod struct_types;
mod tpb;

#[cfg(test)]
pub mod ast_builder;

use ast::*;
use crate::par::LoopPar;
use crate::par::REDUCE_PAR_LABEL;
use crate::py::ast as py_ast;
use crate::utils::debug::*;
use crate::utils::err::*;

use std::collections::BTreeMap;

pub fn from_python(
    ast: py_ast::Ast,
    mut par: BTreeMap<String, LoopPar>,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Insert the special label associated with a reduction into the parallelization mapping. This
    // is used in slicing involving reduction operations.
    par.insert(REDUCE_PAR_LABEL.to_string(), LoopPar::default().reduce());

    let structs = struct_types::find_dict_types(&ast).to_named_structs();
    let env = from_py_ast::IREnv::new(structs.clone(), par);
    let structs = structs.into_iter()
        .map(|(ty, id)| from_py_ast::to_struct_def(&env, id, ty))
        .collect::<CompileResult<Vec<StructDef>>>()?;
    let defs = from_py_ast::to_ir_defs(&env, ast)?;
    let ast = Ast {structs, defs};
    debug_env.print("Initial IR AST", &ast);
    let ast = tpb::propagate_configuration(ast)?;
    let ast = constant_fold::fold(ast);
    Ok(ast)
}

#[cfg(test)]
mod test {
    use crate::utils::err::*;

    use regex::Regex;
    use std::fmt;

    pub fn assert_error_matches<T: fmt::Debug>(r: CompileResult<T>, pat: &str) {
        let err_msg = format!("{}", r.unwrap_err());
        let re = Regex::new(pat).unwrap();
        assert!(
            re.is_match(&err_msg),
            "Error message \"{0}\" did not match expected pattern \"{1}\".",
            err_msg, pat
        );
    }
}
