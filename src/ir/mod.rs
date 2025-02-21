pub mod ast;
mod constant_fold;
mod from_py_ast;
mod pprint;
mod struct_types;

use ast::*;
use crate::par::ParKind;
use crate::par::REDUCE_PAR_LABEL;
use crate::py::ast as py_ast;
use crate::utils::err::*;

use std::collections::BTreeMap;

pub fn from_python(
    def: py_ast::FunDef,
    mut par: BTreeMap<String, Vec<ParKind>>
) -> CompileResult<Ast> {
    // Insert the special label associated with a reduction into the parallelization mapping. This
    // is used in slicing involving reduction operations.
    par.insert(REDUCE_PAR_LABEL.to_string(), vec![ParKind::GpuReduction {}]);

    let structs = struct_types::find_dict_types(&def).to_named_structs();
    let env = from_py_ast::IREnv::new(structs.clone(), par);
    let structs = structs.into_iter()
        .map(|(ty, id)| from_py_ast::to_struct_def(&env, id, ty))
        .collect::<CompileResult<Vec<StructDef>>>()?;
    let fun = from_py_ast::to_ir_def(&env, def)?;
    let ast = Ast {structs, fun};
    let ast = constant_fold::fold(ast);
    Ok(ast)
}
