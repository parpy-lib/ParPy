pub mod ast;
mod from_py_ast;
mod struct_types;
mod symbolize;

use ast::*;
use symbolize::Symbolize;
use crate::par::ParKind;
use crate::py::ast as py_ast;
use crate::utils::err::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub fn from_python(
    def: py_ast::FunDef,
    par: BTreeMap<String, Vec<ParKind>>
) -> CompileResult<Ast> {
    let structs = struct_types::find_struct_types(&def).into_iter()
        .map(|ty| {
            let id = from_py_ast::generate_struct_name(&ty);
            (ty, id)
        })
        .collect::<BTreeMap<py_ast::Type, Name>>();
    let env = from_py_ast::IREnv::new(structs.clone(), par);
    let structs = structs.into_iter()
        .map(|(ty, id)| from_py_ast::to_struct_def(&env, id, ty))
        .collect::<CompileResult<Vec<StructDef>>>()?;
    let fun = from_py_ast::to_ir_def(&env, def)?;
    Ok(Ast {structs, fun})
}

pub fn symbolize(ast: Ast) -> CompileResult<Ast> {
    let (_, ast) = ast.symbolize_default()?;
    Ok(ast)
}
