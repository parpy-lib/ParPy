
use super::ast::*;
use super::par::{GpuMap, GpuMapping};
use crate::ir::ast as ir_ast;
use crate::utils::err::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

pub fn from_general_ir(
    ir_ast: ir_ast::Ast,
    gpu_mapping: BTreeMap<Name, GpuMapping>
) -> CompileResult<Ast> {
    todo!()
}
