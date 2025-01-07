use crate::err::*;
use super::ast::*;

/// This function attempts to infer types where they are missing, to eliminate all occurrences of
/// unknown types and of unspecified integer and floating-point sizes.
pub fn infer_types(ast: Ast) -> CompileResult<Ast> {
    Ok(ast)
}
