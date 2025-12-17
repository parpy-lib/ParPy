use crate::py_internal_error;
use crate::gpu::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn to_ctype<'py>(ty: &Type, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let ctypes = py.import("ctypes")?;
    match ty {
        Type::Scalar {sz} => sz.to_ctype(py),
        Type::Pointer {ty, mem: _} => {
            // Construct a special function type if we have a pointer to a function. Otherwise, we
            // just use a void pointer type (this works regardless of the pointer type on the
            // receiving end).
            match ty.as_ref() {
                Type::Function {result, args} => {
                    let args = vec![to_ctype(&result, py)].into_iter()
                        .chain(args.iter().map(|arg| to_ctype(arg, py)))
                        .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()?;
                    ctypes.call_method1("CFUNCTYPE", PyTuple::new(py, args)?)
                },
                _ => ctypes.getattr("c_void_p"),
            }
        },
        Type::Void => Ok(py.None().into_bound(py)),
        _ => py_internal_error!(Info::default(), "Unsupported type")
    }
}

fn produce_argument_list_entry<'py>(
    entry: &Top,
    py: Python<'py>
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    match entry {
        Top::FunDef {params, ..} => {
            params.iter()
                .map(|p| to_ctype(&p.ty, py))
                .collect::<PyResult<Vec<Bound<'py, PyAny>>>>()
        },
        Top::ExtDecl {i, ..} |
        Top::KernelFunDef {i, ..} => {
            py_internal_error!(i, "Found unexpected entry node in callback handling")
        },
    }
}

pub fn produce_argument_list<'py>(
    ast: &Ast,
    py: Python<'py>
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    // The entry point should always be the last top-level declaration in the GPU AST.
    produce_argument_list_entry(ast.last().unwrap(), py)
}
