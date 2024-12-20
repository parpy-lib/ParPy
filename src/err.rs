use pyo3::PyErr;
use pyo3::exceptions::*;

use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
enum ErrorKind {
    Runtime,
    Type
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    msg : String,
    kind : ErrorKind
}

impl CompileError {
    pub fn runtime_err(msg: String) -> Self {
        CompileError {msg, kind: ErrorKind::Runtime}
    }

    pub fn type_err(msg: String) -> Self {
        CompileError {msg, kind: ErrorKind::Type}
    }
}

impl error::Error for CompileError {}
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parir compile error: {0}", &self.msg)
    }
}

pub type CompileResult<T> = Result<T, CompileError>;

#[macro_export]
macro_rules! parir_runtime_error {
    ($i:tt,$($t:tt)*) => {{
        Err(CompileError::runtime_err($i.error_msg(format!($($t)*))))
    }}
}

#[macro_export]
macro_rules! parir_type_error {
    ($i:tt,$($t:tt)*) => {{
        Err(CompileError::type_err($i.error_msg(format!($($t)*))))
    }}
}

impl From<CompileError> for PyErr {
    fn from(err: CompileError) -> PyErr {
        match err.kind {
            ErrorKind::Runtime => PyRuntimeError::new_err(err.msg),
            ErrorKind::Type => PyTypeError::new_err(err.msg)
        }
    }
}
