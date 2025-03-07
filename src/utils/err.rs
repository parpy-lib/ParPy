use pyo3::PyErr;
use pyo3::exceptions::*;

use std::error;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
enum ErrorKind {
    Compile,
    Name,
    Type
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorKind::Compile => write!(f, "Parir compilation error"),
            ErrorKind::Name => write!(f, "Parir name error"),
            ErrorKind::Type => write!(f, "Parir type error"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    msg: String,
    kind: ErrorKind
}

impl CompileError {
    pub fn compile_err(msg: String) -> Self {
        CompileError {msg, kind: ErrorKind::Compile}
    }

    pub fn name_err(msg: String) -> Self {
        CompileError {msg, kind: ErrorKind::Name}
    }

    pub fn type_err(msg: String) -> Self {
        CompileError {msg, kind: ErrorKind::Type}
    }
}

impl error::Error for CompileError {}
impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{0}: {1}", self.kind, &self.msg)
    }
}

pub type CompileResult<T> = Result<T, CompileError>;

#[macro_export]
macro_rules! parir_compile_error {
    ($i:expr,$($t:tt)*) => {{
        Err(CompileError::compile_err($i.error_msg(format!($($t)*))))
    }}
}

#[macro_export]
macro_rules! parir_type_error {
    ($i:expr,$($t:tt)*) => {{
        Err(CompileError::type_err($i.error_msg(format!($($t)*))))
    }}
}

impl From<CompileError> for PyErr {
    fn from(err: CompileError) -> PyErr {
        match err.kind {
            ErrorKind::Compile => PyRuntimeError::new_err(err.msg),
            ErrorKind::Name => PyNameError::new_err(err.msg),
            ErrorKind::Type => PyTypeError::new_err(err.msg),
        }
    }
}
