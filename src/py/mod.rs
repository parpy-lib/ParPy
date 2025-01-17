pub mod ast;
mod from_py;
mod type_check;

pub use from_py::to_untyped_ir;
pub use type_check::type_check_ast;

#[macro_export]
macro_rules! py_runtime_error {
    ($i:tt, $($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::compile_err($i.error_msg(format!($($t)*)))))
    }
}

#[macro_export]
macro_rules! py_type_error {
    ($i:tt, $($t:tt)*) => {
        Err(Into::<PyErr>::into(CompileError::type_err($i.error_msg(format!($($t)*)))))
    }
}
