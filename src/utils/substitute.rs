use crate::gpu::ast as gpu_ast;
use crate::ir::ast as ir_ast;
use crate::py::ast as py_ast;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

pub type SubEnv<T> = BTreeMap<Name, T>;

pub trait SubVars<T> {
    fn sub_vars(self, env: &SubEnv<T>) -> Self where Self: Sized;
}

impl SubVars<Name> for py_ast::Expr {
    fn sub_vars(self, env: &SubEnv<Name>) -> py_ast::Expr {
        match self {
            py_ast::Expr::Var {id, ty, i} if env.contains_key(&id) => {
                let new_id = env.get(&id).unwrap().clone();
                py_ast::Expr::Var {id: new_id, ty, i}
            },
            _ => self.smap(|e| e.sub_vars(env))
        }
    }
}

impl SubVars<py_ast::Expr> for py_ast::Expr {
    fn sub_vars(self, env: &SubEnv<py_ast::Expr>) -> py_ast::Expr {
        match self {
            py_ast::Expr::Var {id, i, ..} if env.contains_key(&id) => {
                let e = env.get(&id).unwrap().clone();
                e.with_info(i)
            },
            py_ast::Expr::Var {..} | py_ast::Expr::String {..} | py_ast::Expr::Bool {..} |
            py_ast::Expr::Int {..} | py_ast::Expr::Float {..} | py_ast::Expr::UnOp {..} |
            py_ast::Expr::BinOp {..} | py_ast::Expr::ReduceOp {..} | py_ast::Expr::IfExpr {..} |
            py_ast::Expr::Subscript {..} | py_ast::Expr::Slice {..} | py_ast::Expr::Tuple {..} |
            py_ast::Expr::List {..} | py_ast::Expr::Call {..} | py_ast::Expr::Callback {..} |
            py_ast::Expr::Convert {..} | py_ast::Expr::GpuContext {..} |
            py_ast::Expr::Inline {..} | py_ast::Expr::Label {..} |
            py_ast::Expr::StaticBackendEq {..} | py_ast::Expr::StaticTypesEq {..} |
            py_ast::Expr::StaticFail {..} | py_ast::Expr::AllocShared {..} =>  {
                self.smap(|e| e.sub_vars(env))
            }
        }
    }
}

impl SubVars<py_ast::Expr> for py_ast::Stmt {
    fn sub_vars(self, env: &SubEnv<py_ast::Expr>) -> py_ast::Stmt {
        let s = self.smap(|s: py_ast::Stmt| s.sub_vars(env));
        s.smap(|e: py_ast::Expr| e.sub_vars(env))
    }
}

impl SubVars<Name> for ir_ast::Expr {
    fn sub_vars(self, env: &SubEnv<Name>) -> ir_ast::Expr {
        match self {
            ir_ast::Expr::Var {id, ty, i} if env.contains_key(&id) => {
                let new_id = env.get(&id).unwrap().clone();
                ir_ast::Expr::Var {id: new_id, ty, i}
            },
            ir_ast::Expr::PyCallback {id, args, ty, i} => {
                let id = env.get(&id).cloned().unwrap_or(id);
                let args = args.smap(|e: py_ast::Expr| e.sub_vars(env));
                ir_ast::Expr::PyCallback {id, args, ty, i}
            },
            _ => self.smap(|e| e.sub_vars(env))
        }
    }
}

impl SubVars<Name> for ir_ast::Stmt {
    fn sub_vars(self, env: &SubEnv<Name>) -> ir_ast::Stmt {
        let s = self.smap(|s: ir_ast::Stmt| s.sub_vars(env));
        s.smap(|e: ir_ast::Expr| e.sub_vars(env))
    }
}

impl SubVars<gpu_ast::Expr> for gpu_ast::Expr {
    fn sub_vars(self, env: &SubEnv<gpu_ast::Expr>) -> gpu_ast::Expr {
        match self {
            gpu_ast::Expr::Var {id, ..} if env.contains_key(&id) => {
                env.get(&id).unwrap().clone()
            },
            _ => self.smap(|e| e.sub_vars(env))
        }
    }
}

impl SubVars<gpu_ast::Expr> for gpu_ast::Stmt {
    fn sub_vars(self, env: &SubEnv<gpu_ast::Expr>) -> gpu_ast::Stmt {
        let s = self.smap(|s: gpu_ast::Stmt| s.sub_vars(env));
        s.smap(|e: gpu_ast::Expr| e.sub_vars(env))
    }
}
