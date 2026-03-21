use crate::gpu::ast as gpu_ast;
use crate::py::ast as py_ast;
use crate::triton::ast as triton_ast;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

pub type Env = BTreeMap<Name, Name>;

pub trait NormalizeSym {
    fn normalize_sym(self, env: Env) -> (Env, Self);

    fn normalize_symbols(self) -> Self where Self: Sized {
        let (_, id) = self.normalize_sym(Env::new());
        id
    }
}

impl NormalizeSym for Name {
    fn normalize_sym(self, mut env: Env) -> (Env, Name) {
        match env.get(&self) {
            Some(id) => {
                let new_id = id.clone();
                (env, new_id)
            },
            None => {
                let new_id = Name {s: self.s.clone(), sym: Some(env.len() as i64)};
                env.insert(self, new_id.clone());
                (env, new_id)
            }
        }
    }
}

impl<T: NormalizeSym> NormalizeSym for Vec<T> {
    fn normalize_sym(self, env: Env) -> (Env, Vec<T>) {
        self.into_iter()
            .fold((env, vec![]), |(env, mut acc), t| {
                let (env, t) = t.normalize_sym(env);
                acc.push(t);
                (env, acc)
            })
    }
}

impl NormalizeSym for py_ast::Expr {
    fn normalize_sym(self, env: Env) -> (Env, py_ast::Expr) {
        match self {
            py_ast::Expr::Var {id, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, py_ast::Expr::Var {id, ty, i})
            },
            py_ast::Expr::Call {id, args, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, py_ast::Expr::Call {id, args, ty, i})
            },
            py_ast::Expr::Callback {id, args, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, py_ast::Expr::Callback {id, args, ty, i})
            },
            _ => self.smap_accum_l(env, |env, e| e.normalize_sym(env))
        }
    }
}

impl NormalizeSym for gpu_ast::Expr {
    fn normalize_sym(self, env: Env) -> (Env, gpu_ast::Expr) {
        match self {
            gpu_ast::Expr::Var {id, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, gpu_ast::Expr::Var {id, ty, i})
            },
            gpu_ast::Expr::Call {id, args, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, gpu_ast::Expr::Call {id, args, ty, i})
            },
            gpu_ast::Expr::PyCallback {id, args, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, gpu_ast::Expr::PyCallback {id, args, ty, i})
            },
            _ => self.smap_accum_l(env, |env, e| e.normalize_sym(env))
        }
    }
}

impl NormalizeSym for gpu_ast::Stmt {
    fn normalize_sym(self, env: Env) -> (Env, gpu_ast::Stmt) {
        match self {
            gpu_ast::Stmt::Definition {ty, id, expr, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, expr) = expr.normalize_sym(env);
                (env, gpu_ast::Stmt::Definition {ty, id, expr, i})
            },
            gpu_ast::Stmt::For {var_ty, var, init, cond, incr, body, unroll, i} => {
                let (env, var) = var.normalize_sym(env);
                let (env, init) = init.normalize_sym(env);
                let (env, cond) = cond.normalize_sym(env);
                let (env, incr) = incr.normalize_sym(env);
                let (env, body) = body.smap_accum_l(env, |env, s| s.normalize_sym(env));
                (env, gpu_ast::Stmt::For {
                    var_ty, var, init, cond, incr, body, unroll, i
                })
            },
            gpu_ast::Stmt::ParallelReduction {
                var_ty, var, init, cond, incr, body, nthreads, tpb, unroll, i
            } => {
                let (env, var) = var.normalize_sym(env);
                let (env, init) = init.normalize_sym(env);
                let (env, cond) = cond.normalize_sym(env);
                let (env, incr) = incr.normalize_sym(env);
                let (env, body) = body.normalize_sym(env);
                (env, gpu_ast::Stmt::ParallelReduction {
                    var_ty, var, init, cond, incr, body, nthreads, tpb, unroll, i
                })
            },
            gpu_ast::Stmt::KernelLaunch {id, args, grid, smem, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, gpu_ast::Stmt::KernelLaunch {id, args, grid, smem, i})
            },
            gpu_ast::Stmt::AllocDevice {elem_ty, id, sz, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, gpu_ast::Stmt::AllocDevice {elem_ty, id, sz, i})
            },
            gpu_ast::Stmt::AllocShared {elem_ty, id, sz, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, gpu_ast::Stmt::AllocShared {elem_ty, id, sz, i})
            },
            gpu_ast::Stmt::FreeDevice {id, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, gpu_ast::Stmt::FreeDevice {id, i})
            },
            _ => {
                let (env, s) = self.smap_accum_l(env, |env, s: gpu_ast::Stmt| {
                    s.normalize_sym(env)
                });
                s.smap_accum_l(env, |env, e: gpu_ast::Expr| e.normalize_sym(env))
            }
        }
    }
}

impl NormalizeSym for triton_ast::Expr {
    fn normalize_sym(self, env: Env) -> (Env, triton_ast::Expr) {
        match self {
            triton_ast::Expr::Var {id, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                (env, triton_ast::Expr::Var {id, ty, i})
            },
            triton_ast::Expr::Call {id, args, ty, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, triton_ast::Expr::Call {id, args, ty, i})
            },
            _ => self.smap_accum_l(env, |env, e| e.normalize_sym(env))
        }
    }
}

impl NormalizeSym for triton_ast::Stmt {
    fn normalize_sym(self, env: Env) -> (Env, triton_ast::Stmt) {
        match self {
            triton_ast::Stmt::Definition {dst, expr, i} => {
                let (env, dst) = dst.normalize_sym(env);
                let (env, expr) = expr.normalize_sym(env);
                (env, triton_ast::Stmt::Definition {dst, expr, i})
            },
            triton_ast::Stmt::Assign {dst, expr, i} => {
                let (env, dst) = dst.normalize_sym(env);
                let (env, expr) = expr.normalize_sym(env);
                (env, triton_ast::Stmt::Assign {dst, expr, i})
            },
            triton_ast::Stmt::For {var, lo, hi, step, body, i} => {
                let (env, var) = var.normalize_sym(env);
                let (env, lo) = lo.normalize_sym(env);
                let (env, hi) = hi.normalize_sym(env);
                let (env, body) = body.normalize_sym(env);
                (env, triton_ast::Stmt::For {var, lo, hi, step, body, i})
            },
            triton_ast::Stmt::KernelLaunch {id, attrs, block_dims, args, nwarps, i} => {
                let (env, id) = id.normalize_sym(env);
                let (env, attrs) = attrs.normalize_sym(env);
                let (env, args) = args.normalize_sym(env);
                (env, triton_ast::Stmt::KernelLaunch {id, attrs, block_dims, args, nwarps, i})
            },
            _ => {
                let (env, s) = self.smap_accum_l(env, |env, s: triton_ast::Stmt| {
                    s.normalize_sym(env)
                });
                s.smap_accum_l(env, |env, e: triton_ast::Expr| e.normalize_sym(env))
            }
        }
    }
}
