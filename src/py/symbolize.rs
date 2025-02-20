use super::ast::*;
use crate::py_name_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub struct SymbolizeEnv {
    vars: BTreeMap<String, Name>,
    i: Info
}

type SymbolizeResult<T> = PyResult<(SymbolizeEnv, T)>;

impl SymbolizeEnv {
    pub fn has_symbol(&self, id: &Name) -> bool {
        id.has_sym() || self.vars.contains_key(id.get_str())
    }

    pub fn get_symbol(&self, id: Name) -> PyResult<Name> {
        if id.has_sym() {
            Ok(id)
        } else {
            if let Some(n) = self.vars.get(id.get_str()) {
                Ok(n.clone())
            } else {
                py_name_error!(self.i, "Found reference to unknown variable {id}")
            }
        }
    }

    pub fn set_symbol(mut self, id: Name) -> (Self, Name) {
        if id.has_sym() {
            (self, id)
        } else {
            let id = id.with_new_sym();
            self.vars.insert(id.get_str().clone(), id.clone());
            (self, id)
        }
    }

    pub fn set_info(self, i: Info) -> Self {
        SymbolizeEnv {i, ..self}
    }
}

impl Default for SymbolizeEnv {
    fn default() -> Self {
        SymbolizeEnv {vars: BTreeMap::new(), i: Info::default()}
    }
}

pub trait Symbolize {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Self> where Self: Sized;

    fn symbolize_default(self) -> PyResult<Self> where Self: Sized {
        let (_, s) = self.symbolize(SymbolizeEnv::default())?;
        Ok(s)
    }
}

impl <'a, T: Symbolize + 'a> Symbolize for Vec<T> {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Vec<T>> {
        self.into_iter()
            .fold(Ok((env, vec![])), |acc, v| {
                let (env, mut vec) = acc?;
                let (env, v) = v.symbolize(env)?;
                vec.push(v);
                Ok((env, vec))
            })
    }
}

impl Symbolize for Expr {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Expr> {
        let env = env.set_info(self.get_info());
        match self {
            Expr::Var {id, ty, i} => {
                let id = env.get_symbol(id)?;
                Ok((env, Expr::Var {id, ty, i}))
            },
            Expr::String {..} | Expr::Bool {..} | Expr::Int {..} |
            Expr::Float {..} | Expr::UnOp {..} | Expr::BinOp {..} |
            Expr::IfExpr {..} | Expr::Subscript {..} | Expr::Slice {..} |
            Expr::Tuple {..} | Expr::Dict {..} | Expr::Builtin {..} |
            Expr::Convert {..} => {
                self.smap_accum_l_result(Ok(env), |env, e| e.symbolize(env))
            }
        }
    }
}

impl Symbolize for Stmt {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Stmt> {
        match self {
            Stmt::Definition {ty, id, expr, labels, i} => {
                let (env, id) = env.set_symbol(id);
                let (env, expr) = expr.symbolize(env)?;
                Ok((env, Stmt::Definition {ty, id, expr, labels, i}))
            },
            Stmt::Assign {dst, expr, labels, i, ..} => {
                // If we assign to a variable without a symbol, this means it is being introduced
                // here. In this case, we replace the assign node with a definition node,
                // indicating that this introduces and assigns a value to a new variable.
                match dst {
                    Expr::Var {id, ty, i} if !env.has_symbol(&id) => {
                        let (env, id) = env.set_symbol(id);
                        let (env, expr) = expr.symbolize(env)?;
                        Ok((env, Stmt::Definition {ty, id, expr, labels, i}))
                    },
                    _ => {
                        let (env, dst) = dst.symbolize(env)?;
                        let (env, expr) = expr.symbolize(env)?;
                        Ok((env, Stmt::Assign {dst, expr, labels, i}))
                    }
                }
            },
            Stmt::For {var, lo, hi, step, body, labels, i} => {
                let (body_env, var) = env.clone().set_symbol(var);
                let (body_env, lo) = lo.symbolize(body_env)?;
                let (body_env, hi) = hi.symbolize(body_env)?;
                let (_, body) = body.symbolize(body_env)?;
                Ok((env, Stmt::For {var, lo, hi, step, body, labels, i}))
            },
            Stmt::While {..} | Stmt::If {..} | Stmt::WithGpuContext {..} |
            Stmt::Call {..} | Stmt::Label {..} => {
                let (env, s) = self.smap_accum_l_result(Ok(env), |env, e: Expr| e.symbolize(env))?;
                s.smap_accum_l_result(Ok(env), |env, s: Stmt| s.symbolize(env))
            }
        }
    }
}

impl Symbolize for Param {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Param> {
        let Param {id, ty, i} = self;
        let (env, id) = env.set_symbol(id);
        Ok((env, Param {id, ty, i}))
    }
}

impl Symbolize for FunDef {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<FunDef> {
        let FunDef {id, params, body, i} = self;
        let (env, id) = env.set_symbol(id);
        let (env, params) = params.symbolize(env)?;
        let (env, body) = body.symbolize(env)?;
        Ok((env, FunDef {id, params, body, i}))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn name(s: &str) -> Name {
        Name::new(s.to_string())
    }

    fn sym_env(entries: Vec<Name>) -> SymbolizeEnv {
        let vars = entries.into_iter()
            .map(|id| (id.get_str().clone(), id))
            .collect::<BTreeMap<String, Name>>();
        SymbolizeEnv {vars, i: Info::default()}
    }

    fn nvar(id: &Name) -> Expr {
        Expr::Var {
            id: id.clone(), ty: Type::Tensor {sz: ElemSize::Bool, shape: vec![]},
            i: Info::default()
        }
    }

    fn var(s: &str) -> Expr {
        nvar(&Name::new(s.to_string()))
    }

    fn int(v: i64) -> Expr {
        Expr::Int {
            v, ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]}, i: Info::default()
        }
    }

    #[test]
    fn symbolize_unknown_var_fail() {
        assert!(var("x").symbolize_default().is_err());
    }

    #[test]
    fn symbolize_known_var_ok() {
        let x = name("x");
        let env = sym_env(vec![x.clone()]);
        let (_, var) = var("x").symbolize(env).unwrap();
        if let Expr::Var {id, ..} = var {
            assert_eq!(x, id);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn symbolize_defining_assignment_stmt() {
        let id = name("x");
        let i = Info::default();
        let s = Stmt::Assign {
            dst: nvar(&id), expr: int(0), labels: vec![], i: i.clone()
        };
        let env = sym_env(vec![]);
        let (env, stmt) = s.symbolize(env).unwrap();
        assert!(env.vars.len() == 1);
        assert!(env.vars.contains_key(id.get_str()));
        if let Stmt::Definition {id: def_id, ..} = stmt {
            assert!(def_id.has_sym());
        } else {
            assert!(false);
        }
    }

    #[test]
    fn symbolize_reassignment_stmt() {
        let id = name("x");
        let i = Info::default();
        let s = Stmt::Assign {
            dst: nvar(&id), expr: int(0), labels: vec![], i: i.clone()
        };
        let id_sym = id.clone().with_new_sym();
        let env = sym_env(vec![id_sym.clone()]);
        let (env, stmt) = s.symbolize(env).unwrap();
        assert!(env.vars.len() == 1);
        assert_eq!(env.vars.get(id.get_str()), Some(id_sym.clone()).as_ref());
        if let Stmt::Assign {dst: Expr::Var {id: var_id, ..}, ..} = stmt {
            assert_eq!(var_id, id_sym);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn symbolize_for_stmt() {
        let x = name("x");
        let y = name("y");
        let i = Info::default();
        let s = Stmt::For {
            var: x.clone(),
            lo: int(0),
            hi: int(10),
            step: 1,
            body: vec![Stmt::Assign {
                dst: nvar(&y), expr: nvar(&x), labels: vec![], i: i.clone()
            }],
            labels: vec![],
            i: i.clone()
        };
        let env = sym_env(vec![]);
        let (_, stmt) = s.symbolize(env).unwrap();
        if let Stmt::For {var, body, ..} = stmt {
            assert!(var.has_sym());
            assert_eq!(var.get_str(), x.get_str());
            if let Stmt::Definition {id: y_id, ..} = &body[0] {
                assert!(y_id.has_sym());
                assert_eq!(y_id.get_str(), y.get_str());
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }
}
