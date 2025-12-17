use super::ast::*;
use crate::py_name_error;
use crate::ext::types::Symbol;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::SMapAccum;

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct SymbolizeEnv {
    vars: BTreeMap<String, Name>,
    shape_vars: BTreeSet<Name>,
    symbols: BTreeMap<String, Symbol>,
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
            } else if let Some(sym) = self.symbols.get(id.get_str()) {
                // We construct a custom name containing a string value corresponding to the name
                // to which the symbol is bound in Python, but using the symbol value of the inner
                // symbol.
                let id = Name {s: id.s.clone(), sym: sym.id.sym};
                if self.shape_vars.contains(&id) {
                    Ok(id)
                } else {
                    py_name_error!(self.i, "Found reference to unused shape variable {id}")
                }
            } else {
                py_name_error!(self.i, "Found reference to unknown variable {id}")
            }
        }
    }

    pub fn set_symbol(mut self, id: Name) -> (Self, Name) {
        let id = if id.has_sym() {
            id
        } else {
            id.with_new_sym()
        };
        self.vars.insert(id.get_str().clone(), id.clone());
        (self, id)
    }

    pub fn set_info(self, i: Info) -> Self {
        SymbolizeEnv {i, ..self}
    }
}

pub trait Symbolize {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Self> where Self: Sized;
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

impl Symbolize for TensorShape {
    fn symbolize(self, mut env: SymbolizeEnv) -> SymbolizeResult<TensorShape> {
        match self {
            TensorShape::Num {..} => Ok((env, self)),
            TensorShape::Symbol {id} => {
                env.shape_vars.insert(id.clone());
                Ok((env, TensorShape::Symbol {id}))
            }
        }
    }
}

impl Symbolize for Type {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Type> {
        match self {
            Type::Tensor {sz, shape} => {
                let (env, shape) = shape.symbolize(env)?;
                Ok((env, Type::Tensor {sz, shape}))
            },
            _ => self.smap_accum_l_result(Ok(env), |env, ty| ty.symbolize(env))
        }
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
            Expr::Call {id, args, ty, i} => {
                let id = env.get_symbol(id)?;
                let (env, args) = args.symbolize(env)?;
                Ok((env, Expr::Call {id, args, ty, i}))
            },
            Expr::Convert {e, ty, i} => {
                let (env, e) = e.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Convert {e: Box::new(e), ty, i}))
            },
            Expr::String {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
            Expr::UnOp {..} | Expr::BinOp {..} | Expr::ReduceOp {..} | Expr::IfExpr {..} |
            Expr::Subscript {..} | Expr::Slice {..} | Expr::Tuple {..} | Expr::List {..} |
            Expr::Callback {..} | Expr::GpuContext {..} | Expr::Inline {..} | Expr::Label {..} |
            Expr::StaticBackendEq {..} | Expr::StaticTypesEq {..} | Expr::StaticFail {..} |
            Expr::AllocShared {..} => {
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
                let (body_env, step) = step.symbolize(body_env)?;
                let (_, body) = body.symbolize(body_env)?;
                Ok((env, Stmt::For {var, lo, hi, step, body, labels, i}))
            },
            Stmt::Expr {e, i} => {
                let (env, e) = e.symbolize(env)?;
                Ok((env, Stmt::Expr {e, i}))
            },
            Stmt::While {..} | Stmt::If {..} | Stmt::Return {..} |
            Stmt::WithGpuContext {..} => {
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
        let (env, ty) = ty.symbolize(env)?;
        Ok((env, Param {id, ty, i}))
    }
}

impl Symbolize for FunDef {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<FunDef> {
        let FunDef {id, params, body, res_ty, i} = self;
        let (env, id) = env.set_symbol(id);
        let (env, params) = params.symbolize(env)?;
        let (env, body) = body.symbolize(env)?;
        Ok((env, FunDef {id, params, body, res_ty, i}))
    }
}

fn extract_top_name<'py>(t: Bound<'py, PyCapsule>) -> Name {
    let t: &Top = unsafe { t.reference() };
    match t {
        Top::CallbackDecl {id, ..} | Top::ExtDecl {id, ..} |
        Top::FunDef {v: FunDef {id, ..}} => id.clone()
    }
}

fn try_extract_symbol<'py>(
    mut acc: BTreeMap<String, Symbol>,
    entry: (Bound<'py, PyAny>, Bound<'py, PyAny>)
) -> BTreeMap<String, Symbol> {
    let (key, value) = entry;
    match value.extract::<Symbol>() {
        Ok(v) => {
            acc.insert(key.extract::<String>().unwrap(), v);
            acc
        },
        Err(_) => acc
    }
}

fn extract_symbols<'py>(
    vars: &(Bound<'py, PyDict>, Bound<'py, PyDict>)
) -> BTreeMap<String, Symbol> {
    let (globals, locals) = vars;
    globals.iter()
        .chain(locals.iter())
        .fold(BTreeMap::new(), try_extract_symbol)

}

pub fn with_tops<'py>(
    tops: &BTreeMap<String, Bound<'py, PyCapsule>>,
    vars: &(Bound<'py, PyDict>, Bound<'py, PyDict>),
    def: FunDef
) -> PyResult<FunDef> {
    let symbols = extract_symbols(vars);
    let vars = tops.clone()
        .into_iter()
        .map(|(id, cap)| (id, extract_top_name(cap)))
        .collect::<BTreeMap<String, Name>>();
    let env = SymbolizeEnv {
        vars,
        shape_vars: BTreeSet::new(),
        symbols,
        i: def.i.clone()
    };
    let (_, def) = def.symbolize(env)?;
    Ok(def)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::ext::types;
    use crate::py::ast_builder::*;

    fn sym_env(entries: Vec<Name>) -> SymbolizeEnv {
        let vars = entries.into_iter()
            .map(|id| (id.get_str().clone(), id))
            .collect::<BTreeMap<String, Name>>();
        SymbolizeEnv {
            vars,
            shape_vars: BTreeSet::new(),
            symbols: BTreeMap::new(),
            i: Info::default()
        }
    }

    fn nvar(id: &Name) -> Expr {
        Expr::Var {
            id: id.clone(),
            ty: scalar(ElemSize::Bool),
            i: Info::default()
        }
    }

    #[test]
    fn sym_env_empty_has_symbol() {
        let env = sym_env(vec![]);
        assert!(!env.has_symbol(&id("x")));
    }

    #[test]
    fn sym_env_contains_symbol() {
        let env = sym_env(vec![id("x")]);
        assert!(env.has_symbol(&id("x")));
    }

    #[test]
    fn sym_env_get_unknown_sym() {
        let env = sym_env(vec![]);
        assert!(env.get_symbol(id("x")).is_err());
    }

    #[test]
    fn sym_env_get_existing_sym() {
        let env = sym_env(vec![Name::sym_str("x")]);
        assert!(env.get_symbol(id("x")).unwrap().has_sym());
    }

    #[test]
    fn sym_env_get_symbolized_name() {
        let env = sym_env(vec![]);
        assert!(env.get_symbol(Name::sym_str("x")).unwrap().has_sym());
    }

    #[test]
    fn sym_env_set_sym() {
        let env = sym_env(vec![id("x")]);
        assert!(!env.get_symbol(id("x")).unwrap().has_sym());
        let (env, x) = env.set_symbol(id("x"));
        assert!(env.get_symbol(id("x")).unwrap().has_sym());
        assert!(x.has_sym());
    }

    #[test]
    fn symbolize_unknown_var_fail() {
        let e = var("x", Type::Unknown);
        let env = sym_env(vec![]);
        assert_py_error_matches(e.symbolize(env), "unknown variable");
    }

    #[test]
    fn symbolize_known_var_ok() {
        let x = id("x");
        let env = sym_env(vec![x.clone()]);
        let (_, var) = var("x", Type::Unknown).symbolize(env).unwrap();
        if let Expr::Var {id, ..} = var {
            assert_eq!(x, id);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn symbolize_defining_assignment_stmt() {
        let id = id("x");
        let i = Info::default();
        let s = Stmt::Assign {
            dst: nvar(&id), expr: int(0, Some(ElemSize::I64)), labels: vec![], i: i.clone()
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
        let id = id("x");
        let i = Info::default();
        let s = Stmt::Assign {
            dst: nvar(&id), expr: int(0, Some(ElemSize::I64)), labels: vec![], i: i.clone()
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
        let x = id("x");
        let y = id("y");
        let i = Info::default();
        let s = Stmt::For {
            var: x.clone(),
            lo: int(0, Some(ElemSize::I64)),
            hi: int(10, Some(ElemSize::I64)),
            step: int(1, Some(ElemSize::I64)),
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

    #[test]
    fn symbolize_fun_def_with_shape_symbol_annot() -> Result<(), ()> {
        let n = Name::sym_str("");
        let param = Param {
            id: id("x"),
            ty: Type::Tensor {
                sz: fixed_elem_sz(ElemSize::I32),
                shape: vec![TensorShape::Symbol {id: n.clone()}]
            },
            i: i()
        };
        let def = FunDef {
            id: id("f"),
            params: vec![param],
            body: vec![
                assignment(var("y", tyuk()),
                Expr::Var {id: id("N"), ty: tyuk(), i: i()})
            ],
            res_ty: Type::Void,
            i: i()
        };
        let mut env = sym_env(vec![]);
        env.symbols.insert("N".to_string(), types::Symbol {id: n});
        let (_, FunDef {id, mut params, mut body, ..}) = def.symbolize(env).unwrap();
        assert!(id.has_sym());
        let Param {id, ty, ..} = params.pop().unwrap();
        assert!(id.has_sym());
        let shape_id = if let Type::Tensor {mut shape, ..} = ty {
            if let TensorShape::Symbol {id} = shape.pop().unwrap() {
                assert!(id.has_sym());
                Ok(id)
            } else {
                Err(())
            }
        } else {
            Err(())
        }?;
        if let Stmt::Definition {expr, ..} = body.pop().unwrap() {
            if let Expr::Var {id, ..} = expr {
                assert_eq!(id, shape_id);
            }
        } else {
            assert!(false);
        }
        Ok(())
    }

    #[test]
    fn symbolize_undefined_shape_variable() {
        let n = Name::sym_str("");
        let mut env = sym_env(vec![]);
        env.symbols.insert("N".to_string(), types::Symbol {id: n.clone()});
        let e = Expr::Var {id: id("N"), ty: tyuk(), i: i()};
        assert_py_error_matches(e.symbolize(env), "Found reference to unused shape variable N");
    }
}
