use super::ast::*;
use crate::parir_compile_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;

#[derive(Debug)]
pub struct SymbolizeEnv {
    vars: BTreeMap<String, Name>,
    i: Info
}

type SymbolizeResult<T> = CompileResult<(SymbolizeEnv, T)>;

impl SymbolizeEnv {
    pub fn has_symbol(&self, id: &Name) -> bool {
        id.has_sym() || self.vars.contains_key(id.get_str())
    }

    pub fn get_symbol(&self, id: Name) -> CompileResult<Name> {
        if id.has_sym() {
            Ok(id)
        } else {
            if let Some(n) = self.vars.get(id.get_str()) {
                Ok(n.clone())
            } else {
                parir_compile_error!(self.i, "Found reference to unknown variable {id}")
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

    fn symbolize_default(self) -> SymbolizeResult<Self> where Self: Sized {
        self.symbolize(SymbolizeEnv::default())
    }
}

impl<'a, T: Symbolize + 'a> Symbolize for Vec<T> {
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

impl Symbolize for Type {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Type> {
        let ty = match self {
            Type::Struct {id} => {
                let id = env.get_symbol(id)?;
                Type::Struct {id}
            },
            Type::Boolean | Type::Tensor {..} => self
        };
        Ok((env, ty))
    }
}

impl Symbolize for Expr {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Expr> {
        let env = env.set_info(self.get_info());
        match self {
            Expr::Var {id, ty, i} => {
                let id = env.get_symbol(id)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Var {id, ty, i}))
            },
            Expr::Int {v, ty, i} => {
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Int {v, ty, i}))
            },
            Expr::Float {v, ty, i} => {
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Float {v, ty, i}))
            },
            Expr::UnOp {op, arg, ty, i} => {
                let (env, arg) = arg.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
            },
            Expr::BinOp {lhs, op, rhs, ty, i} => {
                let (env, lhs) = lhs.symbolize(env)?;
                let (env, rhs) = rhs.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}))
            },
            Expr::StructFieldAccess {target, label, ty, i} => {
                let (env, target) = target.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::StructFieldAccess {target: Box::new(target), label, ty, i}))
            },
            Expr::TensorAccess {target, idx, ty, i} => {
                let (env, target) = target.symbolize(env)?;
                let (env, idx) = idx.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::TensorAccess {target: Box::new(target), idx: Box::new(idx), ty, i}))
            },
            Expr::Struct {id, fields, ty, i} => {
                let id = env.get_symbol(id)?;
                let (env, fields) = fields.into_iter()
                    .fold(Ok((env, vec![])), |acc, (id, e)| {
                        let (env, mut v) = acc?;
                        let (env, e) = e.symbolize(env)?;
                        v.push((id, e));
                        Ok((env, v))
                    })?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Struct {id, fields, ty, i}))
            },
            Expr::Builtin {func, args, ty, i} => {
                let (env, args) = args.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Builtin {func, args, ty, i}))
            },
            Expr::Convert {e, ty} => {
                let (env, e) = e.symbolize(env)?;
                let (env, ty) = ty.symbolize(env)?;
                Ok((env, Expr::Convert {e: Box::new(e), ty}))
            }
        }
    }
}

impl Symbolize for Stmt {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Stmt> {
        let env = env.set_info(self.get_info());
        match self {
            Stmt::Definition {ty, id, expr, i} => {
                let (env, ty) = ty.symbolize(env)?;
                let (env, id) = env.set_symbol(id);
                let (env, expr) = expr.symbolize(env)?;
                Ok((env, Stmt::Definition {ty, id, expr, i}))
            },
            Stmt::Assign {dst, expr, i} => {
                // If we are assigning to a variable that does not yet have a symbol, it is being
                // introduced. In this case, we replace the assign node with a definition node, to
                // indicate that this introduces and assigns a value to a new variable.
                match dst {
                    Expr::Var {id, ty, ..} if !env.has_symbol(&id) => {
                        let (env, id) = env.set_symbol(id);
                        let (env, ty) = ty.symbolize(env)?;
                        let (env, expr) = expr.symbolize(env)?;
                        Ok((env, Stmt::Definition {ty, id, expr, i}))
                    },
                    _ => {
                        let (env, dst) = dst.symbolize(env)?;
                        let (env, expr) = expr.symbolize(env)?;
                        Ok((env, Stmt::Assign {dst, expr, i}))
                    }
                }
            },
            Stmt::For {var, lo, hi, body, par, i} => {
                let (env, var) = env.set_symbol(var);
                let (env, lo) = lo.symbolize(env)?;
                let (env, hi) = hi.symbolize(env)?;
                let (env, body) = body.symbolize(env)?;
                Ok((env, Stmt::For {var, lo, hi, body, par, i}))
            },
            Stmt::If {cond, thn, els, i} => {
                let (env, cond) = cond.symbolize(env)?;
                let (env, thn) = thn.symbolize(env)?;
                let (env, els) = els.symbolize(env)?;
                Ok((env, Stmt::If {cond, thn, els, i}))
            }
        }
    }
}

impl Symbolize for Field {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Field> {
        let Field {id, ty, i} = self;
        let (env, id) = env.set_symbol(id);
        let (env, ty) = ty.symbolize(env)?;
        Ok((env, Field {id, ty, i}))
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

impl Symbolize for Top {
    fn symbolize(self, env: SymbolizeEnv) -> SymbolizeResult<Top> {
        match self {
            Top::StructDef {id, fields, i} => {
                let (env, id) = env.set_symbol(id);
                let (env, fields) = fields.symbolize(env)?;
                Ok((env, Top::StructDef {id, fields, i}))
            },
            Top::FunDef {id, params, body, i} => {
                let (env, id) = env.set_symbol(id);
                let (env, params) = params.symbolize(env)?;
                let (env, body) = body.symbolize(env)?;
                Ok((env, Top::FunDef {id, params, body, i}))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn name(s: &str) -> Name {
        Name::new(s.to_string())
    }

    fn nosym(n: &Name) -> Name {
        Name::new(n.get_str().clone())
    }

    fn sym_env(entries: Vec<Name>) -> SymbolizeEnv {
        let vars = entries.into_iter()
            .map(|id| (id.get_str().clone(), id))
            .collect::<BTreeMap<String, Name>>();
        SymbolizeEnv {vars, i: Info::default()}
    }

    fn sym_type_with_env(env: SymbolizeEnv, ty: Type) -> CompileResult<Type> {
        let (_, ty) = ty.symbolize(env)?;
        Ok(ty)
    }

    fn sym_type(ty: Type) -> CompileResult<Type> {
        sym_type_with_env(SymbolizeEnv::default(), ty)
    }

    #[test]
    fn symbolize_bool_type() {
        assert_eq!(sym_type(Type::Boolean).unwrap(), Type::Boolean);
    }

    #[test]
    fn symbolize_scalar_type() {
        let ty = Type::Tensor {sz: ElemSize::I32, shape: vec![]};
        assert_eq!(sym_type(ty.clone()).unwrap(), ty);
    }

    #[test]
    fn symbolize_tensor_type() {
        let ty = Type::Tensor {sz: ElemSize::I64, shape: vec![5]};
        assert_eq!(sym_type(ty.clone()).unwrap(), ty);
    }

    #[test]
    fn symbolize_unknown_struct_fail() {
        let ty = Type::Struct {id: Name::new("x".to_string())};
        assert!(sym_type(ty).is_err());
    }

    #[test]
    fn symbolize_known_struct_ok() {
        let x = name("x");
        let env = sym_env(vec![x.clone()]);
        let ty = Type::Struct {id: nosym(&x)};
        let expected = Type::Struct {id: x};
        assert_eq!(sym_type_with_env(env, ty).unwrap(), expected);
    }

    fn nvar(id: &Name) -> Expr {
        Expr::Var {id: id.clone(), ty: Type::Boolean, i: Info::default()}
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
            dst: nvar(&id), expr: int(0), i: i.clone()
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
            dst: nvar(&id), expr: int(0), i: i.clone()
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
            body: vec![Stmt::Assign {dst: nvar(&y), expr: nvar(&x), i: i.clone()}],
            par: vec![],
            i: i.clone()
        };
        let env = sym_env(vec![]);
        let (env, stmt) = s.symbolize(env).unwrap();
        assert!(env.vars.len() == 2);
        assert!(env.vars.contains_key(x.get_str()));
        assert!(env.vars.contains_key(y.get_str()));
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
