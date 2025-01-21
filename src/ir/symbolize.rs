use super::ast::*;
use crate::parir_compile_error;
use crate::err::*;
use crate::info::*;
use crate::name::Name;

use std::collections::BTreeMap;

#[derive(Debug)]
struct SymbolizeEnv {
    vars: BTreeMap<String, Name>
}

type SymbolizeResult<T> = CompileResult<(SymbolizeEnv, T)>;

impl SymbolizeEnv {

    pub fn has_symbol(&self, id: &Name) -> bool {
        id.has_sym() || self.vars.contains_key(id.get_str())
    }

    pub fn get_symbol(&self, i: &Info, id: Name) -> CompileResult<Name> {
        if id.has_sym() {
            Ok(id)
        } else {
            if let Some(n) = self.vars.get(id.get_str()) {
                Ok(n.clone())
            } else {
                parir_compile_error!(i, "Found reference to unknown variable {id}")
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
}

impl Default for SymbolizeEnv {
    fn default() -> Self {
        SymbolizeEnv {vars: BTreeMap::new()}
    }
}

fn symbolize_vec<T>(
    env: SymbolizeEnv,
    nodes: Vec<T>,
    symbolize_fun: impl Fn(SymbolizeEnv, T) -> SymbolizeResult<T>
) -> SymbolizeResult<Vec<T>> {
    nodes.into_iter()
        .fold(Ok((env, vec![])), |acc, v| {
            let (env, mut vec) = acc?;
            let (env, v) = symbolize_fun(env, v)?;
            vec.push(v);
            Ok((env, vec))
        })
}

fn symbolize_vec_borrow<T>(
    env: &SymbolizeEnv,
    nodes: Vec<T>,
    symbolize_fun: impl Fn(&SymbolizeEnv, T) -> CompileResult<T>
) -> CompileResult<Vec<T>> {
    nodes.into_iter()
        .fold(Ok(vec![]), |acc, v| {
            let mut vec = acc?;
            let v = symbolize_fun(&env, v)?;
            vec.push(v);
            Ok(vec)
        })
}

fn symbolize_type(env: &SymbolizeEnv, i: &Info, ty: Type) -> CompileResult<Type> {
    match ty {
        Type::Struct {id} => {
            let id = env.get_symbol(i, id)?;
            Ok(Type::Struct {id})
        },
        Type::Boolean | Type::Tensor {..} => Ok(ty)
    }
}

fn symbolize_expr(env: &SymbolizeEnv, e: Expr) -> CompileResult<Expr> {
    let ty = symbolize_type(env, &e.get_info(), e.get_type().clone())?;
    let e = e.with_type(ty);
    match e {
        Expr::Var {id, ty, i} => {
            let id = env.get_symbol(&i, id)?;
            Ok(Expr::Var {id, ty, i})
        },
        Expr::UnOp {op, arg, ty, i} => {
            let arg = symbolize_expr(env, *arg)?;
            Ok(Expr::UnOp {op, arg: Box::new(arg), ty, i})
        },
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = Box::new(symbolize_expr(env, *lhs)?);
            let rhs = Box::new(symbolize_expr(env, *rhs)?);
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        Expr::StructFieldAccess {target, label, ty, i} => {
            let target = Box::new(symbolize_expr(env, *target)?);
            Ok(Expr::StructFieldAccess {target, label, ty, i})
        },
        Expr::TensorAccess {target, idx, ty, i} => {
            let target = Box::new(symbolize_expr(env, *target)?);
            let idx = Box::new(symbolize_expr(env, *idx)?);
            Ok(Expr::TensorAccess {target, idx, ty, i})
        },
        Expr::Struct {id, fields, ty, i} => {
            let id = env.get_symbol(&i, id)?;
            let fields = symbolize_vec_borrow(&env, fields, |env, (id, e)| {
                let e = symbolize_expr(&env, e)?;
                Ok((id, e))
            })?;
            Ok(Expr::Struct {id, fields, ty, i})
        },
        Expr::Builtin {func, args, ty, i} => {
            let args = symbolize_vec_borrow(&env, args, symbolize_expr)?;
            Ok(Expr::Builtin {func, args, ty, i})
        },
        Expr::Convert {e, ty} => {
            let e = Box::new(symbolize_expr(env, *e)?);
            Ok(Expr::Convert {e, ty})
        },
        Expr::String {..} | Expr::Int {..} | Expr::Float {..} => Ok(e),
    }
}

fn symbolize_stmt(env: SymbolizeEnv, stmt: Stmt) -> SymbolizeResult<Stmt> {
    match stmt {
        Stmt::Definition {ty, id, expr, i} => {
            let ty = symbolize_type(&env, &i, ty)?;
            let (env, id) = env.set_symbol(id);
            let expr = symbolize_expr(&env, expr)?;
            Ok((env, Stmt::Definition {ty, id, expr, i}))
        },
        Stmt::Assign {dst, expr, i} => {
            // If we are assigning to a variable that does not yet have a symbol, it is being
            // introduced. In this case, we replace the assign node with a definition node, to
            // indicate that this introduces and assigns a value to a new variable.
            match dst {
                Expr::Var {id, ty, i: var_i} if !env.has_symbol(&id) => {
                    let (env, id) = env.set_symbol(id);
                    let ty = symbolize_type(&env, &var_i, ty)?;
                    let expr = symbolize_expr(&env, expr)?;
                    Ok((env, Stmt::Definition {ty, id, expr, i}))
                },
                _ => {
                    let dst = symbolize_expr(&env, dst)?;
                    let expr = symbolize_expr(&env, expr)?;
                    Ok((env, Stmt::Assign {dst, expr, i}))
                }
            }
        },
        Stmt::For {var, lo, hi, body, par, i} => {
            let (env, var) = env.set_symbol(var);
            let lo = symbolize_expr(&env, lo)?;
            let hi = symbolize_expr(&env, hi)?;
            let (env, body) = symbolize_vec(env, body, symbolize_stmt)?;
            Ok((env, Stmt::For {var, lo, hi, body, par, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = symbolize_expr(&env, cond)?;
            let (env, thn) = symbolize_vec(env, thn, symbolize_stmt)?;
            let (env, els) = symbolize_vec(env, els, symbolize_stmt)?;
            Ok((env, Stmt::If {cond, thn, els, i}))
        },
    }
}

fn symbolize_field(env: SymbolizeEnv, field: Field) -> SymbolizeResult<Field> {
    let (env, id) = env.set_symbol(field.id);
    let ty = symbolize_type(&env, &field.i, field.ty)?;
    Ok((env, Field {id, ty, ..field}))
}

fn symbolize_param(env: SymbolizeEnv, param: Param) -> SymbolizeResult<Param> {
    let (env, id) = env.set_symbol(param.id);
    let ty = symbolize_type(&env, &param.i, param.ty)?;
    Ok((env, Param {id, ty, ..param}))
}

fn symbolize_top(
    env: SymbolizeEnv,
    top: Top
) -> SymbolizeResult<Top> {
    match top {
        Top::StructDef {id, fields, i} => {
            let (env, id) = env.set_symbol(id);
            let (env, fields) = symbolize_vec(env, fields, symbolize_field)?;
            Ok((env, Top::StructDef {id, fields, i}))
        },
        Top::FunDef {id, params, body, i} => {
            let (env, id) = env.set_symbol(id);
            let (env, params) = symbolize_vec(env, params, symbolize_param)?;
            let (env, body) = symbolize_vec(env, body, symbolize_stmt)?;
            Ok((env, Top::FunDef {id, params, body, i}))
        },
    }
}

pub fn symbolize(ast: Ast) -> CompileResult<Ast> {
    let (_, tops) = symbolize_vec(SymbolizeEnv::default(), ast, symbolize_top)?;
    Ok(tops)
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
        SymbolizeEnv {vars}
    }

    fn sym_type_with_env(env: SymbolizeEnv, ty: Type) -> CompileResult<Type> {
        symbolize_type(&env, &Info::default(), ty)
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
        let env = sym_env(vec![]);
        assert!(symbolize_expr(&env, var("x")).is_err());
    }

    #[test]
    fn symbolize_known_var_ok() {
        let x = name("x");
        let env = sym_env(vec![x.clone()]);
        let var = symbolize_expr(&env, var("x")).unwrap();
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
        let (env, stmt) = symbolize_stmt(env, s).unwrap();
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
        let (env, stmt) = symbolize_stmt(env, s).unwrap();
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
        let (env, stmt) = symbolize_stmt(env, s).unwrap();
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
