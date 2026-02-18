use super::ast::*;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeMap;
use union_find::*;

fn unify_shape_literals(l: usize, r: usize) -> CompileResult<usize> {
    if l == 1 {
        Ok(r)
    } else if r == 1 || l == r {
        Ok(l)
    } else {
        parpy_internal_error!(Info::default(), "Failed to unify shape literals")
    }
}

#[derive(Clone, Debug)]
struct ShapeEnv {
    nvars: usize,
    var_id: BTreeMap<Name, usize>,
    shapes: BTreeMap<usize, usize>,
    uf: QuickUnionUf<UnionBySize>
}

impl ShapeEnv {
    fn get_shape_var(mut self, o: Option<&Name>) -> (Self, Shape) {
        match o {
            Some(id) => {
                match self.var_id.get(id) {
                    Some(n) => {
                        let n = *n;
                        (self, Shape::Var(n))
                    },
                    None => {
                        let n = self.nvars;
                        self.var_id.insert(id.clone(), n);
                        self.nvars += 1;
                        (self, Shape::Var(n))
                    }
                }
            },
            None => {
                let n = self.nvars;
                self.nvars += 1;
                (self, Shape::Var(n))
            }
        }
    }

    fn init_union_find(mut self) -> Self {
        self.uf = QuickUnionUf::<UnionBySize>::new(self.nvars);
        self
    }

    fn propagate_shapes(mut self) -> CompileResult<Self> {
        // We assign a concrete shape to the parent of each entry in the set of literal shapes.
        // Following this, we can use the result of 'uf.find' (the parent of a node) to determine
        // its literal shape (where a missing entry means its shape is 1).
        self.shapes = self.shapes.clone()
            .into_iter()
            .fold(Ok(self.shapes), |acc, (key, sh)| {
                let k = self.uf.find(key);
                let mut acc = acc?;
                let sh = match acc.get(&k) {
                    Some(curr_sh) => unify_shape_literals(sh, *curr_sh),
                    None => Ok(sh),
                }?;
                acc.insert(k, sh);
                Ok(acc)
            })?;
        Ok(self)
    }

    fn lookup_shape(mut self, id: usize) -> (Self, usize) {
        let k = self.uf.find(id);
        let sh = self.shapes.get(&k).cloned().unwrap_or(1);
        (self, sh)
    }
}

impl Default for ShapeEnv {
    fn default() -> Self {
        ShapeEnv {
            nvars: 0,
            var_id: BTreeMap::new(),
            shapes: BTreeMap::new(),
            uf: QuickUnionUf::<UnionBySize>::new(0),
        }
    }
}

fn add_shape_variable_type(env: ShapeEnv, ty: Type, o: Option<&Name>) -> (ShapeEnv, Type) {
    match ty {
        Type::Pointer {sz, ..} => {
            let (env, sh) = env.get_shape_var(o);
            (env, Type::Pointer {sz, shape: sh})
        },
        Type::Tensor {sz, ..} => {
            let (env, sh) = env.get_shape_var(o);
            (env, Type::Tensor {sz, shape: sh})
        },
        Type::Void => (env, ty)
    }
}

fn add_shape_variables_expr(env: ShapeEnv, e: Expr) -> (ShapeEnv, Expr) {
    match e {
        Expr::Var {id, ty, i} => {
            let (env, ty) = add_shape_variable_type(env, ty, None);
            (env, Expr::Var {id, ty, i})
        },
        _ => {
            let (env, e) = e.smap_accum_l(env, add_shape_variables_expr);
            let (env, ty) = add_shape_variable_type(env, e.get_type().clone(), None);
            (env, e.with_type(ty))
        }
    }
}

fn add_shape_variables(env: ShapeEnv, s: Stmt) -> (ShapeEnv, Stmt) {
    match s {
        Stmt::Assign {dst, expr, i} => {
            let (env, expr) = add_shape_variables_expr(env, expr);
            let ty = expr.get_type().clone();
            let (env, _) = add_shape_variable_type(env, ty, Some(&dst));
            (env, Stmt::Assign {dst, expr, i})
        },
        _ => {
            let (env, s) = s.smap_accum_l(env, add_shape_variables);
            s.smap_accum_l(env, add_shape_variables_expr)
        }
    }
}

fn extract_shape_var(sh: &Shape, i: &Info) -> CompileResult<usize> {
    match sh {
        Shape::Var(id) => Ok(*id),
        Shape::Num(_) => parpy_internal_error!(i, "Found non-variable shape type")
    }
}

fn extract_shape_type<'a>(ty: &'a Type, i: &Info) -> CompileResult<&'a Shape> {
    match ty {
        Type::Pointer {shape, ..} |
        Type::Tensor {shape, ..} => Ok(shape),
        _ => parpy_internal_error!(i, "Failed to extract shape variable from type")
    }
}

fn unify_shapes(
    mut env: ShapeEnv,
    lsh: &Shape,
    rsh: &Shape,
    i: &Info
) -> CompileResult<ShapeEnv> {
    let lid = extract_shape_var(lsh, i)?;
    let rid = extract_shape_var(rsh, i)?;
    env.uf.union(lid, rid);
    Ok(env)
}

fn unify_shapes_type(
    env: ShapeEnv,
    lty: &Type,
    rty: &Type,
    i: &Info
) -> CompileResult<ShapeEnv> {
    let lsh = extract_shape_type(lty, i)?;
    let rsh = extract_shape_type(rty, i)?;
    unify_shapes(env, lsh, rsh, i)
}

fn set_fixed_shape(
    mut env: ShapeEnv,
    ty: &Type, 
    sh: usize,
    i: &Info
) -> CompileResult<ShapeEnv> {
    let ty_sh = match ty.get_shape() {
        Some(sh) => Ok(sh),
        None => parpy_compile_error!(i, "Failed to extract shape of type")
    }?;
    let id = extract_shape_var(ty_sh, &i)?;
    env.shapes.insert(id, sh);
    Ok(env)
}

fn unify_shapes_expr(env: ShapeEnv, e: &Expr) -> CompileResult<ShapeEnv> {
    match e {
        Expr::UnOp {arg, ty, i, ..} => {
            let env = unify_shapes_expr(env, arg)?;
            unify_shapes_type(env, arg.get_type(), &ty, &i)
        },
        Expr::BinOp {lhs, rhs, ty, i, ..} => {
            let env = unify_shapes_expr(env, lhs)?;
            let env = unify_shapes_expr(env, rhs)?;
            let env = unify_shapes_type(env, lhs.get_type(), rhs.get_type(), &i)?;
            unify_shapes_type(env, &ty, lhs.get_type(), &i)
        },
        Expr::Reduce {arg, ty, i, ..} => {
            let env = unify_shapes_expr(env, arg)?;
            set_fixed_shape(env, ty, 1, &i)
        },
        Expr::Arange {lo, hi, ty, i} => {
            let sh = hi - lo;
            set_fixed_shape(env, ty, sh, &i)
        },
        Expr::Load {ptr, ty, i, ..} => {
            let env = unify_shapes_expr(env, ptr)?;
            unify_shapes_type(env, ptr.get_type(), &ty, &i)
        },
        Expr::Store {ptr, value, i, ..} => {
            let env = unify_shapes_expr(env, ptr)?;
            let env = unify_shapes_expr(env, value)?;
            unify_shapes_type(env, ptr.get_type(), value.get_type(), &i)
        },
        Expr::Full {value, ty, i, ..} => {
            let env = unify_shapes_expr(env, value)?;
            unify_shapes_type(env, value.get_type(), &ty, &i)
        },
        Expr::Where {cond, thn, els, ty, i} => {
            let env = unify_shapes_expr(env, cond)?;
            let env = unify_shapes_expr(env, thn)?;
            let env = unify_shapes_expr(env, els)?;
            let env = unify_shapes_type(env, cond.get_type(), thn.get_type(), &i)?;
            let env = unify_shapes_type(env, cond.get_type(), els.get_type(), &i)?;
            unify_shapes_type(env, cond.get_type(), &ty, &i)
        },
        Expr::Convert {value, ty, i, ..} => {
            let env = unify_shapes_expr(env, value)?;
            unify_shapes_type(env, value.get_type(), &ty, &i)
        },
        _ => e.sfold_result(Ok(env), unify_shapes_expr)
    }
}

fn unify_shapes_stmt(env: ShapeEnv, s: &Stmt) -> CompileResult<ShapeEnv> {
    match s {
        Stmt::Assign {dst, expr, i} => {
            let env = unify_shapes_expr(env, expr)?;
            let (env, lsh) = env.get_shape_var(Some(&dst));
            let rsh = extract_shape_type(expr.get_type(), &i)?;
            unify_shapes(env, &lsh, rsh, &i)
        },
        Stmt::For {lo, hi, body, i, ..} => {
            let env = unify_shapes_expr(env, lo)?;
            let env = unify_shapes_expr(env, hi)?;
            let env = unify_shapes_type(env, lo.get_type(), hi.get_type(), &i);
            body.sfold_result(env, unify_shapes_stmt)
        },
        _ => {
            let env = s.sfold_result(Ok(env), unify_shapes_stmt);
            s.sfold_result(env, unify_shapes_expr)
        }
    }
}

fn unify_shapes_body(env: ShapeEnv, body: &Vec<Stmt>) -> CompileResult<ShapeEnv> {
    let env = env.init_union_find();
    let env = body.sfold_result(Ok(env), unify_shapes_stmt)?;
    env.propagate_shapes()
}

fn determine_shapes_type(env: ShapeEnv, ty: Type) -> (ShapeEnv, Type) {
    match ty {
        Type::Pointer {sz, shape: Shape::Var(id)} => {
            let (env, n) = env.lookup_shape(id);
            (env, Type::Pointer {sz, shape: Shape::Num(n)})
        },
        Type::Tensor {sz, shape: Shape::Var(id)} => {
            let (env, n) = env.lookup_shape(id);
            (env, Type::Tensor {sz, shape: Shape::Num(n)})
        },
        _ => (env, ty)
    }
}

fn extract_type_shape(ty: &Type, i: &Info) -> CompileResult<usize> {
    match ty.get_shape() {
        Some(Shape::Num(n)) => Ok(*n),
        _ => parpy_compile_error!(i, "Failed to determine shape of expression")
    }
}

fn determine_shapes_expr(env: ShapeEnv, e: Expr) -> CompileResult<(ShapeEnv, Expr)> {
    let (env, ty) = determine_shapes_type(env, e.get_type().clone());
    match e {
        Expr::Var {..} => Ok((env, e.with_type(ty))),
        Expr::Full {shape: _, value, elem_sz, ty: _, i} => {
            let (env, value) = determine_shapes_expr(env, *value)?;
            let shape = extract_type_shape(&ty, &i)?;
            Ok((env, Expr::Full {shape, value: Box::new(value), elem_sz, ty, i}))
        },
        _ => {
            let (env, e) = e.smap_accum_l_result(Ok(env), determine_shapes_expr)?;
            Ok((env, e.with_type(ty)))
        }
    }
}

fn determine_shapes_stmt(env: ShapeEnv, s: Stmt) -> CompileResult<(ShapeEnv, Stmt)> {
    let (env, s) = s.smap_accum_l_result(Ok(env), determine_shapes_stmt)?;
    s.smap_accum_l_result(Ok(env), determine_shapes_expr)
}

fn unify_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            let (env, body) = body.smap_accum_l(ShapeEnv::default(), add_shape_variables);
            let env = unify_shapes_body(env, &body)?;
            let (_, body) = body.smap_accum_l_result(Ok(env), determine_shapes_stmt)?;
            Ok(Top::FunDef {triton_jit: true, id, params, body, i})
        },
        Top::Import {..} | Top::FunDef {..} => Ok(t),
    }
}

pub fn unify(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(unify_top)
}
