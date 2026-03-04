use super::ast::*;
use crate::parpy_internal_error;
use crate::utils::ast::ExprType;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use rustsat::instances::SatInstance;
use rustsat::types::*;
use rustsat::solvers::{Solve, SolverResult};
use rustsat::solvers::external;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::process::Command;

fn type_with_shape(
    ty: Type,
    shape: Shape,
    i: &Info
) -> CompileResult<Type> {
    match ty {
        Type::Pointer {ty, ..} => Ok(Type::Pointer {ty, shape}),
        Type::Tensor {sz, ..} => Ok(Type::Tensor {sz, shape}),
        Type::Function {..} => parpy_internal_error!(i, "Cannot set shape of function type"),
        Type::Void => parpy_internal_error!(i, "Cannot set shape of void type")
    }
}

fn get_type_shape(ty: &Type, i: &Info) -> CompileResult<Shape> {
    match ty {
        Type::Pointer {shape, ..} |
        Type::Tensor {shape, ..} => Ok(shape.clone()),
        Type::Function {..} => parpy_internal_error!(i, "Cannot get shape of function type"),
        Type::Void => parpy_internal_error!(i, "Cannot get shape of void type")
    }
}

#[derive(Clone, Debug)]
enum Constraint {
    Geq {l: Shape, r: Shape},
    Max {var: usize, l: Shape, r: Shape},
}

#[derive(Clone, Debug)]
struct ShapeEnv {
    nvars: usize,
    vars: BTreeMap<Name, usize>,
    constraints: Vec<Constraint>,
    cf_conds: Vec<Shape>,
}

impl Default for ShapeEnv {
    fn default() -> Self {
        ShapeEnv {
            nvars: 0,
            vars: BTreeMap::new(),
            constraints: vec![],
            cf_conds: vec![]
        }
    }
}

impl ShapeEnv {
    fn lookup_shape(mut self, id: &Name) -> (Self, Shape) {
        match self.vars.get(&id) {
            Some(sz) => {
                let n = *sz;
                (self, Shape::Var(n))
            },
            None => {
                let n = self.nvars;
                self.vars.insert(id.clone(), n);
                self.nvars += 1;
                (self, Shape::Var(n))
            }
        }
    }

    fn add_geq_constraint(mut self, l: Shape, r: Shape) -> Self {
        self.constraints.push(Constraint::Geq {l, r});
        self
    }

    fn add_max_constraint(
        mut self,
        l: Shape,
        r: Shape
    ) -> (Self, Shape) {
        let n = self.nvars;
        self.nvars += 1;
        self.constraints.push(Constraint::Max {var: n, l, r});
        (self, Shape::Var(n))
    }

    fn push_control_flow_condition(mut self, cond_shape: Shape) -> Self {
        self.cf_conds.push(cond_shape);
        self
    }

    fn pop_control_flow_condition(mut self) -> Self {
        self.cf_conds.pop();
        self
    }
}

fn add_shape_variables_expr(env: ShapeEnv, e: Expr) -> CompileResult<(ShapeEnv, Expr)> {
    match e {
        Expr::Var {id, ty, i} => {
            let (env, shape) = env.lookup_shape(&id);
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::Var {id, ty, i}))
        },
        Expr::Bool {v, ty, i} => {
            let ty = type_with_shape(ty, Shape::Num(1), &i)?;
            Ok((env, Expr::Bool {v, ty, i}))
        },
        Expr::Int {v, ty, i} => {
            let ty = type_with_shape(ty, Shape::Num(1), &i)?;
            Ok((env, Expr::Int {v, ty, i}))
        },
        Expr::Float {v, ty, i} => {
            let ty = type_with_shape(ty, Shape::Num(1), &i)?;
            Ok((env, Expr::Float {v, ty, i}))
        },
        Expr::UnOp {op, arg, ty, i} => {
            let (env, arg) = add_shape_variables_expr(env, *arg)?;
            let shape = get_type_shape(arg.get_type(), &i)?;
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::UnOp {op, arg: Box::new(arg), ty, i}))
        },
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            let (env, lhs) = add_shape_variables_expr(env, *lhs)?;
            let (env, rhs) = add_shape_variables_expr(env, *rhs)?;
            let l = get_type_shape(lhs.get_type(), &i)?;
            let r = get_type_shape(rhs.get_type(), &i)?;
            let (env, shape) = env.add_max_constraint(l, r);
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}))
        },
        Expr::Reduce {op, arg, ty, i} => {
            let (env, arg) = add_shape_variables_expr(env, *arg)?;
            let ty = type_with_shape(ty, Shape::Num(1), &i)?;
            Ok((env, Expr::Reduce {op, arg: Box::new(arg), ty, i}))
        },
        Expr::Call {id, args, ty, i} => {
            let (env, args) = args.smap_accum_l_result(Ok(env), add_shape_variables_expr)?;
            Ok((env, Expr::Call {id, args, ty, i}))
        },
        Expr::ExtCall {id, args, ty, i} => {
            let (env, args) = args.smap_accum_l_result(Ok(env), add_shape_variables_expr)?;
            Ok((env, Expr::ExtCall {id, args, ty, i}))
        },
        Expr::ProgramId {dim, ty, i} => {
            let ty = type_with_shape(ty, Shape::Num(1), &i)?;
            Ok((env, Expr::ProgramId {dim, ty, i}))
        },
        Expr::Arange {lo, hi, ty, i} => {
            let extract_integer_literal = |e: &Expr, i: &Info| match e {
                Expr::Int {v, ..} => Ok(v.clone()),
                _ => parpy_internal_error!(i, "Found non-literal bound of arange in the \
                                               shape analysis of the Triton backend")
            };
            let l = extract_integer_literal(&lo, &i)?;
            let h = extract_integer_literal(&hi, &i)?;
            let ty = type_with_shape(ty, Shape::Num((h-l) as usize), &i)?;
            Ok((env, Expr::Arange {lo, hi, ty, i}))
        },
        Expr::Load {ptr, mask, ty, i} => {
            let (env, ptr) = add_shape_variables_expr(env, *ptr)?;
            let (env, mask) = mask.smap_accum_l_result(Ok(env), add_shape_variables_expr)?;
            let (env, shape) = match mask {
                Some(ref m) => {
                    let l = get_type_shape(ptr.get_type(), &i)?;
                    let r = get_type_shape(m.get_type(), &i)?;
                    let (env, shape) = env.add_max_constraint(l, r);
                    Ok((env, shape))
                },
                None => Ok((env, get_type_shape(ptr.get_type(), &i)?)),
            }?;
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::Load {ptr: Box::new(ptr), mask, ty, i}))
        },
        Expr::Full {shape, value, elem_sz, ty, i} => {
            let (env, value) = add_shape_variables_expr(env, *value)?;
            let ty = type_with_shape(ty, Shape::Num(shape), &i)?;
            Ok((env, Expr::Full {shape, value: Box::new(value), elem_sz, ty, i}))
        },
        Expr::Where {cond, thn, els, ty, i} => {
            let (env, cond) = add_shape_variables_expr(env, *cond)?;
            let (env, thn) = add_shape_variables_expr(env, *thn)?;
            let (env, els) = add_shape_variables_expr(env, *els)?;
            let cs = get_type_shape(cond.get_type(), &i)?;
            let ts = get_type_shape(thn.get_type(), &i)?;
            let es = get_type_shape(els.get_type(), &i)?;
            let (env, interm_shape) = env.add_max_constraint(cs, ts);
            let (env, shape) = env.add_max_constraint(interm_shape, es);
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::Where {
                cond: Box::new(cond), thn: Box::new(thn), els: Box::new(els), ty, i
            }))
        },
        Expr::Convert {value, ty, i} => {
            let (env, value) = add_shape_variables_expr(env, *value)?;
            let shape = get_type_shape(value.get_type(), &i)?;
            let ty = type_with_shape(ty, shape, &i)?;
            Ok((env, Expr::Convert {value: Box::new(value), ty, i}))
        },
        Expr::AllocBuffer {..} |
        Expr::ToTorch {..} => Ok((env, e)),
    }
}

fn max_cond_shape(env: ShapeEnv, sh: Shape) -> (ShapeEnv, Shape) {
    env.cf_conds.clone()
        .into_iter()
        .fold((env, sh), |(env, l), r| env.add_max_constraint(l, r))
}

fn add_shape_variables_assign(
    env: ShapeEnv,
    dst: Name,
    expr: Expr,
    i: Info
) -> CompileResult<(ShapeEnv, (Name, Expr, Info))> {
    let (env, expr) = add_shape_variables_expr(env, expr)?;
    let expr_shape = get_type_shape(expr.get_type(), &i)?;
    // NOTE(larshum, 2026-02-23): If any outer conditions have a non-unit shape, this impacts the
    // resulting shape of this assignment. Therefore, we consider the resulting shape of the
    // variable to be the maximum of the RHS shape and that of any outer conditions.
    let (env, rhs_shape) = max_cond_shape(env, expr_shape);
    let (env, shape) = env.lookup_shape(&dst);
    let env = env.add_geq_constraint(shape.clone(), rhs_shape);
    Ok((env, (dst, expr, i)))
}

fn add_shape_variables_stmt(env: ShapeEnv, s: Stmt) -> CompileResult<(ShapeEnv, Stmt)> {
    match s {
        Stmt::Definition {dst, expr, i} => {
            let (env, (dst, expr, i)) = add_shape_variables_assign(env, dst, expr, i)?;
            Ok((env, Stmt::Definition {dst, expr, i}))
        },
        Stmt::Assign {dst, expr, i} => {
            let (env, (dst, expr, i)) = add_shape_variables_assign(env, dst, expr, i)?;
            Ok((env, Stmt::Assign {dst, expr, i}))
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let (env, lo) = add_shape_variables_expr(env, lo)?;
            let (env, hi) = add_shape_variables_expr(env, hi)?;
            let lo_shape = get_type_shape(lo.get_type(), &i)?;
            let hi_shape = get_type_shape(hi.get_type(), &i)?;
            let (env, cond_shape) = env.add_max_constraint(lo_shape, hi_shape);
            let (env, shape) = env.lookup_shape(&var);
            let env = env.add_geq_constraint(shape, cond_shape.clone());
            let env = env.push_control_flow_condition(cond_shape);
            let (env, body) = body.smap_accum_l_result(Ok(env), add_shape_variables_stmt)?;
            let env = env.pop_control_flow_condition();
            Ok((env, Stmt::For {var, lo, hi, step, body, i}))
        },
        Stmt::While {cond, body, i} => {
            let (env, cond) = add_shape_variables_expr(env, cond)?;
            let cond_shape = get_type_shape(cond.get_type(), &i)?;
            let env = env.push_control_flow_condition(cond_shape);
            let (env, body) = body.smap_accum_l_result(Ok(env), add_shape_variables_stmt)?;
            let env = env.pop_control_flow_condition();
            Ok((env, Stmt::While {cond, body, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let (env, cond) = add_shape_variables_expr(env, cond)?;
            let cond_shape = get_type_shape(cond.get_type(), &i)?;
            let env = env.push_control_flow_condition(cond_shape);
            let (env, thn) = thn.smap_accum_l_result(Ok(env), add_shape_variables_stmt)?;
            let (env, els) = els.smap_accum_l_result(Ok(env), add_shape_variables_stmt)?;
            let env = env.pop_control_flow_condition();
            Ok((env, Stmt::If {cond, thn, els, i}))
        },
        Stmt::Store {ptr, value, mask, i} => {
            let (env, ptr) = add_shape_variables_expr(env, ptr)?;
            let (env, value) = add_shape_variables_expr(env, value)?;
            let (env, mask) = match mask {
                Some(m) => {
                    let (env, m) = add_shape_variables_expr(env, m)?;
                    Ok((env, Some(m)))
                },
                None => Ok((env, None))
            }?;
            Ok((env, Stmt::Store {ptr, value, mask, i}))
        },
        _ => {
            let (env, s) = s.smap_accum_l_result(Ok(env), add_shape_variables_stmt)?;
            s.smap_accum_l_result(Ok(env), add_shape_variables_expr)
        }
    }
}

fn add_lit_shape(mut acc: BTreeSet<usize>, shape: &Shape) -> BTreeSet<usize> {
    match shape {
        Shape::Var(_) => acc,
        Shape::Num(n) => {
            if *n > 1 { acc.insert(*n); }
            acc
        }
    }
}

fn validate_constraints(env: &ShapeEnv) -> CompileResult<Option<usize>> {
    let literals = env.constraints.iter()
        .fold(BTreeSet::new(), |acc, c| match c {
            Constraint::Geq {l, r} | Constraint::Max {l, r, var: _} => {
                add_lit_shape(add_lit_shape(acc, l), r)
            },
        });
    match literals.len() {
        0 | 1 => Ok(literals.first().cloned()),
        _ => parpy_internal_error!(Info::default(), "Found conflicting block sizes in Triton shape analysis")
    }
}

fn solve_constraints(env: ShapeEnv, n: usize) -> CompileResult<Vec<usize>> {
    let mut instance: SatInstance = SatInstance::new();
    let false_lit = instance.new_lit();
    instance.add_unit(!false_lit);
    let true_lit = instance.new_lit();
    instance.add_unit(true_lit);
    let vars = (0..env.nvars)
        .map(|_| instance.new_var())
        .collect::<Vec<Var>>();
    let get_lit = |shape: &Shape| match shape {
        Shape::Var(n) => vars[*n].pos_lit(),
        Shape::Num(n) => if *n == 1 { false_lit } else { true_lit },
    };
    env.constraints.iter()
        .for_each(|c| match c {
            Constraint::Geq {l, r} => {
                let lv = get_lit(l);
                let rv = get_lit(r);
                instance.add_binary(!rv, lv);
            },
            Constraint::Max {var, l, r} => {
                let v = vars[*var].pos_lit();
                let lv = get_lit(l);
                let rv = get_lit(r);
                instance.add_ternary(!v, lv, rv);
                instance.add_binary(v, !lv);
                instance.add_binary(v, !rv);
            },
        });
    // NOTE(larshum, 2026-03-02): The rustsat library uses the DIMACS format when communicating
    // with extenal solvers. To make the Z3 solver use this format, we have to provide the input in
    // a file using the extension ".dimacs".
    let (_dir, tempfile) = if let Ok(td) = tempfile::tempdir() {
        let fp = td.path().join("temp.dimacs");
        if let Ok(_) = File::create(&fp) {
            Ok((td, fp))
        } else {
            parpy_internal_error!(Info::default(), "Failed to create temporary file for solver")
        }
    } else {
        parpy_internal_error!(Info::default(), "Failed to allocate temporary directory")
    }?;
    // NOTE(larshum, 2026-03-02): We use the Z3 solver binary as an external solver to determine
    // which shape variables should be blocked. This binary should be available after installing
    // the package "z3-solver" which is one of our dependencies.
    which::which("z3")
        .map_err(|_| CompileError::internal_err(
            "Could not find the z3 binary, which is required by the Triton backend.\n\
             This should be installed via the \"z3-solver\" package, which is \
             a dependency of ParPy.".to_string()
        ))?;
    let mut solver = external::Solver::new(
        Command::new("z3"),
        external::InputVia::file_last(tempfile),
        external::OutputVia::pipe(),
        "z3-solver"
    );
    solver.add_cnf(instance.into_cnf().0).unwrap();
    match solver.solve() {
        Ok(SolverResult::Sat) => {
            vars.into_iter()
                .map(|v| solver.lit_val(v.pos_lit()))
                .collect::<Result<Vec<TernaryVal>, _>>()
                .map(|vals| {
                    vals.into_iter()
                        .map(|v| if v == TernaryVal::True { n } else { 1 })
                        .collect::<Vec<usize>>()
                })
                .map_err(|_| {
                    CompileError::internal_err(
                        "Solver failed to determine block shapes of variables".to_string()
                    )
                })
        },
        Ok(_) => {
            parpy_internal_error!(Info::default(), "Failed to determine blocking of variables")
        }
        Err(e) => parpy_internal_error!(Info::default(), "Solver failed: {e}")
    }
}

fn determine_shape(
    mapping: &Vec<usize>,
    shape: Shape
) -> Shape {
    match shape {
        Shape::Var(v) => Shape::Num(mapping[v]),
        Shape::Num(_) => shape,
    }
}

fn determine_type(
    mapping: &Vec<usize>,
    ty: Type
) -> Type {
    match ty {
        Type::Pointer {ty, shape} => {
            let ty = determine_type(&mapping, *ty);
            let shape = determine_shape(&mapping, shape);
            Type::Pointer {ty: Box::new(ty), shape}
        },
        Type::Tensor {sz, shape} => {
            let shape = determine_shape(&mapping, shape);
            Type::Tensor {sz, shape}
        },
        Type::Function {..} |
        Type::Void => ty
    }
}

fn replace_vars_expr(
    mapping: &Vec<usize>,
    e: Expr
) -> Expr {
    let ty = determine_type(&mapping, e.get_type().clone());
    let e = e.with_type(ty);
    e.smap(|e| replace_vars_expr(&mapping, e))
}

fn replace_vars_stmt(mapping: &Vec<usize>, s: Stmt) -> Stmt {
    s.smap(|s| replace_vars_stmt(&mapping, s))
        .smap(|e| replace_vars_expr(&mapping, e))
}

fn explicitly_broadcast_blocked_variable_assignment(
    vars: &BTreeMap<Name, usize>,
    s: Stmt
) -> CompileResult<Stmt> {
    match s {
        Stmt::Definition {dst, expr, i} => {
            let expected_shape = vars.get(&dst).cloned().unwrap_or(1);
            let rhs_shape = match get_type_shape(expr.get_type(), &i)? {
                Shape::Num(n) => Ok(n),
                Shape::Var(_) => parpy_internal_error!(i, "Triton shape analysis failed \
                                                           to resolve shapes of expression")
            }?;
            if expected_shape > rhs_shape {
                let ty = expr.get_type().clone();
                let elem_sz = match ty.get_elem_size() {
                    Some(sz) => Ok(sz.clone()),
                    None => parpy_internal_error!(i, "Failed to extract element size of type")
                }?;
                let expr = Expr::Full {
                    shape: expected_shape,
                    value: Box::new(expr),
                    elem_sz,
                    ty,
                    i: i.clone()
                };
                Ok(Stmt::Definition {dst, expr, i})
            } else {
                Ok(Stmt::Definition {dst, expr, i})
            }
        },
        _ => s.smap_result(|s| explicitly_broadcast_blocked_variable_assignment(&vars, s))
    }
}

fn explicitly_broadcast_variable_assignments(
    vars: &BTreeMap<Name, usize>,
    body: Vec<Stmt>
) -> CompileResult<Vec<Stmt>> {
    body.smap_result(|s| {
        explicitly_broadcast_blocked_variable_assignment(&vars, s)
    })
}

fn rewrite_blocked_for_loops_stmt(
    vars: &BTreeMap<Name, usize>,
    mut acc: Vec<Stmt>,
    s: Stmt
) -> CompileResult<Vec<Stmt>> {
    match s {
        Stmt::For {var, lo, hi, step, body, i} => {
            let mut body = body.sflatten_result(vec![], |acc, s| {
                rewrite_blocked_for_loops_stmt(&vars, acc, s)
            })?;
            let shape = vars.get(&var).cloned().unwrap_or(1);
            if shape > 1 {
                let shape = Shape::Num(shape);
                let sz = match lo.get_type().get_elem_size() {
                    Some(sz) => Ok(sz.clone()),
                    None => parpy_internal_error!(i, "Failed to extract element size of void type")
                }?;
                let ty = Type::Tensor {sz, shape: shape.clone()};
                let var_expr = Expr::Var {
                    id: var.clone(), ty: ty.clone(), i: i.clone()
                };
                acc.push(Stmt::Definition {
                    dst: var.clone(),
                    expr: lo,
                    i: i.clone()
                });
                body.push(Stmt::Assign {
                    dst: var,
                    expr: Expr::BinOp {
                        lhs: Box::new(var_expr.clone()),
                        op: BinOp::Add,
                        rhs: Box::new(Expr::Int {
                            v: step as i128, ty: ty.clone(), i: i.clone()
                        }),
                        ty: ty.clone(),
                        i: i.clone()
                    },
                    i: i.clone()
                });
                let cond = Expr::BinOp {
                    lhs: Box::new(var_expr),
                    op: BinOp::Lt,
                    rhs: Box::new(hi),
                    ty: Type::Tensor {sz: ElemSize::Bool, shape},
                    i: i.clone()
                };
                acc.push(Stmt::While {cond, body, i: i.clone()});
            } else {
                acc.push(Stmt::For {var, lo, hi, step, body, i});
            }
            Ok(acc)
        },
        _ => s.sflatten_result(acc, |acc, s| {
            rewrite_blocked_for_loops_stmt(vars, acc, s)
        })
    }
}

fn unify_top(t: Top) -> CompileResult<Top> {
    match t {
        Top::FunDef {triton_jit: true, id, params, body, i} => {
            let env = Ok(ShapeEnv::default());
            let (env, body) = body.smap_accum_l_result(env, add_shape_variables_stmt)?;
            let vars = env.vars.clone();
            let var_mapping = match validate_constraints(&env)? {
                Some(n) => solve_constraints(env, n),
                None => {
                    // If the body contains no blocked expressions, we set all shape variables to
                    // one.
                    Ok((0..env.nvars)
                        .map(|_| 1)
                        .collect::<Vec<usize>>())
                },
            }?;
            let body = body.smap(|s| replace_vars_stmt(&var_mapping, s));
            let vars = vars.into_iter()
                .map(|(k, v)| (k, var_mapping[v]))
                .collect::<BTreeMap<Name, usize>>();
            let body = body.sflatten_result(vec![], |acc, s| {
                rewrite_blocked_for_loops_stmt(&vars, acc, s)
            })?;
            let body = explicitly_broadcast_variable_assignments(&vars, body)?;
            Ok(Top::FunDef {triton_jit: true, id, params, body, i})
        },
        Top::Import {..} | Top::FunDef {..} => Ok(t),
    }
}

pub fn unify(ast: Ast) -> CompileResult<Ast> {
    ast.smap_result(unify_top)
}
