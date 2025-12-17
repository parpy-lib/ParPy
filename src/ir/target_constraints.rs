use super::ast::*;
use crate::parpy_compile_error;
use crate::parpy_internal_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum TargetConstraint {
    HostOnly, DeviceOnly, Either
}

impl TargetConstraint {
    fn from_target(target: &Target) -> Self {
        match target {
            Target::Host => TargetConstraint::HostOnly,
            Target::Device => TargetConstraint::DeviceOnly,
        }
    }
}

#[derive(Debug)]
struct TargetConstraintEnv {
    mapping: BTreeMap<Name, TargetConstraint>,
    id: Name,
    i: Info
}

impl Default for TargetConstraintEnv {
    fn default() -> Self {
        TargetConstraintEnv {
            mapping: BTreeMap::new(),
            id: Name::new("".to_string()),
            i: Info::default()
        }
    }
}

impl TargetConstraintEnv {
    fn enter_function(mut self, id: &Name, i: &Info) -> Self {
        self.id = id.clone();
        self.i = i.clone();
        self
    }

    fn unify_constraints(
        &self,
        l: TargetConstraint,
        r: TargetConstraint
    ) -> CompileResult<TargetConstraint> {
        match (&l, &r) {
            (TargetConstraint::HostOnly, TargetConstraint::HostOnly) |
            (TargetConstraint::DeviceOnly, TargetConstraint::DeviceOnly) |
            (_, TargetConstraint::Either) => {
                Ok(l)
            },
            (TargetConstraint::Either, _) => Ok(r),
            (TargetConstraint::HostOnly, TargetConstraint::DeviceOnly) |
            (TargetConstraint::DeviceOnly, TargetConstraint::HostOnly) => {
                parpy_compile_error!(
                    self.i,
                    "Function {0} contains calls to host functions and to \
                     device functions, which is not allowed",
                    self.id
                )
            }
        }
    }
}

fn collect_expr(
    env: &TargetConstraintEnv,
    acc: TargetConstraint,
    e: &Expr
) -> CompileResult<TargetConstraint> {
    match e {
        Expr::Call {id, i, ..} => {
            match env.mapping.get(&id) {
                Some(cls) => env.unify_constraints(acc, cls.clone()),
                None => {
                    parpy_internal_error!(i, "Found call to function {id} with \
                                              unknown constraints")
                }
            }
        },
        Expr::PyCallback {..} => {
            env.unify_constraints(acc, TargetConstraint::HostOnly)
        },
        _ => e.sfold_result(Ok(acc), |acc, e| collect_expr(&env, acc, e))
    }
}

fn collect_stmt(
    env: &TargetConstraintEnv,
    acc: TargetConstraint,
    s: &Stmt
) -> CompileResult<TargetConstraint> {
    let acc = s.sfold_result(Ok(acc), |acc, s| collect_stmt(&env, acc, s));
    s.sfold_result(acc, |acc, e| collect_expr(&env, acc, e))
}

fn collect_top(
    mut env: TargetConstraintEnv,
    t: &Top
) -> CompileResult<TargetConstraintEnv> {
    match t {
        Top::ExtDecl {id, target, ..} => {
            let c = TargetConstraint::from_target(target);
            env.mapping.insert(id.clone(), c);
            Ok(env)
        },
        Top::FunDef {v: FunDef {id, body, i, ..}} => {
            let mut env = env.enter_function(&id, &i);
            let c = body.sfold_result(Ok(TargetConstraint::Either), |c, s| {
                collect_stmt(&env, c, s)
            })?;
            env.mapping.insert(id.clone(), c);
            Ok(env)
        }
    }
}

pub fn collect(ast: &Ast) -> CompileResult<BTreeMap<Name, TargetConstraint>> {
    let env = ast.tops.sfold_result(Ok(TargetConstraintEnv::default()), collect_top)?;
    Ok(env.mapping)
}
