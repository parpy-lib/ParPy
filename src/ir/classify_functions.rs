use super::ast::*;
use super::target_constraints::TargetConstraint;
use crate::parpy_compile_error;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum TargetClass {
    Host, Device, Both
}

impl From<&Target> for TargetClass {
    fn from(target: &Target) -> Self {
        match target {
            Target::Host => TargetClass::Host,
            Target::Device => TargetClass::Device
        }
    }
}

#[derive(Debug)]
struct ClassifyEnv<'a> {
    mapping: &'a BTreeMap<Name, TargetConstraint>,
    classification: BTreeMap<Name, TargetClass>,
    target: Target,
}

impl<'a> ClassifyEnv<'a> {
    fn new(mapping: &'a BTreeMap<Name, TargetConstraint>) -> ClassifyEnv<'a> {
        ClassifyEnv {
            mapping,
            classification: BTreeMap::new(),
            target: Target::Host,
        }
    }

    fn unify(
        &self,
        l: &TargetClass,
        r: &TargetClass
    ) -> TargetClass {
        match (l, r) {
            (TargetClass::Host, TargetClass::Host) => TargetClass::Host,
            (TargetClass::Device, TargetClass::Device) => TargetClass::Device,
            (TargetClass::Both, _) | (_, TargetClass::Both) => {
                TargetClass::Both
            },
            (TargetClass::Host, TargetClass::Device) |
            (TargetClass::Device, TargetClass::Host) => {
                TargetClass::Both
            }
        }
    }

    fn with_target(mut self, target: Target) -> (ClassifyEnv<'a>, Target) {
        let old_target = self.target;
        self.target = target;
        (self, old_target)
    }
}

fn classify_expr<'a>(
    mut env: ClassifyEnv<'a>,
    e: &Expr
) -> CompileResult<ClassifyEnv<'a>> {
    match e {
        Expr::Call {id, args, ty, i, ..} => {
            if let Some(TargetConstraint::HostOnly) = env.mapping.get(&id) {
                if env.target == Target::Device && *ty != Type::Void {
                    parpy_compile_error!(
                        i,
                        "Host function returning a non-void value cannot be \
                         called from parallel code."
                    )
                } else {
                    Ok(())
                }
            } else {
                Ok(())
            }?;
            let mut env = args.sfold_result(Ok(env), classify_expr)?;
            let tcl = TargetClass::from(&env.target);
            let class = match env.classification.get(&id) {
                Some(cl) => env.unify(cl, &tcl),
                None => tcl
            };
            env.classification.insert(id.clone(), class);
            Ok(env)
        },
        Expr::PyCallback {id, i, ..} => {
            if let Target::Host = &env.target {
                let tcl = TargetClass::from(&env.target);
                let class = match env.classification.get(&id) {
                    Some(cl) => env.unify(cl, &tcl),
                    None => tcl
                };
                env.classification.insert(id.clone(), class);
                Ok(env)
            } else {
                parpy_compile_error!(i, "Cannot invoke Python callback function \
                                         from code executing on device")
            }
        },
        _ => e.sfold_result(Ok(env), classify_expr)
    }
}

fn classify_stmt<'a>(
    env: ClassifyEnv<'a>,
    s: &Stmt
) -> CompileResult<ClassifyEnv<'a>> {
    match s {
        Stmt::For {lo, hi, body, par, ..} if par.is_parallel() => {
            let env = classify_expr(env, &lo)?;
            let env = classify_expr(env, &hi)?;
            let (env, old_target) = env.with_target(Target::Device);
            let env = body.sfold_result(Ok(env), classify_stmt)?;
            let (env, _) = env.with_target(old_target);
            Ok(env)
        },
        Stmt::AllocShared {..} => {
            let (env, _) = env.with_target(Target::Device);
            Ok(env)
        },
        _ => {
            let env = s.sfold_result(Ok(env), classify_stmt);
            s.sfold_result(env, classify_expr)
        }
    }
}

fn classify_fun_def<'a>(
    env: ClassifyEnv<'a>,
    def: &FunDef
) -> CompileResult<ClassifyEnv<'a>> {
    match env.classification.get(&def.id) {
        None | Some(TargetClass::Host) => {
            let (env, _) = env.with_target(Target::Host);
            def.body.sfold_result(Ok(env), classify_stmt)
        },
        Some(TargetClass::Device) => {
            let (env, _) = env.with_target(Target::Device);
            def.body.sfold_result(Ok(env), classify_stmt)
        },
        Some(TargetClass::Both) => {
            let (env, _) = env.with_target(Target::Host);
            let env = def.body.sfold_result(Ok(env), classify_stmt)?;
            let (env, _) = env.with_target(Target::Device);
            def.body.sfold_result(Ok(env), classify_stmt)
        }
    }
}

fn classify_top<'a>(
    env: ClassifyEnv<'a>,
    t: &Top
) -> CompileResult<ClassifyEnv<'a>> {
    match t {
        Top::ExtDecl {..} => Ok(env),
        Top::FunDef {v} => classify_fun_def(env, &v)
    }
}

fn unify_constraint(
    cl: TargetClass,
    co: &TargetConstraint,
    id: &Name
) -> CompileResult<TargetClass> {
    // The class of a function is determined based on how it is called by other functions. The
    // constraint is determined by how its body calls other functions. The former must satisfy the
    // constraints imposed by the latter to be able to unify them.
    match co {
        TargetConstraint::HostOnly => {
            match cl {
                TargetClass::Host => Ok(TargetClass::Host),
                TargetClass::Device | TargetClass::Both => {
                    parpy_compile_error!(
                        Info::default(),
                        "Function {id} can only be called from host code \
                         (outside parallel loop nests), but found calls from \
                         device code."
                    )
                },
            }
        },
        TargetConstraint::DeviceOnly => {
            match cl {
                TargetClass::Device => Ok(TargetClass::Device),
                TargetClass::Host | TargetClass::Both => {
                    parpy_compile_error!(
                        Info::default(),
                        "Function {id} can only be called from device code \
                         (within parallel loop nests), but found calls from \
                         host code."
                    )
                }
            }
        },
        TargetConstraint::Either => Ok(cl)
    }
}

fn validate_classification<'a>(
    env: ClassifyEnv<'a>
) -> CompileResult<BTreeMap<Name, TargetClass>> {
    env.classification.into_iter()
        .map(|(id, cl)| match env.mapping.get(&id) {
            Some(constraint) => {
                let cl = unify_constraint(cl, &constraint, &id)?;
                Ok((id, cl))
            },
            None => Ok((id, cl))
        })
        .collect::<CompileResult<BTreeMap<Name, TargetClass>>>()
}

pub fn apply(
    ast: &Ast,
    mapping: &BTreeMap<Name, TargetConstraint>
) -> CompileResult<BTreeMap<Name, TargetClass>> {
    let env = ClassifyEnv::new(mapping);
    let env = classify_fun_def(env, &ast.main)?;
    let env = ast.tops.iter().rev()
        .fold(Ok(env), |env, t| {
            classify_top(env?, t)
        })?;
    validate_classification(env)
}
