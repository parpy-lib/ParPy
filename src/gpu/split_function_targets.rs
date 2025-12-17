use crate::parpy_internal_error;
use crate::ir::ast::*;
use crate::ir::TargetClass;
use crate::utils::err::*;
use crate::utils::name::Name;
use crate::utils::smap::{SFlatten, SMapAccum};

use std::collections::BTreeMap;

#[derive(Debug)]
struct Names {
    host_id: Name,
    device_id: Name
}

#[derive(Debug)]
struct SplitTargetsEnv {
    classification: BTreeMap<Name, TargetClass>,
    split_targets: BTreeMap<Name, Names>
}

impl SplitTargetsEnv {
    fn new(classification: BTreeMap<Name, TargetClass>) -> SplitTargetsEnv {
        let (classification, split_targets) = classification.into_iter()
            .fold((BTreeMap::new(), BTreeMap::new()), |acc, (id, class)| {
                let (mut classification, mut split_targets) = acc;
                match class {
                    TargetClass::Host | TargetClass::Device => {
                        classification.insert(id.clone(), class);
                    },
                    TargetClass::Both => {
                        let host_id = id.clone().with_new_sym();
                        classification.insert(host_id.clone(), TargetClass::Host);
                        let device_id = id.clone().with_new_sym();
                        classification.insert(device_id.clone(), TargetClass::Device);
                        split_targets.insert(id, Names {host_id, device_id});
                    }
                };
                (classification, split_targets)
            });
        SplitTargetsEnv {classification, split_targets}
    }
}

fn select_call_id(env: &SplitTargetsEnv, target: &Target, id: Name) -> Name {
    match env.split_targets.get(&id) {
        Some(Names {host_id, device_id}) => match target {
            Target::Host => host_id.clone(),
            Target::Device => device_id.clone()
        },
        None => id
    }
}

fn split_functions_targeting_both_expr(
    env: &SplitTargetsEnv,
    target: &Target,
    e: Expr
) -> Expr {
    match e {
        Expr::Call {id, args, par, ty, i} => {
            let id = select_call_id(env, target, id);
            let args = args.smap(|e| {
                split_functions_targeting_both_expr(env, target, e)
            });
            Expr::Call {id, args, par, ty, i}
        },
        _ => {
            e.smap(|e| split_functions_targeting_both_expr(env, target, e))
        }
    }
}

fn split_functions_targeting_both_stmt(
    env: &SplitTargetsEnv,
    target: &Target,
    s: Stmt
) -> Stmt {
    match s {
        Stmt::For {var, lo, hi, step, body, par, i} if par.is_parallel() => {
            let lo = split_functions_targeting_both_expr(env, target, lo);
            let hi = split_functions_targeting_both_expr(env, target, hi);
            let body = body.smap(|s| {
                split_functions_targeting_both_stmt(env, &Target::Device, s)
            });
            Stmt::For {var, lo, hi, step, body, par, i}
        },
        _ => {
            let s = s.smap(|s| {
                split_functions_targeting_both_stmt(env, target, s)
            });
            s.smap(|e| {
                split_functions_targeting_both_expr(env, target, e)
            })
        }
    }
}

fn lookup_target(env: &SplitTargetsEnv, def: &FunDef) -> CompileResult<Target> {
    let id = &def.id;
    match env.classification.get(&id) {
        Some(TargetClass::Host) => Ok(Target::Host),
        Some(TargetClass::Device) => Ok(Target::Device),
        Some(TargetClass::Both) => {
            parpy_internal_error!(def.i, "Found function {id} classified as both after split")
        },
        None => Ok(Target::Host),
    }
}

fn split_functions_targeting_both_def(
    env: &SplitTargetsEnv,
    def: FunDef
) -> CompileResult<(FunDef, Option<FunDef>)> {
    match env.split_targets.get(&def.id) {
        Some(Names {host_id, device_id}) => {
            let host_body = def.body.clone().smap(|s| {
                split_functions_targeting_both_stmt(&env, &Target::Host, s)
            });
            let host_def = FunDef {id: host_id.clone(), body: host_body, ..def.clone()};
            let device_body = def.body.smap(|s| {
                split_functions_targeting_both_stmt(&env, &Target::Device, s)
            });
            let dev_def = FunDef {id: device_id.clone(), body: device_body, ..def};
            Ok((host_def, Some(dev_def)))
        },
        None => {
            let target = lookup_target(env, &def)?;
            let body = def.body.smap(|s| {
                split_functions_targeting_both_stmt(&env, &target, s)
            });
            Ok((FunDef {body, ..def}, None))
        }
    }
}

fn split_functions_targeting_both_top(
    env: &SplitTargetsEnv,
    mut tops: Vec<Top>,
    t: Top
) -> CompileResult<Vec<Top>> {
    match t {
        Top::ExtDecl {..} => tops.push(t),
        Top::FunDef {v} => {
            let (ldef, opt_rdef) = split_functions_targeting_both_def(env, v)?;
            tops.push(Top::FunDef {v: ldef});
            if let Some(rdef) = opt_rdef {
                tops.push(Top::FunDef {v: rdef});
            }
        }
    };
    Ok(tops)
}

fn split_functions_targeting_both(env: &SplitTargetsEnv, ast: Ast) -> CompileResult<Ast> {
    let tops = ast.tops.sflatten_result(vec![], |acc, t| {
        split_functions_targeting_both_top(env, acc, t)
    })?;
    let main = match split_functions_targeting_both_def(env, ast.main)? {
        (main, None) => Ok(main),
        (d, Some(_)) => parpy_internal_error!(d.i, "Cannot split the main function")
    }?;
    Ok(Ast {tops, main})
}

pub fn apply(
    ast: Ast,
    classification: BTreeMap<Name, TargetClass>
) -> CompileResult<(Ast, BTreeMap<Name, TargetClass>)> {
    let env = SplitTargetsEnv::new(classification);
    let ast = split_functions_targeting_both(&env, ast)?;
    Ok((ast, env.classification))
}
