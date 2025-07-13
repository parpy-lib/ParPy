use crate::prickle_compile_error;
use crate::gpu::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::pprint::PrettyPrint;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

type StructEnv = BTreeMap::<Name, BTreeMap<String, Param>>;
type HostStructEnv = BTreeMap::<Name, Vec<(String, Expr)>>;

fn validate_return_type(ty: &Type) -> CompileResult<()> {
    match ty {
        Type::Void | Type::Boolean | Type::Scalar {..} => Ok(()),
        Type::Pointer {ty, ..} => validate_return_type(&ty),
        Type::Struct {id} => {
            let i = Info::default();
            let id = id.pprint_default();
            prickle_compile_error!(i, "Found struct return type in function {id}, \
                                     which is not supported")
        },
    }
}

fn contains_struct_type(ty: &Type) -> bool {
    match ty {
        Type::Void | Type::Boolean | Type::Scalar {..} => false,
        Type::Pointer {ty, ..} => contains_struct_type(ty),
        Type::Struct {..} => true,
    }
}

fn find_struct_fields<'a>(
    env: &'a StructEnv, ty: &Type, i: Info
) -> CompileResult<&'a BTreeMap<String, Param>> {
    match ty {
        Type::Struct {id} => match env.get(id) {
            Some(fields) => Ok(fields),
            None => {
                let idstr = id.pprint_default();
                prickle_compile_error!(i, "Could not find struct type with name {0}", idstr)
            }
        },
        _ => {
            prickle_compile_error!(i, "Found struct field access on non-variable value")
        }
    }
}

fn flatten_structs_kernel_expr(env: &StructEnv, e: Expr) -> CompileResult<Expr> {
    match e {
        Expr::StructFieldAccess {target, label, ty, i} => {
            let fields = find_struct_fields(env, target.get_type(), i.clone())?;
            match fields.get(&label) {
                Some(Param {id, ..}) => Ok(Expr::Var {id: id.clone(), ty, i}),
                None => {
                    prickle_compile_error!(i, "Found reference to unknown struct \
                                               field {label}")
                }
            }
        },
        _ => e.smap_result(|e| flatten_structs_kernel_expr(env, e))
    }
}

fn flatten_structs_kernel_stmt(env: &StructEnv, s: Stmt) -> CompileResult<Stmt> {
    s.smap_result(|e| flatten_structs_kernel_expr(env, e))?
        .smap_result(|s| flatten_structs_kernel_stmt(env, s))
}

fn flatten_structs_kernel_body(env: &StructEnv, body: Vec<Stmt>) -> CompileResult<Vec<Stmt>> {
    body.smap_result(|s| flatten_structs_kernel_stmt(env, s))
}

fn expand_kernel_launch_argument(env: &HostStructEnv, arg: Expr) -> Vec<Expr> {
    match arg {
        Expr::Var {ref id, ..} => match env.get(&id) {
            Some(fields) => {
                fields.clone()
                   .into_iter()
                   .map(|(_, v)| v)
                   .collect::<Vec<Expr>>()
            },
            None => vec![arg]
        },
        _ => vec![arg]
    }
}

fn flatten_structs_host_stmt(
    mut env: HostStructEnv, s: Stmt
) -> (HostStructEnv, Option<Stmt>) {
    match s {
        Stmt::Definition {id, expr: Expr::Struct {fields, ..}, ..} => {
            env.insert(id, fields);
            (env, None)
        },
        Stmt::KernelLaunch {id, args, grid, i} => {
            let args = args.into_iter()
                .flat_map(|a| expand_kernel_launch_argument(&env, a))
                .collect::<Vec<Expr>>();
            (env, Some(Stmt::KernelLaunch {id, args, grid, i}))
        },
        _ => (env, Some(s))
    }
}

fn flatten_structs_host_body(body: Vec<Stmt>) -> Vec<Stmt> {
    let (_, body) = body.into_iter()
        .fold((BTreeMap::new(), vec![]), |acc: (HostStructEnv, Vec<Stmt>), s| {
            let (env, mut stmts) = acc;
            let (env, o) = flatten_structs_host_stmt(env, s);
            if let Some(s) = o {
                stmts.push(s);
            }
            (env, stmts)
        });
    body
}

fn expand_kernel_param(env: &StructEnv, p: Param) -> CompileResult<Vec<Param>> {
    match &p.ty {
        Type::Struct {id} => {
            match env.get(id) {
                Some(field_params) => {
                    Ok(field_params.clone()
                        .into_iter()
                        .map(|(_, p)| p)
                        .collect::<Vec<Param>>())
                },
                None => {
                    let pid = p.id.pprint_default();
                    prickle_compile_error!(p.i, "Parameter {pid} refers to undefined struct type {id}")
                }
            }
        },
        Type::Pointer {ty, ..} if contains_struct_type(ty) => {
            let pid = p.id.pprint_default();
            prickle_compile_error!(p.i, "Parameter {pid} contains pointer to a \
                                       struct type, which is not supported")
        },
        _ => Ok(vec![p])
    }
}

fn expand_kernel_params(
    env: StructEnv, params: Vec<Param>
) -> CompileResult<(StructEnv, Vec<Param>)> {
    params.into_iter()
        .fold(Ok((env, vec![])), |acc: CompileResult<(StructEnv, Vec<Param>)>, p| {
            let (env, mut params) = acc?;
            let mut p = expand_kernel_param(&env, p)?;
            params.append(&mut p);
            Ok((env, params))
        })
}

fn flatten_structs_top(
    mut env: StructEnv, t: Top
) -> CompileResult<(StructEnv, Option<Top>)> {
    match t {
        Top::KernelFunDef {threads, id, params, body} => {
            let (env, params) = expand_kernel_params(env, params)?;
            let body = flatten_structs_kernel_body(&env, body)?;
            Ok((env, Some(Top::KernelFunDef {threads, id, params, body})))
        },
        Top::FunDef {ret_ty, id, params, body, target} => {
            validate_return_type(&ret_ty)?;
            // NOTE: Currently, we only support simple scalar types as arguments to user-defined
            // functions (these will have a device target). Therefore, we do not need to flatten
            // structs within their bodies.
            let body = if target == Target::Host {
                flatten_structs_host_body(body)
            } else {
                body
            };
            Ok((env, Some(Top::FunDef {ret_ty, id, params, body, target})))
        },
        Top::StructDef {id, fields} => {
            let renamed_fields = fields.into_iter()
                .map(|Field {id: fid, ty, i}| {
                    let pid = Name::new(format!("{0}_{1}", id.get_str(), fid));
                    (fid, Param {id: pid, ty, i})
                })
                .collect::<BTreeMap<String, Param>>();
            env.insert(id, renamed_fields);
            Ok((env, None))
        }
    }
}

pub fn flatten_structs(ast: Ast) -> CompileResult<Ast> {
    let (_, ast) = ast.into_iter()
        .fold(Ok((BTreeMap::new(), vec![])), |acc: CompileResult<(StructEnv, Vec<Top>)>, t| {
            let (env, mut tops) = acc?;
            let (env, t) = flatten_structs_top(env, t)?;
            match t {
                Some(v) => tops.push(v),
                None => ()
            };
            Ok((env, tops))
        })?;
    Ok(ast)
}
