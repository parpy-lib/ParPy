use super::ast::*;
use crate::utils::name::Name;
use crate::utils::info::*;
use crate::utils::smap::SMapAccum;

use std::collections::BTreeMap;

fn collect_smem_attribute(
    acc: Option<(Name, usize)>,
    attr: &KernelAttribute
) -> Option<(Name, usize)> {
    match attr {
        KernelAttribute::SharedMemory {id, bytes} => Some((id.clone(), *bytes)),
        _ => acc
    }
}

fn collect_smem_usage_top(
    mut acc: BTreeMap<Name, usize>,
    t: &Top
) -> BTreeMap<Name, usize> {
    match t {
        Top::FunDef {attrs, id, ..} => {
            match attrs.iter().fold(None, collect_smem_attribute) {
                Some((_, smem)) => {
                    acc.insert(id.clone(), smem);
                    acc
                },
                None => acc
            }
        },
        _ => acc
    }
}

fn collect_usage(ast: &Ast) -> BTreeMap<Name, usize> {
    ast.iter()
        .fold(BTreeMap::new(), collect_smem_usage_top)
}

fn generate_kernel_smem_config_stmt(args: (&Name, &usize)) -> Stmt {
    let (id, nbytes) = args;
    let value = Expr::Int {
        v: *nbytes as i128,
        ty: Type::Scalar {sz: ElemSize::I64},
        i: Info::default()
    };
    Stmt::CheckError {
        e: Expr::FuncSetAttribute {
            func: id.clone(),
            attr: FuncAttribute::MaxDynamicSharedMemorySize,
            value: Box::new(value),
            ty: Type::Error,
            i: Info::default()
        }
    }
}

fn update_entry_point_top(
    smem_usage: &BTreeMap<Name, usize>,
    t: Top
) -> Top {
    match t {
        Top::FunDef {dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body} => {
            // Ensure that the maximum amount of shared memory usage per block is within the limits
            // of the current device. If this is not the case, we make an early exit from the entry
            // point. When the peak is zero bytes, we skip this part to simplify the generated
            // code.
            let max_smem_usage = smem_usage.values().max().unwrap_or(&0);
            let body = if *max_smem_usage > 0 {
                let temp_id = Name::sym_str("x");
                let check_smem_usage_decl_stmt = Stmt::Definition {
                    ty: Type::Scalar {sz: ElemSize::I32},
                    id: temp_id.clone(),
                    expr: Some(Expr::ValidateSharedMemUsage {
                        nbytes: Box::new(Expr::Int {
                            v: *max_smem_usage as i128,
                            ty: Type::Scalar {sz: ElemSize::I64},
                            i: Info::default()
                        }),
                        ty: Type::Error,
                        i: Info::default()
                    })
                };
                let temp_var = Expr::Var {
                    id: temp_id.clone(),
                    ty: Type::Scalar {sz: ElemSize::I32},
                    i: Info::default()
                };
                let check_smem_usage_cond_fail = Stmt::If {
                    cond: Expr::BinOp {
                        lhs: Box::new(temp_var.clone()),
                        op: BinOp::Neq,
                        rhs: Box::new(Expr::Int {
                            v: 0,
                            ty: Type::Scalar {sz: ElemSize::I32},
                            i: Info::default()
                        }),
                        ty: Type::Scalar {sz: ElemSize::Bool},
                        i: Info::default()
                    },
                    thn: vec![Stmt::Return {
                        value: temp_var
                    }],
                    els: vec![]
                };
                let validate_smem_stmts = vec![
                    check_smem_usage_decl_stmt,
                    check_smem_usage_cond_fail
                ];
                let config_smem_usage_stmts = smem_usage.iter()
                    .map(generate_kernel_smem_config_stmt);
                validate_smem_stmts.into_iter()
                    .chain(config_smem_usage_stmts)
                    .chain(body.into_iter())
                    .collect::<Vec<Stmt>>()
            } else {
                body
            };
            Top::FunDef {dev_attr: Attribute::Entry, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

fn update_entry_point(
    smem_usage: BTreeMap<Name, usize>,
    ast: Ast
) -> Ast {
    ast.smap(|t| update_entry_point_top(&smem_usage, t))
}

fn configure_and_validate_in_entry_point(ast: Ast) -> Ast {
    let smem_usage = collect_usage(&ast);
    update_entry_point(smem_usage, ast)
}

fn use_dynamic_shared_memory_in_kernel(t: Top) -> Top {
    match t {
        Top::FunDef {dev_attr, ret_ty, mut attrs, id, params, body} => {
            let (attrs, body) = match attrs.iter().fold(None, collect_smem_attribute) {
                Some((smem_id, nbytes)) => {
                    let attr = KernelAttribute::SharedMemory {
                        id: smem_id.clone(),
                        bytes: nbytes
                    };
                    let pos = attrs.iter().position(|x| *x == attr).unwrap();
                    attrs.remove(pos);
                    let alloc_stmt = Stmt::AllocShared {
                        ty: Type::Scalar {sz: ElemSize::I8},
                        id: smem_id
                    };
                    let body = vec![alloc_stmt].into_iter()
                        .chain(body.into_iter())
                        .collect::<Vec<Stmt>>();
                    (attrs, body)
                },
                None => (attrs, body)
            };
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

pub fn configure_ast(ast: Ast) -> Ast {
    // 1. Insert code in the entry point to ensure the device has a sufficient amount of shared
    //    memory available to run all kernels, and to configure the amount of shared memory to be
    //    used in each kernel.
    let ast = configure_and_validate_in_entry_point(ast);

    // 2. For each kernel that uses shared memory, declare a variable containing its dynamically
    //    allocated shared memory, and remove its shared memory attribute.
    ast.smap(use_dynamic_shared_memory_in_kernel)
}
