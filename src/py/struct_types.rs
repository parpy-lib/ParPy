use super::ast::*;

use itertools::Itertools;

use std::collections::{BTreeMap, BTreeSet};

type StructTypes = BTreeSet<Type>;

fn find_struct_types_type(
    types: StructTypes,
    ty: &Type
) -> StructTypes {
    match ty {
        Type::Tuple {elems, ..} => {
            elems.iter().fold(types, find_struct_types_type)
        },
        Type::Dict {fields, ..} => {
            let mut types = fields.values().fold(types, find_struct_types_type);
            types.insert(ty.clone());
            types
        },
        _ => types
    }
}

fn find_struct_types_expr(
    types: StructTypes,
    e: &Expr
) -> StructTypes {
    let types = find_struct_types_type(types, &e.get_type());
    match e {
        Expr::UnOp {arg, ..} => find_struct_types_expr(types, arg),
        Expr::BinOp {lhs, rhs, ..} => {
            let types = find_struct_types_expr(types, lhs);
            find_struct_types_expr(types, rhs)
        },
        Expr::Subscript {target, idx, ..} => {
            let types = find_struct_types_expr(types, target);
            find_struct_types_expr(types, idx)
        },
        Expr::Tuple {elems, ..} => elems.iter().fold(types, find_struct_types_expr),
        Expr::Dict {fields, ..} => fields.values().fold(types, find_struct_types_expr),
        Expr::Builtin {args, ..} => args.iter().fold(types, find_struct_types_expr),
        Expr::Convert {e, ..} => find_struct_types_expr(types, e),
        _ => types
    }
}

fn find_struct_types_stmt(
    types: StructTypes,
    stmt: &Stmt
) -> StructTypes {
    match stmt {
        Stmt::Assign {dst, expr, ..} => {
            let types = find_struct_types_expr(types, dst);
            find_struct_types_expr(types, expr)
        },
        Stmt::For {lo, hi, body, ..} => {
            let types = find_struct_types_expr(types, lo);
            let types = find_struct_types_expr(types, hi);
            body.iter()
                .fold(types, find_struct_types_stmt)
        },
        Stmt::If {cond, thn, els, ..} => {
            let types = find_struct_types_expr(types, cond);
            let types = thn.iter().fold(types, find_struct_types_stmt);
            els.iter().fold(types, find_struct_types_stmt)
        },
    }
}

fn find_struct_types_def(
    types: StructTypes,
    def: &FunDef
) -> StructTypes {
    let types = def.params
        .iter()
        .fold(types, |types, Param {ty, ..}| find_struct_types_type(types, ty));
    def.body
        .iter()
        .fold(types, find_struct_types_stmt)
}

pub fn find_struct_types(ast: &Ast) -> StructTypes {
    ast.iter()
        .fold(StructTypes::default(), |types, def| {
            find_struct_types_def(types, def)
        })
}
