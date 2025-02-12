use crate::py::ast::*;
use crate::utils::name::Name;

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use itertools::Itertools;

pub struct DictTypes {
    types: BTreeSet<Type>,
    name_hints: BTreeMap<Type, String>
}

impl Default for DictTypes {
    fn default() -> Self {
        DictTypes {types: BTreeSet::new(), name_hints: BTreeMap::new()}
    }
}

impl DictTypes {
    pub fn add_type(mut self, ty: &Type, opt_id: Option<&String>) -> DictTypes {
        self.types.insert(ty.clone());
        if let Some(id) = opt_id {
            self.name_hints.insert(ty.clone(), id.clone());
        }
        self
    }

    pub fn to_named_structs(self) -> BTreeMap<Type, Name> {
        self.types.into_iter()
            .map(|ty| {
                let id = match self.name_hints.get(&ty) {
                    Some(id) => Name::sym_str(&format!("dict_{id}")),
                    None => {
                        let fields = ty.get_dict_type_fields();
                        let field_ids = fields.keys().join("_");
                        Name::sym_str(&format!("dict_{field_ids}"))
                    }
                };
                (ty, id)
            })
            .collect::<BTreeMap<Type, Name>>()
    }
}

fn find_dict_types_type(
    types: DictTypes,
    ty: &Type,
    id: Option<&String>
) -> DictTypes {
    match ty {
        Type::Tuple {elems, ..} => {
            elems.iter().fold(types, |types, ty| find_dict_types_type(types, ty, id))
        },
        Type::Dict {fields, ..} => {
            let types = fields.values().fold(types, |types, ty| {
                find_dict_types_type(types, ty, id)
            });
            types.add_type(ty, id)
        },
        _ => types
    }
}

fn find_dict_types_expr(
    types: DictTypes,
    e: &Expr
) -> DictTypes {
    let types = find_dict_types_type(types, &e.get_type(), None);
    match e {
        Expr::UnOp {arg, ..} => find_dict_types_expr(types, arg),
        Expr::BinOp {lhs, rhs, ..} => {
            let types = find_dict_types_expr(types, lhs);
            find_dict_types_expr(types, rhs)
        },
        Expr::Subscript {target, idx, ..} => {
            let types = find_dict_types_expr(types, target);
            find_dict_types_expr(types, idx)
        },
        Expr::Tuple {elems, ..} => elems.iter().fold(types, find_dict_types_expr),
        Expr::Dict {fields, ..} => fields.values().fold(types, find_dict_types_expr),
        Expr::Builtin {args, ..} => args.iter().fold(types, find_dict_types_expr),
        Expr::Convert {e, ..} => find_dict_types_expr(types, e),
        _ => types
    }
}

fn find_dict_types_stmt(
    types: DictTypes,
    stmt: &Stmt
) -> DictTypes {
    match stmt {
        Stmt::Definition {expr, ..} => find_dict_types_expr(types, expr),
        Stmt::Assign {dst, expr, ..} => {
            let types = find_dict_types_expr(types, dst);
            find_dict_types_expr(types, expr)
        },
        Stmt::For {lo, hi, body, ..} => {
            let types = find_dict_types_expr(types, lo);
            let types = find_dict_types_expr(types, hi);
            body.iter().fold(types, find_dict_types_stmt)
        },
        Stmt::If {cond, thn, els, ..} => {
            let types = find_dict_types_expr(types, cond);
            let types = thn.iter().fold(types, find_dict_types_stmt);
            els.iter().fold(types, find_dict_types_stmt)
        },
        Stmt::While {cond, body, ..} => {
            let types = find_dict_types_expr(types, cond);
            body.iter().fold(types, find_dict_types_stmt)
        },
        Stmt::Label {..} => types
    }
}

fn find_dict_types_def(
    types: DictTypes,
    def: &FunDef
) -> DictTypes {
    let types = def.params
        .iter()
        .fold(types, |types, Param {id, ty, ..}| {
            find_dict_types_type(types, ty, Some(id.get_str()))
        });
    def.body
        .iter()
        .fold(types, find_dict_types_stmt)
}

pub fn find_dict_types(def: &FunDef) -> DictTypes {
    find_dict_types_def(DictTypes::default(), def)
}
