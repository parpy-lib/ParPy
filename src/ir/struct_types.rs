use crate::py::ast::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

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
        Type::Dict {fields, ..} => {
            let types = fields.values().fold(types, |types, ty| {
                find_dict_types_type(types, ty, id)
            });
            types.add_type(ty, id)
        },
        _ => ty.sfold(types, |types, ty| find_dict_types_type(types, ty, id))
    }
}

fn find_dict_types_expr(
    types: DictTypes,
    e: &Expr
) -> DictTypes {
    let types = find_dict_types_type(types, &e.get_type(), None);
    e.sfold(types, find_dict_types_expr)
}

fn find_dict_types_stmt(
    types: DictTypes,
    stmt: &Stmt
) -> DictTypes {
    let types = stmt.sfold(types, find_dict_types_stmt);
    stmt.sfold(types, find_dict_types_expr)
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
    def.body.sfold(types, find_dict_types_stmt)
}

pub fn find_dict_types(def: &FunDef) -> DictTypes {
    find_dict_types_def(DictTypes::default(), def)
}
