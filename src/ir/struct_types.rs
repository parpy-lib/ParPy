use crate::py::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use itertools::Itertools;

#[derive(Debug, PartialEq)]
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

pub fn find_dict_types(ast: &Ast) -> DictTypes {
    ast.defs.sfold(DictTypes::default(), find_dict_types_def)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;
    use crate::utils::info::Info;

    #[test]
    fn find_dict_type_empty() {
        let def = FunDef {
            id: id("f"),
            params: vec![],
            body: vec![],
            res_ty: Type::Void,
            i: Info::default()
        };
        let ast = Ast {exts: vec![], defs: vec![def]};
        assert_eq!(find_dict_types(&ast), DictTypes::default())
    }

    #[test]
    fn find_dict_type_param() {
        let x = "x".to_string();
        let dict_ty = Type::Dict {
            fields: vec![
                (x.clone(), Type::Tensor {sz: ElemSize::I32, shape: vec![]})
            ].into_iter().collect::<BTreeMap<String, Type>>()
        };
        let def = FunDef {
            id: id("f"),
            params: vec![Param {id: id("y"), ty: dict_ty.clone(), i: Info::default()}],
            body: vec![],
            res_ty: Type::Void,
            i: Info::default()
        };
        let ast = Ast {exts: vec![], defs: vec![def]};
        let dt = find_dict_types(&ast);
        let expected_hints = vec![
            (dict_ty.clone(), "y".to_string())
        ].into_iter().collect::<BTreeMap<Type, String>>();
        assert_eq!(dt.name_hints, expected_hints);
        let expected_types = vec![dict_ty].into_iter().collect::<BTreeSet<Type>>();
        assert_eq!(dt.types, expected_types);
    }

    #[test]
    fn find_dict_type_stmt() {
        let dty = dict_ty(vec![("z", scalar(ElemSize::F32))]);
        let s = assignment(
            var("x", scalar(ElemSize::F32)),
            subscript(var("y", dty.clone()), string("z"), scalar(ElemSize::F32))
        );
        let dt = find_dict_types_stmt(DictTypes::default(), &s);
        assert_eq!(dt.name_hints, BTreeMap::new());
        let types = vec![dty].into_iter().collect::<BTreeSet<Type>>();
        assert_eq!(dt.types, types);
    }
}
