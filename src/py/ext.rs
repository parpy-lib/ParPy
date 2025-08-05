use crate::py::ast::*;
use crate::utils::ast::ElemSize;
use crate::utils::info::*;
use crate::utils::name::Name;

use pyo3::pyclass;

#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq)]
pub enum ExtType {
    Scalar(ElemSize),
    Pointer(ElemSize),
}

fn to_type(ty: ExtType) -> Type {
    match ty {
        ExtType::Scalar(sz) => Type::Tensor {sz, shape: vec![]},
        ExtType::Pointer(sz) => Type::Pointer {sz},
    }
}

fn to_param(id: String, ty: ExtType, i: &Info) -> Param {
    Param { id: Name::sym_str(&id), ty: to_type(ty), i: i.clone() }
}

pub fn make_declaration(
    id: String,
    ext_id: String,
    params: Vec<(String, ExtType)>,
    res_ty: ExtType,
    header: Option<String>,
    info: Option<(String, usize, usize, usize, usize)>
) -> Top {
    let i = if let Some((f, l1, c1, l2, c2)) = info {
        Info::new(&f, FilePos::new(l1, c1), FilePos::new(l2, c2))
    } else {
        Info::default()
    };
    let params = params.into_iter()
        .map(|(id, ty)| to_param(id, ty, &i))
        .collect::<Vec<Param>>();
    let res_ty = to_type(res_ty);
    Top::ExtDecl {id: Name::sym_str(&id), ext_id, params, res_ty, header, i}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::py::ast_builder::*;

    #[test]
    fn to_type_scalar() {
        assert_eq!(to_type(ExtType::Scalar(ElemSize::F32)), scalar(ElemSize::F32));
    }

    #[test]
    fn to_type_pointer() {
        let ptr_ty = Type::Pointer {sz: ElemSize::F32};
        assert_eq!(to_type(ExtType::Pointer(ElemSize::F32)), ptr_ty);
    }

    #[test]
    fn to_param_ext_type() {
        let p = to_param("x".to_string(), ExtType::Scalar(ElemSize::F32), &i());
        assert_eq!(p.id.get_str(), "x");
        assert!(p.id.has_sym());
        assert_eq!(p.ty, scalar(ElemSize::F32));
    }

    #[test]
    fn make_declaration_args() {
        let id = "popcount".to_string();
        let ext_id = "__popcount".to_string();
        let params = vec![("x".to_string(), ExtType::Scalar(ElemSize::I32))];
        let res_ty = ExtType::Scalar(ElemSize::I32);
        let info = ("file.cpp".to_string(), 1, 1, 1, 10);
        let decl = make_declaration(
            id.clone(), ext_id.clone(), params, res_ty, None, Some(info)
        );
        if let Top::ExtDecl {id, ext_id, params, res_ty, header, i} = decl {
            assert_eq!(id.get_str(), "popcount");
            assert!(id.has_sym());
            assert_eq!(ext_id, "__popcount");
            assert_eq!(params.len(), 1);
            assert_eq!(res_ty, scalar(ElemSize::I32));
            assert!(header.is_none());
            assert_eq!(i, Info::new("file.cpp", FilePos::new(1, 1), FilePos::new(1, 10)));
        } else {
            panic!("Invalid form of declaration")
        }
    }
}
