use super::ast::*;
use crate::utils::name::Name;

fn collect_threadgroup_attribute(
    acc: Option<(Name, usize)>,
    attr: &FunAttribute
) -> Option<(Name, usize)> {
    match attr {
        FunAttribute::ThreadgroupMemory {id, bytes} => Some((id.clone(), *bytes)),
        _ => acc
    }
}

fn insert_threadgroup_parameter(t: Top) -> Top {
    match t {
        Top::VarDef {ty, id, init: Some(Expr::LoadLibrary {tops, ty: ty_expr, i})} => {
            let tops = tops.into_iter()
                .map(insert_threadgroup_parameter)
                .collect::<Vec<Top>>();
            Top::VarDef {ty, id, init: Some(Expr::LoadLibrary {tops, ty: ty_expr, i})}
        },
        Top::FunDef {mut attrs, is_kernel, ret_ty, id, mut params, body} => {
            println!("{id} {attrs:?}");
            let (attrs, params) = match attrs.iter().fold(None, collect_threadgroup_attribute) {
                Some((mem_id, nbytes)) => {
                    let attr = FunAttribute::ThreadgroupMemory {
                        id: mem_id.clone(),
                        bytes: nbytes
                    };
                    let pos = attrs.iter().position(|x| *x == attr).unwrap();
                    attrs.remove(pos);
                    let param_ty = Type::Pointer {
                        ty: Box::new(Type::Scalar {sz: ElemSize::I8}),
                        mem: MemSpace::Threadgroup
                    };
                    let threadgroup_param = Param {
                        id: mem_id,
                        ty: param_ty,
                        attr: Some(ParamAttribute::Threadgroup {idx: 0})
                    };
                    params.push(threadgroup_param);
                    (attrs, params)
                },
                None => (attrs, params)
            };
            Top::FunDef {attrs, is_kernel, ret_ty, id, params, body}
        },
        _ => t
    }
}

pub fn configure_ast(ast: Ast) -> Ast {
    let tops = ast.tops.into_iter()
        .map(insert_threadgroup_parameter)
        .collect::<Vec<Top>>();
    Ast {tops}
}
