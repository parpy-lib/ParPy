use super::ast::*;
use crate::option;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::*;

use std::collections::BTreeSet;

const CUDA_STANDARD_MAX_BLOCKS_PER_CLUSTER: i64 = 8;

fn find_cluster_dim_attribute_x(attrs: &Vec<KernelAttribute>) -> Option<i64> {
    for attr in attrs {
        if let KernelAttribute::ClusterDims {dims: Dim3 {x, ..}} = attr {
            return Some(*x);
        }
    }
    None
}

fn collect_nonstandard_cluster_kernel_name_top(mut acc: BTreeSet<Name>, t: &Top) -> BTreeSet<Name> {
    match t {
        Top::FunDef {attrs, id, ..} => {
            if let Some(nblocks) = find_cluster_dim_attribute_x(attrs) {
                if nblocks > CUDA_STANDARD_MAX_BLOCKS_PER_CLUSTER {
                    acc.insert(id.clone());
                }
                acc
            } else {
                acc
            }
        },
        _ => acc
    }
}

fn collect_names_of_kernels_using_nonstandard_clusters(ast: &Ast) -> BTreeSet<Name> {
    ast.iter().fold(BTreeSet::new(), collect_nonstandard_cluster_kernel_name_top)
}

fn collect_called_kernels_stmt(
    mut acc: BTreeSet<Name>,
    stmt: &Stmt,
    kernels: &BTreeSet<Name>
) -> BTreeSet<Name> {
    match stmt {
        Stmt::KernelLaunch {id, ..} if kernels.contains(id) => {
            acc.insert(id.clone());
            acc
        },
        _ => stmt.sfold(acc, |acc, s| collect_called_kernels_stmt(acc, s, kernels))
    }
}

fn collect_called_kernels_in_body(stmts: &Vec<Stmt>, kernels: &BTreeSet<Name>) -> BTreeSet<Name> {
    stmts.sfold(BTreeSet::new(), |acc, stmt| collect_called_kernels_stmt(acc, stmt, kernels))
}

fn insert_nonstandard_attribute_config_for_kernels(
    body: Vec<Stmt>,
    used_kernels: BTreeSet<Name>
) -> Vec<Stmt> {
    used_kernels.into_iter()
        .map(|id| {
            let err_id = Name::sym_str("err");
            Stmt::Definition {
                ty: Type::Error,
                id: err_id.clone(),
                expr: Some(Expr::FuncSetAttribute {
                    func: id,
                    attr: FuncAttribute::NonPortableClusterSizeAllowed,
                    value: Box::new(Expr::Int {
                        v: 1, ty: Type::Scalar {sz: ElemSize::I64}, i: Info::default()
                    }),
                    ty: Type::Error,
                    i: Info::default()
                }),
            }
        })
        .chain(body.into_iter())
        .collect::<Vec<Stmt>>()
}

fn add_nonstandard_cluster_attribute_to_kernels_top(t: Top, kernels: &BTreeSet<Name>) -> Top {
    match t {
        Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
            let used_kernels = collect_called_kernels_in_body(&body, kernels);
            let body = insert_nonstandard_attribute_config_for_kernels(body, used_kernels);
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body}
        },
        _ => t
    }
}

fn add_nonstandard_cluster_attribute_to_kernels(ast: Ast, kernels: &BTreeSet<Name>) -> Ast {
    ast.smap(|t| add_nonstandard_cluster_attribute_to_kernels_top(t, kernels))
}

pub fn insert_attribute_for_nonstandard_blocks_per_cluster(
    ast: Ast,
    opts: &option::CompileOptions
) -> Ast {
    // CUDA considers the use of more than eight thread blocks per cluster as non-standard. If we
    // have a GPU supporting this, we can set an attribute to enable using more than eight per
    // cluster.
    if opts.use_cuda_thread_block_clusters &&
       opts.max_thread_blocks_per_cluster > CUDA_STANDARD_MAX_BLOCKS_PER_CLUSTER {
        let cluster_kernel_names = collect_names_of_kernels_using_nonstandard_clusters(&ast);
        add_nonstandard_cluster_attribute_to_kernels(ast, &cluster_kernel_names)
    } else {
        ast
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cuda::ast_builder::*;

    fn mk_set<T: Ord>(v: Vec<T>) -> BTreeSet<T> {
        v.into_iter().collect::<BTreeSet<T>>()
    }

    #[test]
    fn find_cluster_dim_empty_attrs() {
        assert_eq!(find_cluster_dim_attribute_x(&vec![]), None);
    }

    #[test]
    fn find_cluster_dim_no_cluster_attrs() {
        let attrs = vec![KernelAttribute::LaunchBounds {threads: 128}];
        assert_eq!(find_cluster_dim_attribute_x(&attrs), None);
    }

    #[test]
    fn find_cluster_dim_cluster_attr() {
        let attrs = vec![KernelAttribute::ClusterDims {dims: Dim3::default()}];
        let x = Dim3::default().x;
        assert_eq!(find_cluster_dim_attribute_x(&attrs), Some(x));
    }

    #[test]
    fn collect_regular_cluster_kernel_def() {
        let t = Top::FunDef {
            dev_attr: Attribute::Device, ret_ty: Type::Void,
            attrs: vec![KernelAttribute::ClusterDims {dims: Dim3::default()}],
            id: id("f"), params: vec![], body: vec![]
        };
        let res = collect_nonstandard_cluster_kernel_name_top(mk_set(vec![]), &t);
        assert_eq!(res, mk_set(vec![]));
    }

    #[test]
    fn collect_nonstandard_cluster_kernel_def() {
        let max_standard = CUDA_STANDARD_MAX_BLOCKS_PER_CLUSTER;
        let dims = Dim3::default().with_dim(&Dim::X, 2 * max_standard);
        let t = Top::FunDef {
            dev_attr: Attribute::Device, ret_ty: Type::Void,
            attrs: vec![KernelAttribute::ClusterDims {dims}],
            id: id("f"), params: vec![], body: vec![]
        };
        let res = collect_nonstandard_cluster_kernel_name_top(mk_set(vec![]), &t);
        assert_eq!(res, mk_set(vec![id("f")]));
    }

    fn mk_kernel_launch(id: Name) -> Stmt {
        Stmt::KernelLaunch {
            id, blocks: Dim3::default(), threads: Dim3::default(),
            stream: Stream::Default, args: vec![]
        }
    }

    #[test]
    fn collect_kernel_call_ids() {
        let stmts = vec![
            mk_kernel_launch(id("x")),
            mk_kernel_launch(id("y")),
            defn(Type::Error, id("z"), None),
        ];
        let kernels = mk_set(vec![id("y"), id("z")]);
        assert_eq!(collect_called_kernels_in_body(&stmts, &kernels), mk_set(vec![id("y")]));
    }
}
