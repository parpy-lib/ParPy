pub mod ast;
mod constant_fold;
mod from_py_ast;
mod pprint;
mod struct_types;
mod tpb;

use ast::*;
use crate::par::LoopPar;
use crate::par::REDUCE_PAR_LABEL;
use crate::py::ast as py_ast;
use crate::utils::debug::*;
use crate::utils::err::*;

use std::collections::BTreeMap;

pub fn from_python(
    def: py_ast::FunDef,
    mut par: BTreeMap<String, LoopPar>,
    debug_env: &DebugEnv
) -> CompileResult<Ast> {
    // Insert the special label associated with a reduction into the parallelization mapping. This
    // is used in slicing involving reduction operations.
    par.insert(REDUCE_PAR_LABEL.to_string(), LoopPar::default().reduce());

    let structs = struct_types::find_dict_types(&def).to_named_structs();
    let env = from_py_ast::IREnv::new(structs.clone(), par);
    let structs = structs.into_iter()
        .map(|(ty, id)| from_py_ast::to_struct_def(&env, id, ty))
        .collect::<CompileResult<Vec<StructDef>>>()?;
    let fun = from_py_ast::to_ir_def(&env, def)?;
    let ast = Ast {structs, fun};
    debug_env.print("Initial IR AST", &ast);
    let ast = tpb::propagate_configuration(ast)?;
    let ast = constant_fold::fold(ast);
    Ok(ast)
}

#[cfg(test)]
pub mod ir_builder {
    use crate::ir::ast::*;
    use crate::utils::info::*;
    use crate::utils::name::Name;

    pub fn id(x: &str) -> Name {
        Name::new(x.to_string())
    }

    pub fn scalar_ty(sz: ElemSize) -> Type {
        Type::Tensor {sz, shape: vec![]}
    }

    pub fn bool_ty() -> Type {
        scalar_ty(ElemSize::Bool)
    }

    pub fn var(v: &str) -> Expr {
        let id = Name::new(v.to_string());
        Expr::Var {id, ty: scalar_ty(ElemSize::Bool), i: Info::default()}
    }

    pub fn int_with_ty(v: i64, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(scalar_ty(ElemSize::I64));
        Expr::Int {v, ty, i: Info::default()}
    }

    pub fn int(v: i64) -> Expr {
        int_with_ty(v, None)
    }

    pub fn float_with_ty(v: f64, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(scalar_ty(ElemSize::F64));
        Expr::Float {v, ty, i: Info::default()}
    }

    pub fn float(v: f64) -> Expr {
        float_with_ty(v, None)
    }

    pub fn bool_expr(v: bool) -> Expr {
        Expr::Bool {v, ty: scalar_ty(ElemSize::Bool), i: Info::default()}
    }

    pub fn unop(op: UnOp, arg: Expr) -> Expr {
        let ty = arg.get_type().clone();
        let i = arg.get_info();
        Expr::UnOp {op, arg: Box::new(arg), ty, i}
    }

    pub fn binop(lhs: Expr, op: BinOp, rhs: Expr, res_ty: Option<Type>) -> Expr {
        let ty = match res_ty {
            Some(ty) => ty,
            None => lhs.get_type().clone()
        };
        let i = lhs.get_info();
        Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
    }

    pub fn assign(lhs: Expr, rhs: Expr) -> Stmt {
        Stmt::Assign {dst: lhs, expr: rhs, i: Info::default()}
    }

    pub fn loop_par(n: i64) -> LoopPar {
        LoopPar::default().threads(n).unwrap()
    }

    fn for_loop_complete(
        var: Name,
        lo: Expr,
        hi: Expr,
        step: i64,
        par: LoopPar,
        body: Vec<Stmt>
    ) -> Stmt {
        Stmt::For {var, lo, hi, step, body, par, i: Info::default()}
    }

    pub fn for_loop(var: Name, n: i64, body: Vec<Stmt>) -> Stmt {
        let par = loop_par(n);
        for_loop_complete(var, int(0), int(10), 1, par, body)
    }

    fn if_cond_complete(cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        Stmt::If {cond, thn, els, i: Info::default()}
    }

    pub fn if_cond(thn: Vec<Stmt>, els: Vec<Stmt>) -> Stmt {
        if_cond_complete(bool_expr(true), thn, els)
    }

    fn while_loop_complete(cond: Expr, body: Vec<Stmt>) -> Stmt {
        Stmt::While {cond, body, i: Info::default()}
    }

    pub fn while_loop(body: Vec<Stmt>) -> Stmt {
        while_loop_complete(bool_expr(true), body)
    }

    pub fn sync_point(block_local: bool) -> Stmt {
        Stmt::SyncPoint {block_local, i: Info::default()}
    }

    pub fn fun_def(body: Vec<Stmt>) -> FunDef {
        FunDef {
            id: Name::new("main".to_string()),
            params: vec![],
            body,
            i: Info::default()
        }
    }
}
