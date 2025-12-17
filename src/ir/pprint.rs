use super::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;

use std::fmt;

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Tensor {sz, shape} if shape.is_empty() => sz.pprint(env),
            Type::Tensor {sz, shape} => {
                let (env, sz) = sz.pprint(env);
                let shape_str = shape.iter()
                    .map(|s| s.to_string())
                    .join(", ");
                (env, format!("tensor<{sz};{shape_str}>"))
            },
            Type::Pointer {ty} => {
                let (env, ty) = ty.pprint(env);
                (env, format!("ptr<{ty}>"))
            },
            Type::Struct {id} => {
                let (env, id) = id.pprint(env);
                (env, format!("struct {id}"))
            },
            Type::Void => (env, format!("void"))
        }
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::Bool {v, ..} => (env, v.to_string()),
            Expr::Int {v, ..} => (env, v.to_string()),
            Expr::Float {v, ..} => (env, v.to_string()),
            Expr::UnOp {op, arg, ..} => {
                let (env, op) = op.pprint(env);
                let (env, arg) = arg.pprint(env);
                (env, format!("({op} {arg})"))
            },
            Expr::BinOp {lhs, op, rhs, ..} => {
                let (env, lhs) = lhs.pprint(env);
                let (env, op) = op.pprint(env);
                let (env, rhs) = rhs.pprint(env);
                (env, format!("({lhs} {op} {rhs})"))
            },
            Expr::IfExpr {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("({thn} if {cond} else {els})"))
            },
            Expr::StructFieldAccess {target, label, ..} => {
                let (env, target) = target.pprint(env);
                (env, format!("{target}.{label}"))
            },
            Expr::TensorAccess {target, idx, ..} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::Call {id, args, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::PyCallback {id, args, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::Convert {e, ty, ..} => {
                let (env, e) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                (env, format!("{ty}({e})"))
            }
        }
    }
}

impl PrettyPrint for SyncPointKind {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            SyncPointKind::BlockLocal => format!("block_local"),
            SyncPointKind::BlockCluster => format!("block_cluster"),
            SyncPointKind::InterBlock => format!("inter_block"),
        };
        (env, s)
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Definition {ty, id, expr, ..} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{ty} {id} = {expr};"))
            },
            Stmt::Assign {dst, expr, ..} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{dst} = {expr};"))
            },
            Stmt::SyncPoint {kind, ..} => {
                let (env, kind) = kind.pprint(env);
                (env, format!("{indent}sync({kind});"))
            },
            Stmt::For {var, lo, hi, step, body, par, ..} => {
                let (env, var) = var.pprint(env);
                let (env, lo) = lo.pprint(env);
                let (env, hi) = hi.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let (env, par) = par.pprint(env);
                let s = format!(
                    "{0}for (int64_t {1} = {2}..{3}..{4}) {{ // {5}\n{6}\n{0}}}",
                    indent, var, lo, hi, step, par, body
                );
                (env, s)
            },
            Stmt::If {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, thn) = pprint_iter(thn.iter(), env, "\n");
                let (env, els) = pprint_iter(els.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}if ({cond}) {{\n{thn}\n{0}}} else {{\n{els}\n{0}}}", indent))
            },
            Stmt::While {cond, body, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}while ({cond}) {{\n{body}\n{0}}}", indent))
            },
            Stmt::Expr {e, ..} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}{e};"))
            },
            Stmt::Return {value, ..} => {
                let (env, value) = value.pprint(env);
                (env, format!("{indent}return {value};"))
            },
            Stmt::Alloc {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                (env, format!("{indent}{id} = alloc[{elem_ty}]({sz});"))
            },
            Stmt::AllocShared {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                (env, format!("{indent}{id} = alloc_shared[{elem_ty}]({sz});"))
            },
            Stmt::Free {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("{indent}free({id});"))
            }
        }
    }
}

impl PrettyPrint for Field {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Field {id, ty, ..} = self;
        let (env, ty) = ty.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{ty} {id};"))
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty, ..} = self;
        let (env, id) = id.pprint(env);
        let (env, ty) = ty.pprint(env);
        (env, format!("{ty} {id}"))
    }
}

impl PrettyPrint for FunDef {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let FunDef {id, params, body, res_ty, ..} = self;
        let (env, id) = id.pprint(env);
        let (env, params) = pprint_iter(params.iter(), env, ", ");
        let env = env.incr_indent();
        let (env, body) = pprint_iter(body.iter(), env, "\n");
        let env = env.decr_indent();
        let (env, res_ty) = res_ty.pprint(env);
        (env, format!("{res_ty} {id}({params}) {{\n{body}\n}}"))
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::StructDef {id, fields, ..} => {
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let (env, fields) = pprint_iter(fields.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("struct {id} {{\n{fields}\n}};"))
            },
            Top::ExtDecl {id, ext_id, params, res_ty, header, target, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let (env, res_ty) = res_ty.pprint(env);
                let header_str = if let Some(h) = header {
                    format!(" [{h}]")
                } else {
                    "".to_string()
                };
                let (env, target) = target.pprint(env);
                (env, format!("{res_ty} {id}({params}) = {ext_id};{header_str} {target}"))
            },
            Top::FunDef {v} => v.pprint(env),
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Ast {tops, main} = self;
        let (env, tops) = pprint_iter(tops.iter(), env, "\n");
        let (env, main) = main.pprint(env);
        (env, format!("{tops}\n{main}"))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}

impl fmt::Display for Ast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ast_builder::*;
    use crate::utils::info::Info;

    #[test]
    fn print_scalar_type() {
        assert_eq!(scalar(ElemSize::F32).pprint_default(), "float");
    }

    #[test]
    fn print_tensor_vector_type() {
        assert_eq!(shape(vec![10]).pprint_default(), "tensor<int64_t;10>");
    }

    #[test]
    fn print_pointer_type() {
        let ptrty = Type::Pointer {ty: Box::new(scalar(ElemSize::F16))};
        assert_eq!(ptrty.pprint_default(), "ptr<half>");
    }

    #[test]
    fn print_struct_type() {
        let sty = Type::Struct {id: id("x")};
        assert_eq!(sty.pprint_default(), "struct x");
    }

    #[test]
    fn print_void_type() {
        assert_eq!(Type::Void.pprint_default(), "void");
    }

    #[test]
    fn print_if_expr() {
        let ifexpr = Expr::IfExpr {
            cond: Box::new(bool_expr(true)),
            thn: Box::new(int(1, None)),
            els: Box::new(int(0, None)),
            ty: scalar(ElemSize::I64),
            i: Info::default()
        };
        assert_eq!(ifexpr.pprint_default(), "(1 if true else 0)");
    }

    #[test]
    fn print_assign_stmt() {
        let ty = scalar(ElemSize::I64);
        let s = assign(var("x", ty.clone()), int(1, None));
        assert_eq!(s.pprint_default(), "x = 1;");
    }

    #[test]
    fn print_assign_indent_stmt() {
        let ty = scalar(ElemSize::I64);
        let s = assign(var("x", ty.clone()), int(1, None));
        let env = PrettyPrintEnv::new().incr_indent();
        let (env, s) = s.pprint(env);
        assert_eq!(s, format!("{}x = 1;", env.print_indent()));
    }

    #[test]
    fn print_block_local_sync_point_stmt() {
        let sp = Stmt::SyncPoint {kind: SyncPointKind::BlockLocal, i: Info::default()};
        assert_eq!(sp.pprint_default(), "sync(block_local);");
    }

    #[test]
    fn print_inter_block_sync_point_stmt() {
        let sp = Stmt::SyncPoint {kind: SyncPointKind::InterBlock, i: Info::default()};
        assert_eq!(sp.pprint_default(), "sync(inter_block);");
    }

    #[test]
    fn print_struct_def() {
        let fields = vec![
            Field {id: "x".to_string(), ty: scalar(ElemSize::F64), i: Info::default()},
            Field {id: "y".to_string(), ty: scalar(ElemSize::F32), i: Info::default()},
        ];
        let def = Top::StructDef {id: id("point"), fields, i: Info::default()};
        let indent = PrettyPrintEnv::new().incr_indent().print_indent();
        let s = format!("struct point {{\n{0}double x;\n{0}float y;\n}};", indent);
        assert_eq!(def.pprint_default(), s);
    }

    #[test]
    fn print_fun_def() {
        let params = vec![
            Param {id: id("x"), ty: scalar(ElemSize::F64), i: Info::default()},
            Param {id: id("y"), ty: scalar(ElemSize::F32), i: Info::default()},
        ];
        let body = vec![
            assign(var("z", scalar(ElemSize::F64)), var("x", scalar(ElemSize::F64))),
        ];
        let def = FunDef {
            id: id("f"), params, body, res_ty: Type::Void, i: Info::default()
        };
        let indent = PrettyPrintEnv::new().incr_indent().print_indent();
        let s = format!("void f(double x, float y) {{\n{}z = x;\n}}", indent);
        assert_eq!(def.pprint_default(), s);
    }
}
