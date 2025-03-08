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
            Type::Struct {id} => {
                let (env, id) = id.pprint(env);
                (env, format!("struct {id}"))
            }
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
            Expr::Convert {e, ty} => {
                let (env, e) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                (env, format!("{ty}({e})"))
            }
        }
    }
}

impl PrettyPrint for LoopParallelism {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let LoopParallelism {nthreads, reduction} = self;
        (env, format!("{{nthreads = {nthreads}, reduction = {reduction}}}"))
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
            Stmt::SyncPoint {block_local, ..} => {
                if *block_local {
                    (env, format!("{indent}__syncthreads();"))
                } else {
                    (env, format!("{indent}sync();"))
                }
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
                    "{0}for (int64_t {1} = {2}; {1} < {3}; {1} += {4}) {{ // {5}\n{6}\n{0}}}",
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
            Stmt::Alloc {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                (env, format!("{indent}{id} = alloc[{elem_ty}]({sz});"))
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

impl PrettyPrint for StructDef {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let StructDef {id, fields, ..} = self;
        let (env, id) = id.pprint(env);
        let env = env.incr_indent();
        let (env, fields) = pprint_iter(fields.iter(), env, "\n");
        let env = env.decr_indent();
        (env, format!("struct {id} {{\n{fields}\n}};"))
    }
}

impl PrettyPrint for FunDef {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let FunDef {id, params, body, ..} = self;
        let (env, id) = id.pprint(env);
        let (env, params) = pprint_iter(params.iter(), env, ", ");
        let env = env.incr_indent();
        let (env, body) = pprint_iter(body.iter(), env, "\n");
        let env = env.decr_indent();
        (env, format!("void {id}({params}) {{\n{body}\n}}"))
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Ast {structs, fun} = self;
        let (env, structs) = pprint_iter(structs.iter(), env, "\n");
        let (env, fun) = fun.pprint(env);
        (env, format!("{structs}\n{fun}"))
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
