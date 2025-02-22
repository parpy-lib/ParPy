use super::ast::*;
use crate::utils::pprint::*;

use std::fmt;

impl PrettyPrint for Builtin {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Builtin::Exp => "exp",
            Builtin::Inf => "inf",
            Builtin::Log => "log",
            Builtin::Max => "max",
            Builtin::Min => "min",
            Builtin::Abs => "abs",
            Builtin::Cos => "cos",
            Builtin::Sin => "sin",
            Builtin::Sqrt => "sqrt",
            Builtin::Tanh => "tanh",
            Builtin::Atan2 => "atan2",
            Builtin::Sum => "sum",
            Builtin::Convert {..} => "<convert>",
            Builtin::Label => "<label>",
            Builtin::GpuContext => "<gpu_context>",
            Builtin::Ext {id} => &format!("{id}")
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for UnOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            UnOp::Sub => "-",
            UnOp::Not => "!",
            UnOp::BitNeg => "~",
            UnOp::Exp => "exp",
            UnOp::Log => "log",
            UnOp::Cos => "cos",
            UnOp::Sin => "sin",
            UnOp::Sqrt => "sqrt",
            UnOp::Tanh => "tanh",
            UnOp::Abs => "abs"
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for BinOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::FloorDiv => "//",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Pow => "**",
            BinOp::And => "&&",
            BinOp::Or => "||",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::BitShl => "<<",
            BinOp::BitShr => ">>",
            BinOp::Eq => "==",
            BinOp::Neq => "!=",
            BinOp::Leq => "<=",
            BinOp::Geq => ">=",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Max => "max",
            BinOp::Min => "min",
            BinOp::Atan2 => "atan2"
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::String {v, ..} => (env, format!("{v}")),
            Expr::Bool {v, ..} => (env, format!("{v}")),
            Expr::Int {v, ..} => (env, format!("{v}")),
            Expr::Float {v, ..} => (env, format!("{v}")),
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
                (env, format!("{thn} if {cond} else {els}"))
            },
            Expr::Subscript {target, idx, ..} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::Slice {lo, hi, ..} => {
                let pprint_opt = |env, o: &Option<Box<Expr>>| match o {
                    Some(e) => e.pprint(env),
                    None => (env, String::new())
                };
                let (env, lo) = pprint_opt(env, lo);
                let (env, hi) = pprint_opt(env, hi);
                (env, format!("{lo}:{hi}"))
            },
            Expr::Tuple {elems, ..} => {
                let (env, elems) = pprint_iter(elems.iter(), env, ", ");
                (env, format!("({elems})"))
            },
            Expr::Builtin {func, args, ..} => {
                let (env, func) = func.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{func}({args})"))
            },
            Expr::Convert {e, ..} => {
                (env, format!("{e}"))
            }
        }
    }

}
impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Definition {id, expr, ..} => {
                let (env, id) = id.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{id} = {expr}"))
            },
            Stmt::Assign {dst, expr, ..} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{dst} = {expr}"))
            },
            Stmt::For {var, lo, hi, step, body, ..} => {
                let (env, var) = var.pprint(env);
                let (env, lo) = lo.pprint(env);
                let (env, hi) = hi.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{indent}for {var} in range({lo}, {hi}, {step}):\n{body}"))
            },
            Stmt::While {cond, body, ..} => {
                let (env, cond) = cond.pprint(env);
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                (env, format!("{indent}while {cond}:\n{body}"))
            },
            Stmt::If {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, thn_str) = pprint_iter(thn.iter(), env, "\n");
                let (env, els_str) = pprint_iter(els.iter(), env, "\n");
                let env = env.decr_indent();
                if els.is_empty() {
                    (env, format!("{indent}if {cond}:\n{thn_str}"))
                } else {
                    (env, format!(
                        "{0}if {cond}:\n{thn_str}\n{0}else:\n{els_str}",
                        indent
                    ))
                }
            },
            Stmt::WithGpuContext {body, ..} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{indent}with parir.gpu:\n{body}"))
            },
            Stmt::Scope {body, ..} => {
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                (env, format!("{indent}{body}"))
            },
            Stmt::Call {func, args, ..} => {
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{indent}{func}({args})"))
            },
            Stmt::Label {label, ..} => {
                (env, format!("{indent}parir.label(\"{label}\")"))
            }
        }
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ..} = self;
        id.pprint(env)
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
        (env, format!("def {id}({params}):\n{body}"))
    }
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}

impl fmt::Display for FunDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.pprint_default())
    }
}
