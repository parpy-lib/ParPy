use super::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;
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
            Builtin::Prod => "prod",
            Builtin::Convert {..} => "<convert>",
            Builtin::Label => "<label>",
            Builtin::GpuContext => "<gpu_context>",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Type::String => format!("<string>"),
            Type::Tensor {sz, shape} => format!("<tensor|{sz:?},{shape:?}>"),
            Type::Tuple {elems} => format!("<({0})>", elems.into_iter().join(", ")),
            Type::Dict {fields} => {
                let s = fields.into_iter()
                    .map(|(k, v)| format!("'{k}': {v}"))
                    .join(", ");
                format!("<{{{0}}}>", s)
            },
            Type::Void => format!("<void>"),
            Type::Unknown => format!("<unknown>")
        };
        (env, s.to_string())
    }
}

impl PrettyPrintUnOp<Type> for Expr {
    fn extract_unop<'a>(&'a self) -> Option<(&'a UnOp, &'a Expr)> {
        if let Expr::UnOp {op, arg, ..} = self {
            Some((op, arg))
        } else {
            None
        }
    }

    fn is_function(op: &UnOp) -> bool {
        match op {
            UnOp::Sub | UnOp::Not | UnOp::BitNeg => false,
            UnOp::Addressof | UnOp::Exp | UnOp::Log | UnOp::Cos | UnOp::Sin |
            UnOp::Sqrt | UnOp::Tanh | UnOp::Abs => true,
        }
    }

    fn print_unop(op: &UnOp, _argty: &Type) -> String {
        let s = match op {
            UnOp::Sub => "-",
            UnOp::Not => "!",
            UnOp::BitNeg => "~",
            UnOp::Exp => "exp",
            UnOp::Log => "log",
            UnOp::Cos => "cos",
            UnOp::Sin => "sin",
            UnOp::Sqrt => "sqrt",
            UnOp::Tanh => "tanh",
            UnOp::Abs => "abs",
            UnOp::Addressof => "addressof"
        };
        s.to_string()
    }
}

impl PrettyPrint for UnOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        (env, Expr::print_unop(self, &Type::Unknown))
    }
}

impl PrettyPrintBinOp<Type> for Expr {
    fn extract_binop<'a>(&'a self) -> Option<(&'a Expr, &'a BinOp, &'a Expr, &'a Type)> {
        if let Expr::BinOp {lhs, op, rhs, ty, ..} = self {
            Some((lhs, op, rhs, ty))
        } else {
            None
        }
    }

    fn is_infix(op: &BinOp, _argty: &Type) -> bool {
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Div |
            BinOp::Rem | BinOp::Pow | BinOp::And | BinOp::Or | BinOp::BitAnd |
            BinOp::BitOr | BinOp::BitXor | BinOp::BitShl | BinOp::BitShr |
            BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt |
            BinOp::Gt => true,
            BinOp::Max | BinOp::Min | BinOp::Atan2 => false,
        }
    }

    fn print_binop(op: &BinOp, _argty: &Type, _ty: &Type) -> String {
        let s = match op {
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
        s.to_string()
    }

    fn associativity(op: &BinOp) -> Assoc {
        match op {
            BinOp::Pow => Assoc::Right,
            _ => Assoc::Left
        }
    }
}

impl PrettyPrint for BinOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        (env, Expr::print_binop(self, &Type::Unknown, &Type::Unknown))
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::String {v, ..} => (env, format!("{v}")),
            Expr::Bool {v, ..} => {
                (env, if *v { format!("True") } else { format!("False") })
            },
            Expr::Int {v, ..} => (env, format!("{v}")),
            Expr::Float {v, ..} => (env, format!("{v:?}")),
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
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
            Expr::Call {id, args, ..} => {
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::NeutralElement {op, tyof, ..} => {
                let (env, op) = op.pprint(env);
                let (env, tyof) = tyof.pprint(env);
                (env, format!("neutral_element<{op}, {tyof}>"))
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
            Stmt::Return {value, ..} => {
                let (env, value) = value.pprint(env);
                (env, format!("{indent}return {value}"))
            },
            Stmt::WithGpuContext {body, ..} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{indent}with prickle.gpu:\n{body}"))
            },
            Stmt::Scope {body, ..} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{indent}if True: # scope\n{body}"))
            },
            Stmt::Call {func, args, ..} => {
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{indent}{func}({args})"))
            },
            Stmt::Label {label, ..} => {
                (env, format!("{indent}prickle.label(\"{label}\")"))
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

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_iter(self.iter(), env, "\n")
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::py::ast_builder::*;

    #[test]
    fn print_1d_tensor_type() {
        let ty = Type::Tensor {sz: ElemSize::I16, shape: vec![10]};
        assert_eq!(ty.pprint_default(), "<tensor|I16,[10]>");
    }

    #[test]
    fn print_2d_tensor_type() {
        let ty = Type::Tensor {sz: ElemSize::U8, shape: vec![10, 20]};
        assert_eq!(ty.pprint_default(), "<tensor|U8,[10, 20]>");
    }

    #[test]
    fn print_dict_type() {
        let ty = dict_ty(vec![
            ("x", scalar(ElemSize::I32)),
            ("y", scalar(ElemSize::F32))
        ]);
        assert_eq!(ty.pprint_default(), "<{'x': int32, 'y': float32}>");
    }

    fn uint(v: i128) -> Expr {
        int(v, Some(ElemSize::I64))
    }

    fn ufloat(v: f64) -> Expr {
        float(v, Some(ElemSize::F32))
    }

    fn uadd(l: Expr, r: Expr) -> Expr {
        binop(l, BinOp::Add, r, scalar(ElemSize::I64))
    }

    fn umul(l: Expr, r: Expr) -> Expr {
        binop(l, BinOp::Mul, r, scalar(ElemSize::I64))
    }

    fn upow(l: Expr, r: Expr) -> Expr {
        binop(l, BinOp::Pow, r, scalar(ElemSize::F32))
    }

    fn umax(l: Expr, r: Expr) -> Expr {
        binop(l, BinOp::Max, r, scalar(ElemSize::F32))
    }

    #[test]
    fn print_sub_unop() {
        let e = unop(UnOp::Sub, uint(1));
        assert_eq!(e.pprint_default(), "-1");
    }

    #[test]
    fn print_log_unop() {
        let e = unop(UnOp::Log, ufloat(1.5));
        assert_eq!(e.pprint_default(), "log(1.5)");
    }

    #[test]
    fn print_addition_left_assoc_no_paren() {
        let e = uadd(uadd(uint(1), uint(2)), uint(3));
        assert_eq!(e.pprint_default(), "1 + 2 + 3");
    }

    #[test]
    fn print_parenthesized_addition() {
        let e = uadd(uint(1), uadd(uint(2), uint(3)));
        assert_eq!(e.pprint_default(), "1 + (2 + 3)")
    }

    #[test]
    fn print_mul_add_no_paren() {
        let e = uadd(umul(uint(1), uint(2)), uint(3));
        assert_eq!(e.pprint_default(), "1 * 2 + 3");
    }

    #[test]
    fn print_mul_add_paren() {
        let e = umul(uint(1), uadd(uint(2), uint(3)));
        assert_eq!(e.pprint_default(), "1 * (2 + 3)");
    }

    #[test]
    fn print_nested_mul_paren() {
        let e = umul(umul(uint(1), uint(2)), umul(uint(3), uint(4)));
        assert_eq!(e.pprint_default(), "1 * 2 * (3 * 4)");
    }

    #[test]
    fn print_pow_rightassoc() {
        let e = upow(upow(ufloat(1.0), ufloat(2.0)), ufloat(3.0));
        assert_eq!(e.pprint_default(), "(1.0 ** 2.0) ** 3.0");
    }

    #[test]
    fn print_max_func_call_style() {
        let e = umax(ufloat(1.0), ufloat(2.0));
        assert_eq!(e.pprint_default(), "max(1.0, 2.0)");
    }

    #[test]
    fn print_nested_max_func_calls() {
        let e = umax(umax(ufloat(1.0), ufloat(2.0)), umax(ufloat(3.0), ufloat(4.0)));
        assert_eq!(e.pprint_default(), "max(max(1.0, 2.0), max(3.0, 4.0))");
    }

    #[test]
    fn print_slice_assign_implicit_ends() {
        let s = assignment(
            subscript(var("x", scalar(ElemSize::I64)), slice(None, None)),
            ufloat(1.0)
        );
        assert_eq!(s.pprint_default(), "x[:] = 1.0");
    }

    #[test]
    fn print_slice_assign_implicit_upper() {
        let s = assignment(
            subscript(var("x", scalar(ElemSize::I64)), slice(Some(uint(0)), None)),
            ufloat(1.0)
        );
        assert_eq!(s.pprint_default(), "x[0:] = 1.0");
    }

    #[test]
    fn print_slice_assign_implicit_lower() {
        let s = assignment(
            subscript(var("x", scalar(ElemSize::I64)), slice(None, Some(uint(5)))),
            ufloat(1.0)
        );
        assert_eq!(s.pprint_default(), "x[:5] = 1.0");
    }

    #[test]
    fn print_slice_assign_explicit_bounds() {
        let s = assignment(
            subscript(var("x", scalar(ElemSize::I64)), slice(Some(uint(1)), Some(uint(5)))),
            ufloat(1.0)
        );
        assert_eq!(s.pprint_default(), "x[1:5] = 1.0");
    }

    #[test]
    fn print_if_with_else() {
        let thn = vec![assignment(var("x", scalar(ElemSize::I64)), uint(1))];
        let els = vec![assignment(var("x", scalar(ElemSize::I64)), uint(2))];
        let s = if_stmt(bool_expr(true, Some(ElemSize::Bool)), thn, els);
        assert_eq!(s.pprint_default(), "if True:\n  x = 1\nelse:\n  x = 2");
    }

    #[test]
    fn print_if_without_else() {
        let thn = vec![assignment(var("x", scalar(ElemSize::I64)), uint(1))];
        let s = if_stmt(bool_expr(true, Some(ElemSize::Bool)), thn, vec![]);
        assert_eq!(s.pprint_default(), "if True:\n  x = 1");
    }
}
