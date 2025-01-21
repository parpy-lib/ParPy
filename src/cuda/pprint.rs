use super::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BTreeMap;

impl PrettyPrint for ElemSize {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            ElemSize::I8 => "int8_t",
            ElemSize::I16 => "int16_t",
            ElemSize::I32 => "int32_t",
            ElemSize::I64 => "int64_t",
            ElemSize::U8 => "uint8_t",
            ElemSize::F16 => "half",
            ElemSize::F32 => "float",
            ElemSize::F64 => "double",
        };
        (env, s.to_string())
    }
}

impl<T: PrettyPrint> PrettyPrint for BTreeMap<String, T> {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (env, s) = self.iter()
            .fold((env, vec![]), |(env, mut strs), (id, v)| {
                let (env, s) = v.pprint(env);
                let s = format!("{0}: {1}", id, s);
                strs.push(s);
                (env, strs)
            });
        (env, s.into_iter().join(", "))
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Boolean => (env, "bool".to_string()),
            Type::Tensor {sz, ..} => sz.pprint(env),
            Type::Struct {id} => {
                let (env, id) = pprint_var(env, &id);
                (env, format!("struct {id}"))
            },
        }
    }
}

impl PrettyPrint for UnOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            UnOp::Sub => (env, "-".to_string()),
        }
    }
}

impl PrettyPrint for BinOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            BinOp::Add => (env, "+".to_string()),
            BinOp::Sub => (env, "-".to_string()),
            BinOp::Mul => (env, "*".to_string()),
            BinOp::Div | BinOp::FloorDiv => (env, "/".to_string()),
            BinOp::Mod => (env, "%".to_string()),
            BinOp::BitAnd => (env, "&".to_string()),
            BinOp::Eq => (env, "==".to_string()),
            BinOp::Neq => (env, "!=".to_string()),
            BinOp::Lt => (env, "<".to_string()),
            BinOp::Gt => (env, ">".to_string()),
        }
    }
}

fn try_get_binop(e: &Box<Expr>) -> Option<BinOp> {
    match e.borrow() {
        Expr::BinOp {op, ..} => Some(op.clone()),
        _ => None
    }
}

fn parenthesize_if_lower_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp, 
    s: String
) -> String {
    match inner_op_opt {
        Some(inner_op) => {
            // If the inner operator has a lower precedence than the outer operator, we add
            // parentheses to enforce the correct evaluation order.
            if let Ordering::Less = inner_op.precedence().cmp(&outer_op.precedence()) {
                format!("({s})")
            } else {
                s
            }
        }
        None => s,
    }
}

fn pprint_builtin(
    func: &Builtin,
    args_str: String,
    ty: &Type
) -> String {
    let sz = if let Some(sz) = ty.get_scalar_elem_size() {
        sz
    } else {
        let (_, ty_str) = ty.pprint(PrettyPrintEnv::new());
        panic!("Failed to compile built-in {func} due to invalid type {ty_str}")
    };
    let s = match func {
        Builtin::Exp => {
            match sz {
                ElemSize::F16 => "hexp",
                ElemSize::F32 => "__expf",
                ElemSize::F64 => "exp",
                _ => panic!("")
            }
        },
        Builtin::Inf => {
            match sz {
                ElemSize::F16 => "CUDART_INF_FP16",
                ElemSize::F32 => "(1.0f / 0.0f)",
                ElemSize::F64 => "(1.0 / 0.0)",
                _ => panic!("")
            }
        },
        Builtin::Log => {
            match sz {
                ElemSize::F16 => "hlog",
                ElemSize::F32 => "__logf",
                ElemSize::F64 => "log",
                _ => panic!("")
            }
        },
        Builtin::Max => {
            match sz {
                ElemSize::F16 => "__hmax",
                ElemSize::F32 => "fmaxf",
                ElemSize::F64 => "fmax",
                _ => "max"
            }
        },
        Builtin::Min => {
            match sz {
                ElemSize::F16 => "__hmin",
                ElemSize::F32 => "fminf",
                ElemSize::F64 => "fmin",
                _ => "min"
            }
        },
    };
    if args_str.is_empty() {
        s.to_string()
    } else {
        format!("{s}({args_str})")
    }
}


impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => (env, id.to_string()),
            Expr::Int {v, ..} => (env, v.to_string()),
            Expr::Float {v, ..} => (env, v.to_string()),
            Expr::UnOp {op, arg, ..} => {
                let (env, op_str) = op.pprint(env);
                let (env, arg_str) = arg.pprint(env);
                (env, format!("{op_str}{arg_str}"))
            },
            Expr::BinOp {lhs, op, rhs, ..} => {
                let (env, lhs_str) = lhs.pprint(env);
                let (env, op_str) = op.pprint(env);
                let (env, rhs_str) = rhs.pprint(env);

                // We consider the precedence of the left- and right-hand side operands and use
                // this to omit parentheses when they are unnecessary.
                let lhs_op = try_get_binop(lhs);
                let rhs_op = try_get_binop(rhs);
                let lhs_str = parenthesize_if_lower_precedence(lhs_op, &op, lhs_str);
                let rhs_str = parenthesize_if_lower_precedence(rhs_op, &op, rhs_str);
                (env, format!("{lhs_str} {op_str} {rhs_str}"))
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
            Expr::Struct {id, fields, ..} => {
                let (env, id) = pprint_var(env, &id);
                let (env, fields) = fields.iter()
                    .fold((env, vec![]), |(env, mut strs), (id, e)| {
                        let (env, e) = e.pprint(env);
                        strs.push(format!("{id}: {e}"));
                        (env, strs)
                    });
                let fields = fields.into_iter().join(", ");
                (env, format!("struct {id} {{{fields}}}"))
            },
            Expr::Builtin {func, args, ty, ..} => {
                let (env, args_str) = pprint_iter(args.iter(), env, ", ");
                let builtin = pprint_builtin(func, args_str, ty);
                (env, builtin)
            },
            Expr::Convert {e, ty} => {
                let (env, e) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                (env, format!("({ty}){e}"))
            },
        }
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Definition {ty, id, expr, ..} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = pprint_var(env, &id);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{ty} {id} = {expr};"))
            },
            Stmt::Assign {dst, expr, ..} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{dst} = {expr};"))
            },
            Stmt::For {init, cond, incr, body, ..} => {
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}for ({init}; {cond}; {incr}) {{\n{body}\n{0}}}", indent))
            },
            Stmt::If {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, thn) = pprint_iter(thn.iter(), env, "\n");
                let (env, els) = pprint_iter(els.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{indent}if {cond}:\n{thn}\nelse:\n{els}"))
            },
        }
    }
}

impl PrettyPrint for Attribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Attribute::Global => "__global__",
            Attribute::Host => "__host__",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Field {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Field {id, ty, ..} = self;
        let (env, id) = pprint_var(env, &id);
        let (env, ty) = ty.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{ty} {id};"))
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty, ..} = self;
        let (env, id) = pprint_var(env, &id);
        let (env, ty) = ty.pprint(env);
        (env, format!("{ty} {id}"))
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::StructDef {id, fields, ..} => {
                let (env, id) = pprint_var(env, &id);
                let env = env.incr_indent();
                let (env, fields) = pprint_iter(fields.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("struct {id} {{\n{fields}\n}}"))
            },
            Top::FunDef {attr, id, params, body, ..} => {
                let (env, attr) = attr.pprint(env);
                let (env, id) = pprint_var(env, &id);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{attr}\nvoid {id}({params}) {{\n{body}\n}}"))
            },
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::info::Info;
    use crate::utils::name::Name;

    fn var(s: &str) -> Expr {
        Expr::Var {id: Name::new(s.to_string()), ty: Type::Boolean, i: Info::default()}
    }

    fn bop(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr::BinOp {
            lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty: Type::Boolean, i: Info::default()
        }
    }

    fn add(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Add, rhs)
    }

    fn mul(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Mul, rhs)
    }

    fn print<T: PrettyPrint>(v: T) -> String {
        let (_, s) = v.pprint(PrettyPrintEnv::new());
        s
    }

    #[test]
    fn pprint_precedence_same_level() {
        let s = print(add(var("x"), add(var("y"), var("z"))));
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_same_level_paren() {
        let s = print(add(add(var("x"), var("y")), var("z")));
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_print_paren() {
        let s = print(mul(add(var("x"), var("y")), add(var("y"), var("z"))));
        assert_eq!(&s, "(x + y) * (y + z)");
    }

    #[test]
    fn pprint_precedence_omit_paren() {
        let s = print(add(var("x"), add(mul(var("y"), var("y")), var("z"))));
        assert_eq!(&s, "x + y * y + z");
    }

    fn scalar_ty(sz: ElemSize) -> Type {
        Type::Tensor {sz, shape: vec![]}
    }

    fn pp_builtin(b: Builtin, s: &str, ty: Type) -> String {
        pprint_builtin(&b, s.to_string(), &ty)
    }

    #[test]
    fn pprint_exp_f32() {
        let s = pp_builtin(Builtin::Exp, "x", scalar_ty(ElemSize::F32));
        assert_eq!(&s, "__expf(x)");
    }

    #[test]
    fn pprint_log_f64() {
        let s = pp_builtin(Builtin::Log, "x", scalar_ty(ElemSize::F64));
        assert_eq!(&s, "log(x)");
    }

    #[test]
    fn pprint_max_f32() {
        let s = pp_builtin(Builtin::Max, "x, y", scalar_ty(ElemSize::F32));
        assert_eq!(&s, "fmaxf(x, y)");
    }

    #[test]
    fn pprint_max_i64() {
        let s = pp_builtin(Builtin::Max, "x, y", scalar_ty(ElemSize::I64));
        assert_eq!(&s, "max(x, y)");
    }
}

