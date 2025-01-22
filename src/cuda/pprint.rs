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
            Type::Void => (env, "void".to_string()),
            Type::Boolean => (env, "bool".to_string()),
            Type::Scalar {sz} => sz.pprint(env),
            Type::Pointer {sz} => {
                let (env, sz) = sz.pprint(env);
                (env, format!("{sz}*"))
            },
            Type::Struct {id} => {
                let (env, id) = id.pprint(env);
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

impl PrettyPrint for Dim {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Dim::X => "x",
            Dim::Y => "y",
            Dim::Z => "z",
        };
        (env, s.to_string())
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
            Expr::ArrayAccess {target, idx, ..} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::Struct {id, fields, ..} => {
                let (env, id) = id.pprint(env);
                let (env, fields) = fields.iter()
                    .fold((env, vec![]), |(env, mut strs), (id, e)| {
                        let (env, e) = e.pprint(env);
                        strs.push(format!("{id}: {e}"));
                        (env, strs)
                    });
                let fields = fields.into_iter().join(", ");
                (env, format!("{id} {{{fields}}}"))
            },
            Expr::Convert {e, ty} => {
                let (env, e) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                (env, format!("({ty}){e}"))
            },
            Expr::ThreadIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("threadIdx.{dim}"))
            },
            Expr::BlockIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("blockIdx.{dim}"))
            },
            Expr::Exp {arg, ty, ..} => {
                let fun = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "hexp",
                    Some(ElemSize::F32) => "__expf",
                    Some(ElemSize::F64) => "exp",
                    _ => panic!(""),
                };
                let (env, arg) = arg.pprint(env);
                (env, format!("{fun}({arg})"))
            },
            Expr::Inf {ty, ..} => {
                let s = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "CUDART_INF_FP16",
                    Some(ElemSize::F32) => "(1.0f / 0.0f)",
                    Some(ElemSize::F64) => "(1.0 / 0.0)",
                    _ => panic!("")
                };
                (env, s.to_string())
            },
            Expr::Log {arg, ty, ..} => {
                let fun = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "hlog",
                    Some(ElemSize::F32) => "__logf",
                    Some(ElemSize::F64) => "log",
                    _ => panic!("")
                };
                let (env, arg) = arg.pprint(env);
                (env, format!("{fun}({arg})"))
            },
            Expr::Max {lhs, rhs, ty, ..} => {
                let fun = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "__hmax",
                    Some(ElemSize::F32) => "fmaxf",
                    Some(ElemSize::F64) => "fmax",
                    Some(_) => "max",
                    None => panic!("")
                };
                let (env, lhs) = lhs.pprint(env);
                let (env, rhs) = rhs.pprint(env);
                (env, format!("{fun}({lhs}, {rhs})"))
            },
            Expr::Min {lhs, rhs, ty, ..} => {
                let fun = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "__hmin",
                    Some(ElemSize::F32) => "fminf",
                    Some(ElemSize::F64) => "fmin",
                    Some(_) => "min",
                    None => panic!("")
                };
                let (env, lhs) = lhs.pprint(env);
                let (env, rhs) = rhs.pprint(env);
                (env, format!("{fun}({lhs}, {rhs})"))
            },
        }
    }
}

impl PrettyPrint for Dim3 {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Dim3 {x, y, z} = self;
        (env, format!("{x}, {y}, {z}"))
    }
}

impl PrettyPrint for LaunchArgs {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let LaunchArgs {blocks, threads} = self;
        let (env, blocks) = blocks.pprint(env);
        let (env, threads) = threads.pprint(env);
        let indent = env.print_indent();
        let s = format!(
            "{0}dim3 tpb({1});\n{0}dim3 blocks({2});",
            indent, threads, blocks
        );
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
            Stmt::For {var_ty, var, init, cond, incr, body, ..} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let incr = incr.to_string();
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!(
                    "{0}for ({1} {2} = {3}; {2} < {4}; {2} += {5}) {{\n{6}\n{0}}}",
                    indent, var_ty, var, init, cond, incr, body
                );
                (env, s)
            },
            Stmt::If {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, thn) = pprint_iter(thn.iter(), env, "\n");
                let (env, s) = if els.is_empty() {
                    let s = format!(
                        "{0}if ({1}) {{\n{2}\n{0}}}",
                        indent, cond, thn
                    );
                    (env, s)
                } else {
                    let (env, els) = pprint_iter(els.iter(), env, "\n");
                    let s = format!(
                        "{0}if ({1}) {{\n{2}\n{0}}} else {{\n{3}\n{0}}}",
                        indent, cond, thn, els
                    );
                    (env, s)
                };
                let env = env.decr_indent();
                (env, s)
            },
            Stmt::Syncthreads {..} => {
                (env, "__syncthreads();".to_string())
            },
            Stmt::KernelLaunch {id, launch_args, args, ..} => {
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let inner_indent = env.print_indent();
                let (env, launch_args) = launch_args.pprint(env);
                let env = env.decr_indent();
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let s = format!(
                    "{0}{{\n{1}\n{2}{3}<<<blocks, tpb>>>({4});\n{0}}}",
                    indent, launch_args, inner_indent, id, args
                );
                (env, s)
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
        let (env, id) = id.pprint(env);
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
            Top::FunDef {attr, id, params, body, ..} => {
                let (env, attr) = attr.pprint(env);
                let (env, id) = id.pprint(env);
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
    use crate::utils::pprint;

    fn var(s: &str) -> Expr {
        Expr::Var {id: Name::new(s.to_string()), ty: Type::Boolean, i: Info::default()}
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: Type::Scalar {sz: ElemSize::I64}, i: Info::default()}
    }

    fn bop(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr::BinOp {
            lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty: Type::Boolean,
            i: Info::default()
        }
    }

    fn add(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Add, rhs)
    }

    fn mul(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Mul, rhs)
    }

    fn scalar_ty(sz: ElemSize) -> Type {
        Type::Scalar {sz}
    }

    fn int64_ty() -> Type {
        scalar_ty(ElemSize::I64)
    }

    #[test]
    fn pprint_precedence_same_level() {
        let s = add(var("x"), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_same_level_paren() {
        let s = add(add(var("x"), var("y")), var("z")).pprint_default();
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_print_paren() {
        let s = mul(add(var("x"), var("y")), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "(x + y) * (y + z)");
    }

    #[test]
    fn pprint_precedence_omit_paren() {
        let s = add(var("x"), add(mul(var("y"), var("y")), var("z"))).pprint_default();
        assert_eq!(&s, "x + y * y + z");
    }

    #[test]
    fn pprint_dims() {
        assert_eq!(Dim::X.pprint_default(), "x");
        assert_eq!(Dim::Y.pprint_default(), "y");
        assert_eq!(Dim::Z.pprint_default(), "z");
    }

    #[test]
    fn pprint_thread_idx_x() {
        let s = Expr::ThreadIdx {dim: Dim::X, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "threadIdx.x");
    }

    #[test]
    fn pprint_block_idx_y() {
        let s = Expr::BlockIdx {dim: Dim::Y, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "blockIdx.y");
    }

    fn exp(arg: Expr, ty: Type) -> Expr {
        Expr::Exp {arg: Box::new(arg), ty, i: Info::default()}
    }

    fn log(arg: Expr, ty: Type) -> Expr {
        Expr::Log {arg: Box::new(arg), ty, i: Info::default()}
    }

    fn max(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
        Expr::Max {
            lhs: Box::new(lhs), rhs: Box::new(rhs), ty, i: Info::default()
        }
    }

    #[test]
    fn pprint_exp_f32() {
        let s = exp(var("x"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "__expf(x)");
    }

    #[test]
    #[should_panic]
    fn pprint_exp_int_type_fails() {
        exp(var("x"), scalar_ty(ElemSize::I32)).pprint_default();
    }

    #[test]
    #[should_panic]
    fn pprint_exp_invalid_type_fails() {
        exp(var("x"), Type::Boolean).pprint_default();
    }

    #[test]
    fn pprint_log_f64() {
        let s = log(var("x"), scalar_ty(ElemSize::F64)).pprint_default();
        assert_eq!(&s, "log(x)");
    }

    #[test]
    fn pprint_max_f32() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "fmaxf(x, y)");
    }

    #[test]
    fn pprint_max_i64() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::I64)).pprint_default();
        assert_eq!(&s, "max(x, y)");
    }

    #[test]
    fn pprint_launch_args() {
        let args = LaunchArgs {
            blocks: Dim3 {x: 1, y: 2, z: 3},
            threads: Dim3 {x: 4, y: 5, z: 6},
        };
        let s = args.pprint_default();
        assert_eq!(&s, "dim3 tpb(4, 5, 6);\ndim3 blocks(1, 2, 3);");
    }

    #[test]
    fn pprint_syncthreads() {
        let s = Stmt::Syncthreads {i: Info::default()}.pprint_default();
        assert_eq!(&s, "__syncthreads();");
    }

    #[test]
    fn pprint_for_loop() {
        let for_loop = Stmt::For {
            var_ty: scalar_ty(ElemSize::I64),
            var: Name::new("i".to_string()),
            init: int(0),
            cond: int(10),
            incr: 1,
            body: vec![
                Stmt::Assign {dst: var("x"), expr: var("y"), i: Info::default()}
            ],
            i: Info::default(),
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "for (int64_t i = 0; i < 10; i += 1) {{\n{indent}x = y;\n}}"
        );
        assert_eq!(for_loop.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond() {
        let cond = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(var("x")),
                op: BinOp::Eq,
                rhs: Box::new(var("y")),
                ty: Type::Boolean,
                i: Info::default()
            },
            thn: vec![
                Stmt::Assign {dst: var("x"), expr: var("y"), i: Info::default()},
            ],
            els: vec![
                Stmt::Assign {dst: var("y"), expr: var("x"), i: Info::default()}
            ],
            i: Info::default()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "if (x == y) {{\n{0}x = y;\n}} else {{\n{0}y = x;\n}}", indent
        );
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_empty_else() {
        let cond = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(var("x")),
                op: BinOp::Eq,
                rhs: Box::new(var("y")),
                ty: Type::Boolean,
                i: Info::default()
            },
            thn: vec![
                Stmt::Assign {dst: var("x"), expr: var("y"), i: Info::default()},
            ],
            els: vec![],
            i: Info::default()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("if (x == y) {{\n{indent}x = y;\n}}");
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_elseif() {

    }

    #[test]
    fn pprint_kernel_launch() {
        let id = "kernel";
        let kernel = Stmt::KernelLaunch {
            id: Name::new(id.to_string()),
            launch_args: LaunchArgs {
                blocks: Dim3 {x: 1, y: 1, z: 1},
                threads: Dim3 {x: 128, y: 1, z: 1}
            },
            args: vec![var("x"), var("y")],
            i: Info::default()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "{{\n{0}dim3 tpb(128, 1, 1);\n{0}dim3 blocks(1, 1, 1);\n{0}{1}<<<blocks, tpb>>>(x, y);\n}}",
            indent, id
        );
        assert_eq!(kernel.pprint_default(), expected);
    }

    #[test]
    fn pprint_struct_def() {
        let def = Top::StructDef {
            id: Name::new("point".to_string()),
            fields: vec![
                Field {id: Name::new("x".to_string()), ty: scalar_ty(ElemSize::F32), i: Info::default()},
                Field {id: Name::new("y".to_string()), ty: scalar_ty(ElemSize::F32), i: Info::default()},
            ],
            i: Info::default()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "struct point {{\n{0}float x;\n{0}float y;\n}};", indent
        );
        assert_eq!(def.pprint_default(), expected)
    }

    #[test]
    fn pprint_attributes() {
        assert_eq!(Attribute::Global.pprint_default(), "__global__");
        assert_eq!(Attribute::Host.pprint_default(), "__host__");
    }

    #[test]
    fn pprint_fun_def() {
        let def = Top::FunDef {
            ret_ty: Type::Void,
            attr: Attribute::Global,
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![
                Stmt::Assign {dst: var("x"), expr: var("y"), i: Info::default()}
            ],
            i: Info::default()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("__global__\nvoid f() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }
}

