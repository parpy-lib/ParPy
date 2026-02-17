use super::ast::*;
use crate::utils::ast::*;
use crate::utils::pprint::*;

fn print_tuple_shape(sh: &usize) -> String {
    if *sh == 1 {
        "()".to_string()
    } else {
        format!("({sh},)")
    }
}

fn pprint_elem_size(sz: &ElemSize) -> String {
    match sz {
        ElemSize::Bool => "triton.language.int1",
        ElemSize::I8 => "triton.language.int8",
        ElemSize::I16 => "triton.language.int16",
        ElemSize::I32 => "triton.language.int32",
        ElemSize::I64 => "triton.language.int64",
        ElemSize::U8 => "triton.language.uint8",
        ElemSize::U16 => "triton.language.uint16",
        ElemSize::U32 => "triton.language.uint32",
        ElemSize::U64 => "triton.language.uint64",
        ElemSize::F16 => "triton.language.float16",
        ElemSize::F32 => "triton.language.float32",
        ElemSize::F64 => "triton.language.float64",
    }.to_string()
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
            UnOp::Sub | UnOp::Not | UnOp::BitNeg | UnOp::Addressof => false,
            UnOp::Sqrt => true,
        }
    }

    fn print_unop(op: &UnOp, _argty: &Type) -> Option<String> {
        let s = match op {
            UnOp::Sub => Some("-"),
            UnOp::Not => Some("not "),
            UnOp::BitNeg => Some("~"),
            UnOp::Addressof => None,
            UnOp::Sqrt => Some("triton.language.sqrt"),
        }?;
        Some(s.to_string())
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
            BinOp::BitOr | BinOp::BitShl | BinOp::BitShr | BinOp::BitXor |
            BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt |
            BinOp::Gt => true,
            BinOp::Max | BinOp::Min => false
        }
    }

    fn print_binop(op: &BinOp, _argty: &Type, _ty: &Type) -> Option<String> {
        let s = match op {
            BinOp::Add => Some("+"),
            BinOp::Sub => Some("-"),
            BinOp::Mul => Some("*"),
            BinOp::FloorDiv => Some("//"),
            BinOp::Div => Some("/"),
            BinOp::Rem => Some("%"),
            BinOp::Pow => Some("**"),
            BinOp::And => Some("and"),
            BinOp::Or => Some("or"),
            BinOp::BitAnd => Some("&"),
            BinOp::BitOr => Some("|"),
            BinOp::BitXor => Some("^"),
            BinOp::BitShl => Some("<<"),
            BinOp::BitShr => Some(">>"),
            BinOp::Eq => Some("=="),
            BinOp::Neq => Some("!="),
            BinOp::Leq => Some("<="),
            BinOp::Geq => Some(">="),
            BinOp::Lt => Some("<"),
            BinOp::Gt => Some(">"),
            BinOp::Max => Some("triton.language.maximum"),
            BinOp::Min => Some("triton.language.minimum"),
        }?;
        Some(s.to_string())
    }

    fn associativity(op: &BinOp) -> Assoc {
        match op {
            BinOp::Pow => Assoc::Right,
            _ => Assoc::Left
        }
    }
}

impl PrettyPrint for ReduceOp {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            ReduceOp::Min => "triton.language.min",
            ReduceOp::Max => "triton.language.max",
            ReduceOp::Sum => "triton.language.sum",
            ReduceOp::Prod => "_parpy_builtin_prod",
            ReduceOp::Any => "_parpy_builtin_any",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ty: _, i: _} => id.pprint(env),
            Expr::Bool {v, ty: _, i: _} => {
                (env, if *v { "True" } else { "False" }.to_string())
            },
            Expr::Int {v, ty: _, i: _} => (env, format!("{v}")),
            Expr::Float {v, ty: _, i: _} => {
                if v.is_infinite() {
                    (env, format!("float(\"{v:?}\")"))
                } else {
                    (env, format!("{v:?}"))
                }
            },
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
            Expr::Reduce {op, arg, ty: _, i: _} => {
                let (env, op) = op.pprint(env);
                let (env, arg) = arg.pprint(env);
                (env, format!("{op}({arg})"))
            },
            Expr::Call {id, args, ty: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::ExtCall {id, args, ty: _, i: _} => {
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::ProgramId {dim: Dim::X, ty: _, i: _} => (env, format!("triton.language.program_id(0)")),
            Expr::ProgramId {dim: Dim::Y, ty: _, i: _} => (env, format!("triton.language.program_id(1)")),
            Expr::ProgramId {dim: Dim::Z, ty: _, i: _} => (env, format!("triton.language.program_id(2)")),
            Expr::Arange {lo, hi, ty: _, i: _} => (env, format!("triton.language.arange({lo}, {hi})")),
            Expr::Load {ptr, mask, ty: _, i: _} => {
                let (env, ptr) = ptr.pprint(env);
                match mask {
                    Some(m) => {
                        let (env, m) = m.pprint(env);
                        (env, format!("triton.language.load({ptr}, mask={m})"))
                    },
                    None => (env, format!("triton.language.load({ptr})"))
                }
            },
            Expr::Store {ptr, value, mask, ty: _, i: _} => {
                let (env, ptr) = ptr.pprint(env);
                let (env, value) = value.pprint(env);
                match mask {
                    Some(m) => {
                        let (env, m) = m.pprint(env);
                        (env, format!("triton.language.store({ptr}, {value}, mask={m})"))
                    },
                    None => (env, format!("triton.language.store({ptr}, {value})"))
                }
            },
            Expr::Full {shape, value, elem_sz, ty: _, i: _} => {
                let shape = print_tuple_shape(shape);
                let (env, value) = value.pprint(env);
                let elem_sz = pprint_elem_size(elem_sz);
                (env, format!("triton.language.full({shape}, {value}, {elem_sz})"))
            },
            Expr::Where {cond, thn, els, ty: _, i: _} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("triton.language.where({cond}, {thn}, {els})"))
            },
        }
    }
}

impl PrettyPrintCond<Expr> for Stmt {
    fn extract_if<'a>(&'a self) -> Option<(&'a Expr, &'a Vec<Stmt>, &'a Vec<Stmt>)> {
        if let Stmt::If {cond, thn, els, ..} = self {
            Some((cond, thn, els))
        } else {
            None
        }
    }

    fn extract_elseif<'a>(&'a self) -> Option<(&'a Expr, &'a Vec<Stmt>, &'a Vec<Stmt>)> {
        if let Stmt::If {els: outer_els, ..} = self {
            if let [Stmt::If {cond, thn, els, ..}] = &outer_els[..] {
                Some((cond, thn, els))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Assign {dst, expr, i: _} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{0}{dst} = {expr}", indent))
            },
            Stmt::For {var, lo, hi, step, body, i: _} => {
                let (env, var) = var.pprint(env);
                let (env, lo) = lo.pprint(env);
                let (env, hi) = hi.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}for {var} in range({lo}, {hi}, {step}):\n{body}", indent))
            },
            Stmt::While {cond, body, i: _} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}while {cond}:\n{body}", indent))
            },
            Stmt::If {..} => self.print_cond_pythonic(env),
            Stmt::Return {value, i: _} => {
                let (env, value) = value.pprint(env);
                (env, format!("{0}return {value}", indent))
            },
            Stmt::Expr {e, i: _} => {
                let (env, e) = e.pprint(env);
                (env, format!("{0}{e}", indent))
            },
            Stmt::Barrier {i: _} => (env, format!("{0}triton.language.debug_barrier()", indent)),
            Stmt::KernelLaunch {id, block_dims, args, nwarps, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, block_dims) = block_dims.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{0}{id}[lambda _: ({block_dims})]({args}, num_warps={nwarps})", indent))
            },
        }
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::Import {package, as_str, i: _} => {
                if let Some(s) = as_str {
                    (env, format!("import {package} as {s}"))
                } else {
                    (env, format!("import {package}"))
                }
            },
            Top::FunDef {triton_jit, id, params, body, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let prefix = if *triton_jit { "@triton.jit\n" } else { "" };
                (env, format!("{prefix}def {id}({params}):\n{body}"))
            },
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Ast {tops} = self;
        pprint_iter(tops.iter(), env, "\n")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;
    use crate::triton::ast_builder::*;
    use crate::utils::name::Name;

    #[test]
    fn print_variable() {
        assert_eq!(var("x", None).pprint_default(), "x");
    }

    #[test]
    fn print_program_id() {
        let e = Expr::ProgramId {dim: Dim::Y, ty: Type::Void, i: i()};
        assert_eq!(e.pprint_default(), "triton.language.program_id(1)");
    }

    #[test]
    fn print_arange() {
        let e = Expr::Arange {lo: 0, hi: 16, ty: Type::Void, i: i()};
        assert_eq!(e.pprint_default(), "triton.language.arange(0, 16)");
    }

    #[test]
    fn print_load() {
        let e = Expr::Load {
            ptr: Box::new(var("x", None)),
            mask: None,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.load(x)");
    }

    #[test]
    fn print_load_with_mask() {
        let e = Expr::Load {
            ptr: Box::new(var("x", None)),
            mask: Some(Box::new(var("y", None))),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.load(x, mask=y)");
    }

    #[test]
    fn print_store() {
        let e = Expr::Store {
            ptr: Box::new(var("x", None)),
            value: Box::new(var("y", None)),
            mask: None,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.store(x, y)");
    }

    #[test]
    fn print_store_with_mask() {
        let e = Expr::Store {
            ptr: Box::new(var("x", None)),
            value: Box::new(var("y", None)),
            mask: Some(Box::new(var("z", None))),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.store(x, y, mask=z)");
    }

    #[test]
    fn print_full() {
        let e = Expr::Full {
            shape: 32,
            value: Box::new(int(1)),
            elem_sz: ElemSize::I32,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.full((32,), 1, triton.language.int32)");
    }

    #[test]
    fn print_full_singleton() {
        let e = Expr::Full {
            shape: 1,
            value: Box::new(float(1.0)),
            elem_sz: ElemSize::F32,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.full((), 1.0, triton.language.float32)");
    }

    #[test]
    fn print_where() {
        let e = Expr::Where {
            cond: Box::new(var("x", None)),
            thn: Box::new(var("y", None)),
            els: Box::new(var("z", None)),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "triton.language.where(x, y, z)");
    }

    #[test]
    fn print_assign() {
        let s = Stmt::Assign {
            dst: Name::sym_str("x"),
            expr: var("y", None),
            i: i()
        };
        assert_eq!(s.pprint_default(), "x = y");
    }

    #[test]
    fn print_barrier() {
        let s = Stmt::Barrier {i: i()};
        assert_eq!(s.pprint_default(), "triton.language.debug_barrier()")
    }

    #[test]
    fn print_import() {
        let t = Top::Import {
            package: "triton".to_string(),
            as_str: None,
            i: i()
        };
        assert_eq!(t.pprint_default(), "import triton");
    }

    #[test]
    fn print_import_as() {
        let t = Top::Import {
            package: "triton.language".to_string(),
            as_str: Some("tl".to_string()),
            i: i()
        };
        assert_eq!(t.pprint_default(), "import triton.language as tl")
    }

    #[test]
    fn print_fun_def() {
        let t = Top::FunDef {
            triton_jit: true,
            id: Name::sym_str("f"),
            params: vec![
                Name::sym_str("x"),
                Name::sym_str("y"),
            ],
            body: vec![
                Stmt::Assign {dst: Name::sym_str("w"), expr: var("k", None), i: i()}
            ],
            i: i()
        };
        assert_eq!(t.pprint_default(), "@triton.jit\ndef f(x, y):\n  w = k");
    }
}
