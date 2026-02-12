use super::ast::*;
use crate::utils::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;

fn print_tuple_shape(sh: &Vec<i64>) -> String {
    match &sh[..] {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => format!("({0})", sh.iter().join(", "))
    }
}

fn pprint_elem_size(sz: &ElemSize) -> String {
    match sz {
        ElemSize::Bool => "tl.bool",
        ElemSize::I8 => "tl.int8",
        ElemSize::I16 => "tl.int16",
        ElemSize::I32 => "tl.int32",
        ElemSize::I64 => "tl.int64",
        ElemSize::U8 => "tl.uint8",
        ElemSize::U16 => "tl.uint16",
        ElemSize::U32 => "tl.uint32",
        ElemSize::U64 => "tl.uint64",
        ElemSize::F16 => "tl.float16",
        ElemSize::F32 => "tl.float32",
        ElemSize::F64 => "tl.float64",
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
            UnOp::Not => Some("not"),
            UnOp::BitNeg => Some("~"),
            UnOp::Addressof => None,
            UnOp::Sqrt => Some("tl.sqrt"),
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
            BinOp::And => Some(" and "),
            BinOp::Or => Some(" or "),
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
            BinOp::Max => Some("tl.maximum"),
            BinOp::Min => Some("tl.minimum"),
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
            ReduceOp::Min => "tl.min",
            ReduceOp::Max => "tl.max",
            ReduceOp::Sum => "tl.sum",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ty: _, i: _} => id.pprint(env),
            Expr::Bool {v, ty: _, i: _} => (env, format!("{v}")),
            Expr::Int {v, ty: _, i: _} => (env, format!("{v}")),
            Expr::Float {v, ty: _, i: _} => (env, format!("{v:?}")),
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
            Expr::ProgramId {dim: Dim::X, ty: _, i: _} => (env, format!("tl.program_id(0)")),
            Expr::ProgramId {dim: Dim::Y, ty: _, i: _} => (env, format!("tl.program_id(0)")),
            Expr::ProgramId {dim: Dim::Z, ty: _, i: _} => (env, format!("tl.program_id(0)")),
            Expr::Arange {lo, hi, ty: _, i: _} => (env, format!("tl.arange({lo}, {hi})")),
            Expr::Load {ptr, mask, ty: _, i: _} => {
                let (env, ptr) = ptr.pprint(env);
                match mask {
                    Some(m) => {
                        let (env, m) = m.pprint(env);
                        (env, format!("tl.load({ptr}, mask={m})"))
                    },
                    None => (env, format!("tl.load({ptr})"))
                }
            },
            Expr::Store {ptr, value, mask, ty: _, i: _} => {
                let (env, ptr) = ptr.pprint(env);
                let (env, value) = value.pprint(env);
                match mask {
                    Some(m) => {
                        let (env, m) = m.pprint(env);
                        (env, format!("tl.store({ptr}, {value}, mask={m})"))
                    },
                    None => (env, format!("tl.store({ptr}, {value})"))
                }
            },
            Expr::Full {shape, value, elem_sz, ty: _, i: _} => {
                let shape = print_tuple_shape(shape);
                let (env, value) = value.pprint(env);
                let elem_sz = pprint_elem_size(elem_sz);
                (env, format!("tl.full({shape}, {value}, {elem_sz})"))
            },
            Expr::Where {cond, thn, els, ty: _, i: _} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("tl.where({cond}, {thn}, {els})"))
            },
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
                let ii = env.print_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}for {var} in range({lo}, {hi}, {step}):\n{ii}{body}", indent))
            },
            Stmt::While {cond, body, i: _} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let ii = env.print_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}while {cond}:\n{ii}{body}", indent))
            },
            Stmt::If {cond, thn, els, i: _} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let ii = env.print_indent();
                let (env, thn) = pprint_iter(thn.iter(), env, "\n");
                let (env, els) = pprint_iter(els.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}if {cond}:\n{1}{thn}\n{0}else:\n{1}{els}", indent, ii))
            },
            Stmt::Return {value, i: _} => {
                let (env, value) = value.pprint(env);
                (env, format!("{0}return {value}", indent))
            },
            Stmt::Expr {e, i: _} => {
                let (env, e) = e.pprint(env);
                (env, format!("{0}{e}", indent))
            },
            Stmt::Barrier {i: _} => (env, format!("{0}tl.debug_barrier()", indent)),
        }
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty} = self;
        let (env, id) = id.pprint(env);
        match ty {
            Some(sz) => (env, format!("{id} : {0}", pprint_elem_size(sz))),
            None => (env, id)
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
            Top::TritonFunDef {id, params, body, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("@triton.jit\ndef {id}({params}):\n{body}"))
            },
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Ast {tops} = self;
        let (env, tops) = pprint_iter(tops.iter(), env, "\n");
        (env, format!("{tops}"))
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
        assert_eq!(var("x").pprint_default(), "x");
    }

    #[test]
    fn print_program_id() {
        let e = Expr::ProgramId {dim: Dim::Y, ty: Type::Void, i: i()};
        assert_eq!(e.pprint_default(), "tl.program_id(1)");
    }

    #[test]
    fn print_arange() {
        let e = Expr::Arange {lo: 0, hi: 16, ty: Type::Void, i: i()};
        assert_eq!(e.pprint_default(), "tl.arange(0, 16)");
    }

    #[test]
    fn print_load() {
        let e = Expr::Load {
            ptr: Box::new(var("x")),
            mask: None,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.load(x)");
    }

    #[test]
    fn print_load_with_mask() {
        let e = Expr::Load {
            ptr: Box::new(var("x")),
            mask: Some(Box::new(var("y"))),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.load(x, mask=y)");
    }

    #[test]
    fn print_store() {
        let e = Expr::Store {
            ptr: Box::new(var("x")),
            value: Box::new(var("y")),
            mask: None,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.store(x, y)");
    }

    #[test]
    fn print_store_with_mask() {
        let e = Expr::Store {
            ptr: Box::new(var("x")),
            value: Box::new(var("y")),
            mask: Some(Box::new(var("z"))),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.store(x, y, mask=z)");
    }

    #[test]
    fn print_full() {
        let e = Expr::Full {
            shape: vec![32],
            value: Box::new(int(1)),
            elem_sz: ElemSize::I32,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.full((32,), 1, tl.int32)");
    }

    #[test]
    fn print_where() {
        let e = Expr::Where {
            cond: Box::new(var("x")),
            thn: Box::new(var("y")),
            els: Box::new(var("z")),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.where(x, y, z)");
    }

    #[test]
    fn print_assign() {
        let s = Stmt::Assign {
            dst: Name::sym_str("x"),
            expr: var("y"),
            i: i()
        };
        assert_eq!(s.pprint_default(), "x = y");
    }

    #[test]
    fn print_barrier() {
        let s = Stmt::Barrier {i: i()};
        assert_eq!(s.pprint_default(), "tl.debug_barrier()")
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
        let t = Top::TritonFunDef {
            id: Name::sym_str("f"),
            params: vec![
                Param {id: Name::sym_str("x"), ty: None},
                Param {id: Name::sym_str("y"), ty: Some(ElemSize::F32)},
            ],
            body: vec![
                Stmt::Assign {dst: Name::sym_str("w"), expr: var("k"), i: i()}
            ],
            i: i()
        };
        assert_eq!(t.pprint_default(), "@triton.jit\ndef f(x, y : tl.float32):\n  w = k");
    }
}
