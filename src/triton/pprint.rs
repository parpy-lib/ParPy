use super::ast::*;
use crate::utils::ast::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;

use itertools::Itertools;

fn pprint_elem_size(sz: &ElemSize) -> String {
    match sz {
        ElemSize::Bool => "tl.int1",
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

fn print_dtype(sz: &ElemSize) -> String {
    match sz {
        ElemSize::Bool => "parpy.types.Bool",
        ElemSize::I8 => "parpy.types.I8",
        ElemSize::I16 => "parpy.types.I16",
        ElemSize::I32 => "parpy.types.I32",
        ElemSize::I64 => "parpy.types.I64",
        ElemSize::U8 => "parpy.types.U8",
        ElemSize::U16 => "parpy.types.U16",
        ElemSize::U32 => "parpy.types.U32",
        ElemSize::U64 => "parpy.types.U64",
        ElemSize::F16 => "parpy.types.F16",
        ElemSize::F32 => "parpy.types.F32",
        ElemSize::F64 => "parpy.types.F64",
    }.to_string()
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Pointer {ty, shape: _} => {
                let (env, ty) = ty.pprint(env);
                (env, format!("tl.pointer_type({ty})"))
            },
            Type::Tensor {sz, shape: _} => (env, pprint_elem_size(&sz)),
            Type::Function {..} => (env, format!("<function type>")),
            Type::List => (env, format!("<list type>")),
            Type::String => (env, format!("<string type>")),
            Type::Void => (env, "void".to_string())
        }
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
            UnOp::Sub | UnOp::BitNeg | UnOp::Addressof => false,
            UnOp::Not | UnOp::Sqrt => true,
        }
    }

    fn print_unop(op: &UnOp, _argty: &Type) -> Option<String> {
        let s = match op {
            UnOp::Sub => Some("-"),
            UnOp::Not => Some("_parpy_builtin_not"),
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
            BinOp::Rem | BinOp::And | BinOp::Or | BinOp::BitAnd | BinOp::BitOr |
            BinOp::BitShl | BinOp::BitShr | BinOp::BitXor | BinOp::Eq |
            BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => true,
            BinOp::Pow | BinOp::Max | BinOp::Min => false
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
            BinOp::Pow => Some("libdevice.pow"),
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
            Expr::String {v, ty: _, i: _} => (env, format!("\"{v}\"")),
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
            Expr::List {elems, ty: _, i: _} => {
                let (env, elems) = pprint_iter(elems.iter(), env, ", ");
                (env, format!("[{elems}]"))
            },
            Expr::ProgramId {dim: Dim::X, ty: _, i: _} => (env, format!("tl.program_id(0)")),
            Expr::ProgramId {dim: Dim::Y, ty: _, i: _} => (env, format!("tl.program_id(1)")),
            Expr::ProgramId {dim: Dim::Z, ty: _, i: _} => (env, format!("tl.program_id(2)")),
            Expr::NumPrograms {dim: Dim::X, ty: _, i: _} => (env, format!("tl.num_programs(0)")),
            Expr::NumPrograms {dim: Dim::Y, ty: _, i: _} => (env, format!("tl.num_programs(1)")),
            Expr::NumPrograms {dim: Dim::Z, ty: _, i: _} => (env, format!("tl.num_programs(2)")),
            Expr::Arange {lo, hi, ty: _, i: _} => {
                let (env, lo) = lo.pprint(env);
                let (env, hi) = hi.pprint(env);
                (env, format!("tl.arange({lo}, {hi})"))
            },
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
            Expr::Full {shape, value, elem_sz, ty: _, i: _} => {
                let (env, shape) = shape.pprint(env);
                let (env, value) = value.pprint(env);
                let elem_sz = pprint_elem_size(elem_sz);
                (env, format!("tl.full(({shape},), {value}, {elem_sz})"))
            },
            Expr::Where {cond, thn, els, ty: _, i: _} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("tl.where({cond}, {thn}, {els})"))
            },
            Expr::Convert {value, ty, i: _} => {
                let (env, value) = value.pprint(env);
                let (env, ty) = ty.pprint(env);
                (env, format!("tl.cast({value}, {ty})"))
            },
            Expr::AllocBuffer {nelems, elem_sz, ty: _, i: _} => {
                let dtype = print_dtype(elem_sz);
                (env, format!("_parpy_builtin_alloc({nelems}, {dtype})"))
            },
            Expr::ToTorch {e, ty: _, i: _} => {
                let (env, e) = e.pprint(env);
                (env, format!("_parpy_builtin_to_torch({e})"))
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
            Stmt::Definition {dst, expr, i: _} |
            Stmt::Assign {dst, expr, i: _} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{0}{dst} = {expr}", indent))
            },
            Stmt::For {var, lo, hi, step, body, i: _} => {
                let (env, var) = var.pprint(env);
                let (env, lo) = lo.pprint(env);
                let (env, hi) = hi.pprint(env);
                let (env, step) = step.pprint(env);
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
            Stmt::Pass {i: _} => (env, format!("{0}pass", indent)),
            Stmt::Barrier {i: _} => (env, format!("{0}tl.debug_barrier()", indent)),
            Stmt::Store {ptr, value, mask, i: _} => {
                let (env, ptr) = ptr.pprint(env);
                let (env, value) = value.pprint(env);
                let (env, mask) = match mask {
                    Some(m) => {
                        let (env, m) = m.pprint(env);
                        (env, format!(", mask={m}"))
                    },
                    None => (env, "".to_string())
                };
                (env, format!("{0}tl.store({ptr}, {value}{mask})", indent))
            },
            Stmt::KernelLaunch {id, block_dims, args, nwarps: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, block_dims) = block_dims.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{0}{id}[lambda _: ({block_dims})]({args})", indent))
            },
        }
    }
}

impl PrettyPrint for AutotuneConfig {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        let AutotuneConfig {mapping, warp_count} = self;
        let (env, m) = mapping.iter()
            .fold((env, vec![]), |(env, mut acc), (id, e)| {
                let (env, id) = id.pprint(env);
                let (env, e) = e.pprint(env);
                acc.push(format!("\"{id}\": {e}"));
                (env, acc)
            });
        let mapping = m.iter().join(", ");
        (env, format!("{indent}triton.Config({{{mapping}}}, num_warps={warp_count})"))
    }
}

impl PrettyPrint for Decorator {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Decorator::Autotune {configs, key, restore_value} => {
                let env = env.incr_indent();
                let indent = env.print_indent();
                let env = env.incr_indent();
                let (env, configs) = pprint_iter(configs.iter(), env, ",\n");
                let env = env.decr_indent();
                let env = env.decr_indent();
                let key = key.iter().map(|s| format!("\"{s}\"")).join(", ");
                let (env, restore_value) = restore_value.into_iter()
                    .fold((env, vec![]), |(env, mut strs), id| {
                        let (env, id) = id.pprint(env);
                        strs.push(format!("\"{id}\""));
                        (env, strs)
                    });
                let restore_value = restore_value.into_iter().join(", ");
                (env, format!(
                    "@triton.autotune(\n{0}configs=[\n{configs}\n{0}],\n{0}key=[{key}],\n{0}restore_value=[{restore_value}]\n)", indent
                ))
            },
        }
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (env, id) = self.id.pprint(env);
        match self.annot_ty {
            AnnotType::Any => (env, format!("{id}")),
            AnnotType::Constexpr => (env, format!("{id}: tl.constexpr")),
        }
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::Import {package, as_str, i: _} => {
                // We add the string of the imported package into the environment, to effectively
                // reserve its name. This avoids naming conflicts that could occur because
                // user-defined code uses a variable with the same name as an imported package.
                if let Some(s) = as_str {
                    let (env, _) = Name::sym_str(s).pprint(env);
                    (env, format!("import {package} as {s}"))
                } else {
                    let (env, _) = Name::sym_str(package).pprint(env);
                    (env, format!("import {package}"))
                }
            },
            Top::KernelFunDef {decorators, id, params, body, i: _} => {
                let (env, decorators) = if decorators.is_empty() {
                    (env, "".to_string())
                } else {
                    let (env, decorators) = pprint_iter(decorators.iter(), env, "\n");
                    (env, format!("{decorators}\n"))
                };
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{decorators}@triton.jit\ndef {id}({params}):\n{body}"))
            },
            Top::FunDef {id, params, body, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("def {id}({params}):\n{body}"))
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

    use std::collections::BTreeMap;

    #[test]
    fn print_variable() {
        assert_eq!(var("x", None).pprint_default(), "x");
    }

    #[test]
    fn print_program_id() {
        let e = Expr::ProgramId {dim: Dim::Y, ty: Type::Void, i: i()};
        assert_eq!(e.pprint_default(), "tl.program_id(1)");
    }

    #[test]
    fn print_arange() {
        let e = Expr::Arange {
            lo: Box::new(int(0)),
            hi: Box::new(int(16)),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.arange(0, 16)");
    }

    #[test]
    fn print_load() {
        let e = Expr::Load {
            ptr: Box::new(var("x", None)),
            mask: None,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.load(x)");
    }

    #[test]
    fn print_load_with_mask() {
        let e = Expr::Load {
            ptr: Box::new(var("x", None)),
            mask: Some(Box::new(var("y", None))),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.load(x, mask=y)");
    }

    #[test]
    fn print_full() {
        let e = Expr::Full {
            shape: Box::new(int(32)),
            value: Box::new(int(1)),
            elem_sz: ElemSize::I32,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.full((32,), 1, tl.int32)");
    }

    #[test]
    fn print_full_singleton() {
        let e = Expr::Full {
            shape: Box::new(int(1)),
            value: Box::new(float(1.0)),
            elem_sz: ElemSize::F32,
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "tl.full((1,), 1.0, tl.float32)");
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
        assert_eq!(e.pprint_default(), "tl.where(x, y, z)");
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
        assert_eq!(s.pprint_default(), "tl.debug_barrier()")
    }

    #[test]
    fn print_store() {
        let s = Stmt::Store {
            ptr: var("x", None),
            value: var("y", None),
            mask: None,
            i: i()
        };
        assert_eq!(s.pprint_default(), "tl.store(x, y)");
    }

    #[test]
    fn print_store_with_mask() {
        let s = Stmt::Store {
            ptr: var("x", None),
            value: var("y", None),
            mask: Some(var("z", None)),
            i: i()
        };
        assert_eq!(s.pprint_default(), "tl.store(x, y, mask=z)");
    }

    #[test]
    fn print_kernel_launch() {
        let block_dims = Dim3 {x: 10, y: 5, z: 1024};
        let s = Stmt::KernelLaunch {
            id: Name::sym_str("f"),
            block_dims,
            args: vec![],
            nwarps: 2,
            i: i()
        };
        assert_eq!(s.pprint_default(), "f[lambda _: (10, 5, 1024)]()");
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
    fn print_kernel_fun_def() {
        let t = Top::KernelFunDef {
            decorators: vec![],
            id: Name::sym_str("f"),
            params: vec![
                Param {
                    id: Name::sym_str("x"),
                    ty: Type::Void,
                    annot_ty: AnnotType::Any,
                    i: i()
                },
                Param {
                    id: Name::sym_str("y"),
                    ty: Type::Void,
                    annot_ty: AnnotType::Any,
                    i: i()
                },
            ],
            body: vec![
                Stmt::Assign {dst: Name::sym_str("w"), expr: var("k", None), i: i()}
            ],
            i: i()
        };
        assert_eq!(t.pprint_default(), "@triton.jit\ndef f(x, y):\n  w = k");
    }

    #[test]
    fn print_autotune_decorator() {
        let bsize = Name::sym_str("BLOCK_SIZE");
        let mut m1 = BTreeMap::new();
        m1.insert(bsize.clone(), int(128));
        let mut m2 = BTreeMap::new();
        m2.insert(bsize, int(512));
        let decorator = Decorator::Autotune {
            configs: vec![
                AutotuneConfig { mapping: m1, warp_count: 4 },
                AutotuneConfig { mapping: m2, warp_count: 8 },
            ],
            key: vec!["a".to_string(), "b".to_string()],
            restore_value: vec![Name::new("a".to_string())],
        };
        assert_eq!(
            decorator.pprint_default(),
            concat!(
                "@triton.autotune(\n",
                "  configs=[\n",
                "    triton.Config({\"BLOCK_SIZE\": 128}, num_warps=4),\n",
                "    triton.Config({\"BLOCK_SIZE\": 512}, num_warps=8)\n",
                "  ],\n",
                "  key=[\"a\", \"b\"],\n",
                "  restore_value=[\"a\"]\n",
                ")"
            )
        )
    }

    #[test]
    fn print_escaped_import_name() {
        let import_triton = Top::Import {
            package: "triton".to_string(),
            as_str: None,
            i: i()
        };
        let import_tl = Top::Import {
            package: "triton.language".to_string(),
            as_str: Some("tl".to_string()),
            i: i()
        };
        let env = PrettyPrintEnv::default();
        let (env, _) = import_triton.pprint(env);
        let (env, _) = import_tl.pprint(env);
        let e = Expr::Var {
            id: Name::sym_str("triton"),
            ty: Type::Void,
            i: i()
        };
        let (env, s) = e.pprint(env);
        assert!(s.starts_with("triton") && s != "triton");
        let e = Expr::Var {
            id: Name::sym_str("tl"),
            ty: Type::Void,
            i: i()
        };
        let (_, s) = e.pprint(env);
        assert!(s.starts_with("tl") && s != "tl");
    }
}
