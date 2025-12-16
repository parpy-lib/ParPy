use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::pprint::*;

use itertools::Itertools;

impl PrettyPrint for MemSpace {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            MemSpace::Host => "",
            MemSpace::Device => "device",
            MemSpace::Threadgroup => "threadgroup",
        };
        (env, s.to_string())
    }
}

fn memcopy_kind(src: &MemSpace, dst: &MemSpace) -> usize {
    match (src, dst) {
        (MemSpace::Host, MemSpace::Host) => 0,
        (MemSpace::Host, MemSpace::Device) => 1,
        (MemSpace::Device, MemSpace::Host) => 2,
        (MemSpace::Device, MemSpace::Device) => 3,
        _ => 4
    }
}

fn pprint_type(decl: String, ty: &Type, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
    let join_space = |fst: String, snd: String, env: PrettyPrintEnv| {
        if snd.is_empty() {
            (env, fst)
        } else {
            (env, format!("{fst} {snd}"))
        }
    };
    match ty {
        Type::Void => join_space("void".to_string(), decl, env),
        Type::Scalar {sz} => {
            let (env, sz) = sz.pprint(env);
            join_space(sz, decl, env)
        },
        Type::Pointer {ty, mem} => {
            let (env, mem) = mem.pprint(env);
            let decl = match ty.as_ref() {
                Type::Function {..} => format!("(*{decl})"),
                _ => format!("*{decl}")
            };
            let (env, s) = pprint_type(decl, ty, env);
            if mem.is_empty() {
                (env, s)
            } else {
                (env, format!("{mem} {s}"))
            }
        },
        Type::Function {result, args} => {
            let (env, args) = args.iter()
                .fold((env, vec![]), |(env, mut acc), ty| {
                    let (env, ty) = pprint_type("".to_string(), ty, env);
                    acc.push(ty);
                    (env, acc)
                });
            let args = args.into_iter().join(", ");
            pprint_type(format!("{decl}({args})"), result, env)
        },
        Type::MTLBufferPtr => join_space("metal_buffer*".to_string(), decl, env),
        Type::MTLFunction => join_space("MTL::Function".to_string(), decl, env),
        Type::MTLLibrary => join_space("MTL::Library".to_string(), decl, env),
        Type::Uint3 => join_space("uint3".to_string(), decl, env),
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_type("".to_string(), self, env)
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
            UnOp::Sub | UnOp::Not | UnOp::BitNeg | UnOp::Addressof => false,
        }
    }

    fn print_unop(op: &UnOp, _argty: &Type) -> Option<String> {
        let s = match op {
            UnOp::Sub => "-",
            UnOp::Not => "!",
            UnOp::BitNeg => "~",
            UnOp::Addressof => "&",
        };
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
            BinOp::Pow | BinOp::Max | BinOp::Min => false,
            _ => true
        }
    }

    fn print_binop(op: &BinOp, _argty: &Type, _ty: &Type) -> Option<String> {
        let s = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::FloorDiv | BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Pow => "metal::pow",
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
            BinOp::Max => "metal::max",
            BinOp::Min => "metal::min",
        };
        Some(s.to_string())
    }

    fn associativity(_op: &BinOp) -> Assoc {
        Assoc::Left
    }
}

fn print_simd_op(op: &SimdOp) -> String {
    let s = match op {
        SimdOp::Sum => "simd_sum",
        SimdOp::Prod => "simd_product",
        SimdOp::Max => "simd_max",
        SimdOp::Min => "simd_min",
    };
    format!("metal::{s}")
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::Bool {v, ..} => (env, v.to_string()),
            Expr::Int {v, ..} => (env, v.to_string()),
            Expr::Float {v, ty, ..} => {
                let s = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "HUGE_VALH",
                    Some(ElemSize::F32) => "HUGE_VALF",
                    _ => panic!("Invalid type of floating-point literal")
                };
                print_float(env, v, s)
            },
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
            Expr::Assign {lhs, rhs, ..} => {
                let (env, lhs) = lhs.pprint(env);
                let (env, rhs) = rhs.pprint(env);
                (env, format!("({lhs} = {rhs})"))
            },
            Expr::Ternary {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("({cond} ? {thn} : {els})"))
            },
            Expr::ArrayAccess {target, idx, ..} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::HostArrayAccess {target, idx, ty, ..} => {
                let (env, ty) = ty.pprint(env);
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("(({ty}*){target}->buf->contents())[{idx}]"))
            },
            Expr::Call {id, args, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::Convert {e, ty} => {
                let (env, e_str) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                let s = if e.is_leaf_node() {
                    format!("({ty}){e_str}")
                } else {
                    format!("({ty})({e_str})")
                };
                (env, s)
            },
            Expr::KernelLaunch {id, blocks, threads, smem, args, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let Dim3 {x: bx, y: by, z: bz} = blocks;
                let Dim3 {x: tx, y: ty, z: tz} = threads;
                (env, format!("parpy_metal::launch_kernel({id}, {{{args}}}, \
                               {bx}, {by}, {bz}, {tx}, {ty}, {tz}, {smem})"))
            },
            Expr::AllocDevice {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, ty) = elem_ty.pprint(env);
                (env, format!("parpy_metal::alloc(&{id}, {sz} * sizeof({ty}))"))
            },
            Expr::Projection {e, label, ..} => {
                let (env, e) = e.pprint(env);
                (env, format!("{e}.{label}"))
            },
            Expr::SimdOp {op, arg, ..} => {
                let (env, arg) = arg.pprint(env);
                let fun_str = print_simd_op(op);
                (env, format!("{fun_str}({arg})"))
            },
            // These nodes should have been replaced by named references, to avoid the risk of
            // users defining variables with the same name.
            Expr::ThreadIdx {..} => panic!("Thread index should have been eliminated"),
            Expr::BlockIdx {..} => panic!("Block index should have been eliminated"),
            Expr::GetFun {lib, fun_id, ..} => {
                let (env, lib) = lib.pprint(env);
                let (env, fun_id) = fun_id.pprint(env);
                (env, format!("parpy_metal::get_fun({lib}, \"{fun_id}\")"))
            },
            Expr::LoadLibrary {tops, ..} => {
                // The library code is included as a string literal. We escape the end of each line
                // with a newline to get proper errors of the generated Metal code. We also add a
                // backslash as required for multi-line string literals.
                if tops.is_empty() {
                    (env, format!("NULL"))
                } else {
                    let (env, tops) = pprint_iter(tops.iter(), env, "\n");
                    let tops = tops.lines()
                        .map(|l| format!("{l}\\n\\"))
                        .join("\n");
                    (env, format!("parpy_metal::load_library(\"\\\n{tops}\n\")"))
                }
            },
        }
    }
}

impl PrettyPrintCond<Expr> for Stmt {
    fn extract_if<'a>(&'a self) -> Option<(&'a Expr, &'a Vec<Stmt>, &'a Vec<Stmt>)> {
        if let Stmt::If {cond, thn, els} = self {
            Some((cond, thn, els))
        } else {
            None
        }
    }

    fn extract_elseif<'a>(&'a self) -> Option<(&'a Expr, &'a Vec<Stmt>, &'a Vec<Stmt>)> {
        if let Stmt::If {els: outer_els, ..} = self {
            if let [Stmt::If {cond, thn, els}] = &outer_els[..] {
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
            Stmt::Definition {ty, id, expr} => {
                let (env, id) = id.pprint(env);
                let (env, s) = pprint_type(id, ty, env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{s} = {expr};"))
            },
            Stmt::For {var_ty, var, init, cond, incr, body, unroll} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let unroll_str = if *unroll {
                    format!("#pragma unroll\n{indent}")
                } else {
                    "".to_string()
                };
                let s = format!(
                    "{0}{unroll_str}\
                     for ({var_ty} {1} = {init}; {cond}; {1} = {incr}) \
                     {{\n{body}\n{0}}}",
                    indent, var
                );
                (env, s)
            },
            Stmt::If {..} => self.print_cond(env),
            Stmt::While {cond, body} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!("{0}while ({1}) {{\n{2}\n{0}}}", indent, cond, body);
                (env, s)
            },
            Stmt::Return {value} => {
                let (env, value) = value.pprint(env);
                (env, format!("{indent}return {value};"))
            },
            Stmt::Expr {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}{e};"))
            },
            Stmt::ThreadgroupBarrier => {
                (env, format!("{indent}metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"))
            },
            Stmt::SubmitWork => {
                (env, format!("{indent}parpy_metal::submit_work();"))
            },
            Stmt::AllocThreadgroup {elem_ty, id, sz} => {
                let (env, ty) = elem_ty.pprint(env);
                let (env, id) = id.pprint(env);
                (env, format!("{indent}threadgroup {ty} {id}[{sz}];"))
            },
            Stmt::CopyMemory {elem_ty, src, src_mem, dst, dst_mem, sz} => {
                let (env, ty) = elem_ty.pprint(env);
                let (env, src) = src.pprint(env);
                let (env, dst) = dst.pprint(env);
                let k = memcopy_kind(&src_mem, &dst_mem);
                (env, format!("{indent}parpy_metal::copy((void*){dst}, \
                               (void*){src}, {sz} * sizeof({ty}), {k});"))
            },
            Stmt::FreeDevice {id} => {
                let (env, id) = id.pprint(env);
                (env, format!("{indent}parpy_metal::free({id});"))
            },
            Stmt::CheckError {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}parpy_metal_check_error({e});"))
            },
        }
    }
}

impl PrettyPrint for ParamAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            ParamAttribute::Buffer {idx} => format!("buffer({idx})"),
            ParamAttribute::ThreadIndex => format!("thread_position_in_threadgroup"),
            ParamAttribute::BlockIndex => format!("threadgroup_position_in_grid"),
        };
        (env, s)
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        let Param {id, ty, attr} = self;
        let (env, id) = id.pprint(env);
        let (env, s) = pprint_type(id, ty, env);
        if let Some(a) = attr {
            let (env, a) = a.pprint(env);
            (env, format!("{indent}{s} [[{a}]]"))
        } else {
            (env, format!("{indent}{s}"))
        }
    }
}

impl PrettyPrint for FunAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            FunAttribute::LaunchBounds {threads} => {
                (env, format!("[[max_total_threads_per_threadgroup({threads})]]"))
            },
            FunAttribute::ExternC => {
                (env, format!("extern \"C\""))
            },
            FunAttribute::SharedMemory {..} => {
                (env, format!(""))
            },
        }
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::Include {header} => (env, format!("#include {header}")),
            Top::ExtDecl {ret_ty: _, id, ext_id, params} => {
                let param_ids = params.iter()
                    .map(|Param {id, ..}| id.get_str())
                    .join(", ");
                (env, format!("#define {id}({0}) {ext_id}({0})", param_ids))
            },
            Top::VarDef {ty, id, init} => {
                let (env, id) = id.pprint(env);
                let (env, s) = pprint_type(id, ty, env);
                let (env, init) = if let Some(e) = init {
                    let (env, e) = e.pprint(env);
                    (env, format!(" = {e}"))
                } else {
                    (env, "".to_string())
                };
                (env, format!("{s}{init};"))
            },
            Top::FunDef {attrs, is_kernel, ret_ty, id, params, body} => {
                let (env, attrs) = pprint_iter(attrs.iter(), env, "\n");
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let (env, params) = pprint_iter(params.iter(), env, ",\n");
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let kernel_attr = if *is_kernel { "kernel " } else { "" }.to_string();
                (env, format!("{attrs}\n{kernel_attr}{ret_ty} {id}(\n{params}\n) {{\n{body}\n}}"))
            },
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        // Perform the pretty-printing in reverse order, to ensure that the name of the main
        // function is never escaped over called functions.
        let (env, strs) = self.tops.iter().rev()
            .fold((env, vec![]), |acc, t| {
                let (env, mut strs) = acc;
                let (env, s) = t.pprint(env);
                strs.push(s);
                (env, strs)
            });
        (env, strs.into_iter().rev().join("\n"))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::metal::ast_builder::*;
    use crate::utils::info::Info;

    use strum::IntoEnumIterator;

    #[test]
    fn print_mem_space_host() {
        assert_eq!(MemSpace::Host.pprint_default(), "");
    }

    #[test]
    fn print_mem_space_device() {
        assert_eq!(MemSpace::Device.pprint_default(), "device");
    }

    #[test]
    fn memcopy_host_to_host() {
        assert_eq!(memcopy_kind(&MemSpace::Host, &MemSpace::Host), 0);
    }

    #[test]
    fn memcopy_host_to_device() {
        assert_eq!(memcopy_kind(&MemSpace::Host, &MemSpace::Device), 1);
    }

    #[test]
    fn memcopy_device_to_host() {
        assert_eq!(memcopy_kind(&MemSpace::Device, &MemSpace::Host), 2);
    }

    #[test]
    fn memcopy_device_to_device() {
        assert_eq!(memcopy_kind(&MemSpace::Device, &MemSpace::Device), 3);
    }

    #[test]
    fn print_host_pointer() {
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Host
        };
        assert_eq!(ty.pprint_default(), "float *");
    }

    #[test]
    fn print_named_host_pointer() {
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Host
        };
        let (_, s) = pprint_type("x".to_string(), &ty, PrettyPrintEnv::default());
        assert_eq!(s, "float *x");
    }

    #[test]
    fn print_device_pointer() {
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Device,
        };
        assert_eq!(ty.pprint_default(), "device float *");
    }

    #[test]
    fn extract_unop_some() {
        let e = unop(UnOp::Sub, int(1, ElemSize::I64));
        assert_eq!(e.extract_unop(), Some((&UnOp::Sub, &int(1, ElemSize::I64))));
    }

    #[test]
    fn extract_unop_none() {
        let e = int(1, ElemSize::I64);
        assert_eq!(e.extract_unop(), None);
    }

    #[test]
    fn sub_is_not_function() {
        assert!(!Expr::is_function(&UnOp::Sub));
    }

    #[test]
    fn extract_binop_some() {
        let lhs = int(0, ElemSize::I64);
        let rhs = int(1, ElemSize::I64);
        let ty = scalar(ElemSize::I64);
        let e = binop(lhs.clone(), BinOp::Add, rhs.clone(), ty.clone());
        assert_eq!(e.extract_binop(), Some((&lhs, &BinOp::Add, &rhs, &ty)));
    }

    #[test]
    fn extract_binop_none() {
        let e = int(1, ElemSize::I64);
        assert_eq!(e.extract_binop(), None);
    }

    #[test]
    fn add_is_infix() {
        assert!(Expr::is_infix(&BinOp::Add, &scalar(ElemSize::I64)));
    }

    #[test]
    fn pow_is_not_infix() {
        assert!(!Expr::is_infix(&BinOp::Pow, &scalar(ElemSize::I64)));
    }

    #[test]
    fn print_all_unary_operators() {
        let ty = scalar(ElemSize::F32);
        for op in UnOp::iter() {
            assert!(Expr::print_unop(&op, &ty).is_some());
        }
    }

    #[test]
    fn print_all_binary_operators() {
        let ty = scalar(ElemSize::F32);
        for op in BinOp::iter() {
            assert!(Expr::print_binop(&op, &ty, &ty).is_some());
        }
    }

    #[test]
    fn print_inf_f16() {
        assert_eq!(float(f64::INFINITY, ElemSize::F16).pprint_default(), "HUGE_VALH");
    }

    #[test]
    fn print_inf_f32() {
        assert_eq!(float(f64::INFINITY, ElemSize::F32).pprint_default(), "HUGE_VALF");
    }

    #[test]
    #[should_panic]
    fn print_inf_f64() {
        assert_eq!(float(f64::INFINITY, ElemSize::F64).pprint_default(), "HUGE_VAL");
    }

    fn simd_op(op: SimdOp) -> Expr {
        Expr::SimdOp {
            op,
            arg: Box::new(var("x", scalar(ElemSize::F32))),
            ty: scalar(ElemSize::F32),
            i: Info::default()
        }
    }

    #[test]
    fn print_simd_op_add() {
        assert_eq!(simd_op(SimdOp::Sum).pprint_default(), "metal::simd_sum(x)");
    }

    #[test]
    fn print_simd_op_mul() {
        assert_eq!(simd_op(SimdOp::Prod).pprint_default(), "metal::simd_product(x)");
    }

    #[test]
    fn print_simd_op_max() {
        assert_eq!(simd_op(SimdOp::Max).pprint_default(), "metal::simd_max(x)");
    }

    #[test]
    fn print_simd_op_min() {
        assert_eq!(simd_op(SimdOp::Min).pprint_default(), "metal::simd_min(x)");
    }

    #[test]
    #[should_panic]
    fn print_thread_index_fails() {
        let e = Expr::ThreadIdx {dim: Dim::X, ty: Type::Uint3, i: Info::default()};
        e.pprint_default();
    }

    #[test]
    #[should_panic]
    fn print_block_index_fails() {
        let e = Expr::BlockIdx {dim: Dim::X, ty: Type::Uint3, i: Info::default()};
        e.pprint_default();
    }

    #[test]
    fn print_scalar_param() {
        let p = Param {id: id("x"), ty: scalar(ElemSize::F16), attr: None};
        assert_eq!(p.pprint_default(), "half x");
    }

    #[test]
    fn print_buffer_param() {
        let p = Param {id: id("x"), ty: Type::MTLBufferPtr, attr: Some(ParamAttribute::Buffer {idx: 0})};
        assert_eq!(p.pprint_default(), "metal_buffer* x [[buffer(0)]]");
    }

    #[test]
    fn print_thread_idx_param() {
        let p = Param {id: id("x"), ty: Type::Uint3, attr: Some(ParamAttribute::ThreadIndex)};
        assert_eq!(p.pprint_default(), "uint3 x [[thread_position_in_threadgroup]]");
    }

    #[test]
    fn print_block_idx_param() {
        let p = Param {id: id("x"), ty: Type::Uint3, attr: Some(ParamAttribute::BlockIndex)};
        assert_eq!(p.pprint_default(), "uint3 x [[threadgroup_position_in_grid]]");
    }

    #[test]
    fn print_array_access() {
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Device
        };
        let e = Expr::ArrayAccess {
            target: Box::new(var("x", ty)),
            idx: Box::new(int(1, ElemSize::I64)),
            ty: scalar(ElemSize::F32),
            i: Info::default(),
        };
        assert_eq!(e.pprint_default(), "x[1]");
    }

    #[test]
    fn print_host_array_access() {
        let ty = Type::Pointer {
            ty: Box::new(scalar(ElemSize::F32)),
            mem: MemSpace::Device
        };
        let e = Expr::HostArrayAccess {
            target: Box::new(var("x", ty)),
            idx: Box::new(int(1, ElemSize::I64)),
            ty: scalar(ElemSize::F32),
            i: Info::default(),
        };
        assert_eq!(e.pprint_default(), "((float*)x->buf->contents())[1]");
    }

    #[test]
    fn print_var_def_empty_init() {
        let t = Top::VarDef {ty: scalar(ElemSize::F32), id: id("x"), init: None};
        assert_eq!(t.pprint_default(), "float x;");
    }

    #[test]
    fn print_var_def_some_init() {
        let t = Top::VarDef {
            ty: scalar(ElemSize::F32), id: id("x"),
            init: Some(float(2.5, ElemSize::F32)),
        };
        assert_eq!(t.pprint_default(), "float x = 2.5;");
    }
}
