use super::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;

use std::borrow::Borrow;

impl PrettyPrint for MemSpace {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            MemSpace::Host => "",
            MemSpace::Device => "device",
        };
        (env, s.to_string())
    }
}

fn memcopy_kind(src: &MemSpace, dst: &MemSpace) -> usize {
    match (src, dst) {
        (MemSpace::Host, MemSpace::Host) => 0,
        (MemSpace::Host, MemSpace::Device) => 1,
        (MemSpace::Device, MemSpace::Host) => 2,
        (MemSpace::Device, MemSpace::Device) => 3
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Void => (env, "void".to_string()),
            Type::Boolean => (env, "bool".to_string()),
            Type::Scalar {sz} => sz.pprint(env),
            Type::Pointer {ty, mem} => {
                let (env, ty) = ty.pprint(env);
                let (env, mem) = mem.pprint(env);
                (env, format!("{mem} {ty}*"))
            },
            Type::Buffer => (env, "MTL::Buffer*".to_string()),
            Type::Uint3 => (env, "uint3".to_string()),
        }
    }
}

pub fn print_unop(op: &UnOp) -> String {
    let s = match op {
        UnOp::Sub => "-",
        UnOp::Not => "!",
        UnOp::BitNeg => "~",
        UnOp::Addressof => "&",
        UnOp::Exp => "metal::exp",
        UnOp::Log => "metal::log",
        UnOp::Cos => "metal::cos",
        UnOp::Sin => "metal::sin",
        UnOp::Sqrt => "metal::sqrt",
        UnOp::Tanh => "metal::tanh",
        UnOp::Abs => "metal::abs",
    };
    s.to_string()
}

pub fn print_binop(op: &BinOp) -> String {
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
        BinOp::Atan2 => "metal::atan2",
    };
    s.to_string()
}

fn try_get_binop(e: &Box<Expr>) -> Option<BinOp> {
    match e.borrow() {
        Expr::BinOp {op, ..} => Some(op.clone()),
        _ => None
    }
}

fn is_infix(op: &BinOp) -> bool {
    match op {
        BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => false,
        _ => true
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::Bool {v, ..} => (env, v.to_string()),
            Expr::Int {v, ..} => (env, v.to_string()),
            Expr::Float {v, ty, ..} => {
                let s = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F32) => "HUGE_VALF",
                    Some(ElemSize::F64) => "HUGE_VAL",
                    _ => panic!("Invalid type of floating-point literal")
                };
                print_float(env, v, s)
            },
            Expr::UnOp {op, arg, ..} => {
                let op_str = print_unop(&op);
                let (env, arg_str) = arg.pprint(env);
                (env, format!("{op_str}({arg_str})"))
            },
            Expr::BinOp {lhs, op, rhs, ..} => {
                let (env, lhs_str) = lhs.pprint(env);
                let op_str = print_binop(&op);
                let (env, rhs_str) = rhs.pprint(env);
                if is_infix(&op) {
                    let lhs_op = try_get_binop(lhs);
                    let rhs_op = try_get_binop(rhs);
                    let lhs_str = parenthesize_if_lower_precedence(lhs_op, &op, lhs_str);
                    let rhs_str = parenthesize_if_lower_or_same_precedence(rhs_op, &op, rhs_str);
                    (env, format!("{lhs_str} {op_str} {rhs_str}"))
                } else {
                    (env, format!("{op_str}({lhs_str}, {rhs_str})"))
                }
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
                (env, format!("(({ty}*){target}->contents())[{idx}]"))
            },
            Expr::Call {id, args, ..} => {
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
            Expr::KernelLaunch {id, blocks, threads, args, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let Dim3 {x: bx, y: by, z: bz} = blocks;
                let Dim3 {x: tx, y: ty, z: tz} = threads;
                (env, format!("prickle_metal::launch_kernel({id}, {{{args}}}, \
                               {bx}, {by}, {bz}, {tx}, {ty}, {tz})"))
            },
            Expr::AllocDevice {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, ty) = elem_ty.pprint(env);
                (env, format!("prickle_metal::alloc(&{id}, {sz} * sizeof({ty}))"))
            },
            Expr::Projection {e, label, ..} => {
                let (env, e) = e.pprint(env);
                (env, format!("{e}.{label}"))
            },
            Expr::SimdOp {op, arg, i, ..} => {
                let (env, arg) = arg.pprint(env);
                let fun_str = match op {
                    BinOp::Add => "metal::simd_sum",
                    BinOp::Mul => "metal::simd_product",
                    BinOp::Max => "metal::simd_max",
                    BinOp::Min => "metal::simd_min",
                    _ => {
                        let op = print_binop(op);
                        let msg = format!("Reduction on unsupported binary operation {op}");
                        panic!("{}", i.error_msg(msg))
                    }
                };
                (env, format!("{fun_str}({arg})"))
            },
            // These nodes should be eliminated and replaced by named references, to avoid the risk
            // of users defining variables with the same name.
            Expr::ThreadIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("<threadIdx.{dim}>"))
            },
            Expr::BlockIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("<blockIdx.{dim}>"))
            },
        }
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Definition {ty, id, expr} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{ty} {id} = {expr};"))
            },
            Stmt::Assign {dst, expr} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{dst} = {expr};"))
            },
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!(
                    "{0}for ({1} {2} = {3}; {4}; {2} = {5}) {{\n{6}\n{0}}}",
                    indent, var_ty, var, init, cond, incr, body
                );
                (env, s)
            },
            Stmt::If {cond, thn, els} => {
                let f = |els: Vec<Stmt>| {
                    match &els[..] {
                        [Stmt::If {cond, thn, els}] if !thn.is_empty() => {
                            Some((cond.clone(), thn.clone(), els.clone()))
                        },
                        _ => None
                    }
                };
                print_if_condition(env, cond, thn, els, f)
            },
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
            Stmt::ThreadgroupBarrier => {
                (env, format!("{indent}metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);"))
            },
            Stmt::SubmitWork => {
                (env, format!("{indent}prickle_metal::submit_work();"))
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
                (env, format!("{indent}prickle_metal::copy((void*){dst}, \
                               (void*){src}, {sz} * sizeof({ty}), {k});"))
            },
            Stmt::FreeDevice {id} => {
                let (env, id) = id.pprint(env);
                (env, format!("{indent}prickle_metal::free({id});"))
            },
            Stmt::CheckError {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}prickle_check_error({e});"))
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
        let (env, ty_str) = ty.pprint(env);
        if let Some(a) = attr {
            let (env, a) = a.pprint(env);
            (env, format!("{indent}{ty_str} {id} [[{a}]]"))
        } else {
            (env, format!("{indent}{ty_str} {id}"))
        }
    }
}

impl PrettyPrint for KernelAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            KernelAttribute::LaunchBounds {threads} => {
                (env, format!("[[max_total_threads_per_threadgroup({threads})]]"))
            },
        }
    }
}

fn pprint_metal_top(env: PrettyPrintEnv, t: &Top) -> (PrettyPrintEnv, String) {
    match t {
        Top::KernelDef {attrs, id, params, body} => {
            let (env, id) = id.pprint(env);
            let (env, attrs) = pprint_iter(attrs.iter(), env, "\n");
            let env = env.incr_indent();
            let (env, params) = pprint_iter(params.iter(), env, ",\n");
            let (env, body) = pprint_iter(body.iter(), env, "\n");
            let env = env.decr_indent();
            (env, format!("{attrs}\nkernel void {id}(\n{params}\n) {{\n{body}\n}}"))
        },
        Top::FunDef {ret_ty, id, params, body} => {
            let (env, ret_ty) = ret_ty.pprint(env);
            let (env, id) = id.pprint(env);
            let env = env.incr_indent();
            let (env, params) = pprint_iter(params.iter(), env, ",\n");
            let (env, body) = pprint_iter(body.iter(), env, "\n");
            let env = env.decr_indent();
            (env, format!("{ret_ty} {id}(\n{params}\n) {{\n{body}\n}}"))
        },
    }
}

fn pprint_host_top(env: PrettyPrintEnv, t: &Top) -> (PrettyPrintEnv, String) {
    match t {
        Top::KernelDef {id, ..} => panic!("Found definition of kernel {id} in host code"),
        Top::FunDef {ret_ty, id, params, body} => {
            let (env, ret_ty) = ret_ty.pprint(env);
            let (env, id) = id.pprint(env);
            let (env, params) = pprint_iter(params.iter(), env, ",\n");
            let env = env.incr_indent();
            let (env, body) = pprint_iter(body.iter(), env, "\n");
            let env = env.decr_indent();
            (env, format!("extern \"C\" {ret_ty} {id}(\n{params}\n) {{\n{body}\n}}"))
        },
    }
}

fn pprint_tops(
    env: PrettyPrintEnv,
    tops: &Vec<Top>,
    pptop: impl Fn(PrettyPrintEnv, &Top) -> (PrettyPrintEnv, String)
) -> (PrettyPrintEnv, String) {
    let (env, strs) = tops.iter()
        .fold((env, vec![]), |(env, mut strs), t| {
            let (env, s) = pptop(env, t);
            strs.push(s);
            (env, strs)
        });
    (env, strs.into_iter().join("\n"))
}

fn pprint_metal_tops(env: PrettyPrintEnv, tops: &Vec<Top>) -> (PrettyPrintEnv, String) {
    pprint_tops(env, tops, pprint_metal_top)
}

fn pprint_host_tops(env: PrettyPrintEnv, tops: &Vec<Top>) -> (PrettyPrintEnv, String) {
    pprint_tops(env, tops, pprint_host_top)
}

fn generate_metal_kernel_function_definitions(
    env: PrettyPrintEnv, metal_tops: &Vec<Top>
) -> (PrettyPrintEnv, String) {
    let (env, tops) = metal_tops.iter()
        .fold((env, vec![]), |(env, mut strs), t| {
            let env = if let Top::KernelDef {id, ..} = t {
                let (env, id) = id.pprint(env);
                let s = format!("MTL::Function* {id} = prickle_metal::get_fun(lib, \"{id}\");");
                strs.push(s);
                env
            } else {
                env
            };
            (env, strs)
        });
    (env, tops.into_iter().join("\n"))
}

// Adds a backslash at the end of each line of the given string. This is required for the C++
// compiler to consider a multi-line string valid.
fn add_backslash_at_end_of_line(s: String) -> String {
    s.lines()
        .map(|l| format!("{l}\\n\\"))
        .join("\n")
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Ast {includes, metal_tops, host_tops} = self;
        let includes = includes.iter()
            .map(|s| format!("#include {s}"))
            .join("\n");
        let (env, metal_tops_str) = pprint_metal_tops(env, metal_tops);
        let metal_tops_str = add_backslash_at_end_of_line(metal_tops_str);
        let (env, host_tops_str) = pprint_host_tops(env, host_tops);
        let (env, metal_fun_defs) = generate_metal_kernel_function_definitions(env, metal_tops);
        (env, format!("\
            {includes}\n\
            MTL::Library* lib = prickle_metal::load_library(\"\\\n{metal_tops_str}\n\");\n\
            {metal_fun_defs}\n\
            {host_tops_str}"))
    }
}
