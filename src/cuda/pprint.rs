use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::pprint::*;

use itertools::Itertools;

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Void => (env, "void".to_string()),
            Type::Scalar {sz} => sz.pprint(env),
            Type::Pointer {ty} => {
                let (env, ty) = ty.pprint(env);
                (env, format!("{ty}*"))
            },
            Type::Struct {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("{id}"))
            },
            Type::Error => (env, format!("cudaError_t")),
            Type::Stream => (env, format!("cudaStream_t")),
            Type::Graph => (env, format!("cudaGraph_t")),
            Type::GraphExec => (env, format!("cudaGraphExec_t")),
            Type::GraphExecUpdateResultInfo => (env, format!("cudaGraphExecUpdateResultInfo"))
        }
    }
}

impl PrettyPrint for FuncAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            FuncAttribute::NonPortableClusterSizeAllowed =>
                format!("cudaFuncAttributeNonPortableClusterSizeAllowed"),
        };
        (env, s)
    }
}

impl PrettyPrint for Error {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Error::Success => format!("cudaSuccess"),
        };
        (env, s)
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
            UnOp::Exp | UnOp::Log | UnOp::Cos | UnOp::Sin | UnOp::Sqrt |
            UnOp::Tanh | UnOp::Abs => true,
        }
    }

    fn print_unop(op: &UnOp, argty: &Type) -> Option<String> {
        let o = match op {
            UnOp::Sub => Some("-"),
            UnOp::Not => Some("!"),
            UnOp::BitNeg => Some("~"),
            UnOp::Addressof => Some("&"),
            UnOp::Exp => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hexp"),
                Some(ElemSize::F32) => Some("__expf"),
                Some(ElemSize::F64) => Some("exp"),
                _ => None
            },
            UnOp::Log => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hlog"),
                Some(ElemSize::F32) => Some("__logf"),
                Some(ElemSize::F64) => Some("log"),
                _ => None
            },
            UnOp::Cos => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hcos"),
                Some(ElemSize::F32) => Some("__cosf"),
                Some(ElemSize::F64) => Some("cos"),
                _ => None
            },
            UnOp::Sin => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hsin"),
                Some(ElemSize::F32) => Some("__sinf"),
                Some(ElemSize::F64) => Some("sin"),
                _ => None
            },
            UnOp::Sqrt => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hsqrt"),
                Some(ElemSize::F32) => Some("sqrtf"),
                Some(ElemSize::F64) => Some("sqrt"),
                _ => None
            },
            UnOp::Tanh => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("htanh"),
                Some(ElemSize::F32) => Some("tanhf"),
                Some(ElemSize::F64) => Some("tanh"),
                _ => None
            },
            UnOp::Abs => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__habs"),
                Some(ElemSize::F32) => Some("fabsf"),
                Some(ElemSize::F64) => Some("fabs"),
                Some(_) => Some("abs"),
                _ => None
            },
        };
        o.map(|s| s.to_string())
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

    fn is_infix(op: &BinOp, ty: &Type) -> bool {
        let is_f16 = match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => true,
            _ => false
        };
        match op {
            BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => false,
            BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt |
            BinOp::Gt if is_f16 => false,
            _ => true
        }
    }

    fn print_binop(op: &BinOp, argty: &Type, ty: &Type) -> Option<String> {
        let o = match op {
            BinOp::Add => Some("+"),
            BinOp::Sub => Some("-"),
            BinOp::Mul => Some("*"),
            BinOp::FloorDiv | BinOp::Div => Some("/"),
            BinOp::Rem => Some("%"),
            BinOp::Pow => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hpow"),
                Some(ElemSize::F32) => Some("__powf"),
                Some(ElemSize::F64) => Some("pow"),
                _ => None
            },
            BinOp::And => Some("&&"),
            BinOp::Or => Some("||"),
            BinOp::BitAnd => Some("&"),
            BinOp::BitOr => Some("|"),
            BinOp::BitXor => Some("^"),
            BinOp::BitShl => Some("<<"),
            BinOp::BitShr => Some(">>"),
            BinOp::Eq => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__heq"),
                _ => Some("==")
            },
            BinOp::Neq => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hne"),
                _ => Some("!=")
            },
            BinOp::Leq => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hle"),
                _ => Some("<=")
            }
            BinOp::Geq => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hge"),
                _ => Some(">=")
            },
            BinOp::Lt => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hlt"),
                _ => Some("<")
            },
            BinOp::Gt => match argty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hgt"),
                _ => Some(">")
            },
            BinOp::Max => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hmax"),
                Some(ElemSize::F32) => Some("fmaxf"),
                Some(ElemSize::F64) => Some("fmax"),
                Some(_) => Some("max"),
                None => None,
            },
            BinOp::Min => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hmin"),
                Some(ElemSize::F32) => Some("fminf"),
                Some(ElemSize::F64) => Some("fmin"),
                Some(_) => Some("min"),
                None => None,
            },
            BinOp::Atan2 => Some("atan2"),
        };
        o.map(|s| s.to_string())
    }

    fn associativity(_op: &BinOp) -> Assoc {
        Assoc::Left
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
                    Some(ElemSize::F16) => "CUDART_INF_FP16",
                    Some(ElemSize::F32) => "HUGE_VALF",
                    Some(ElemSize::F64) => "HUGE_VAL",
                    _ => panic!("Invalid type of floating-point literal")
                };
                print_float(env, v, s)
            },
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
            Expr::Ternary {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("({cond} ? {thn} : {els})"))
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
                let outer_indent = env.print_indent();
                let env = env.incr_indent();
                let indent = env.print_indent();
                let fields = fields.into_iter().join(&format!(",\n{indent}"));
                let env = env.decr_indent();
                (env, format!("{id} {{\n{indent}{fields}\n{outer_indent}}}"))
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
            Expr::ThreadIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("threadIdx.{dim}"))
            },
            Expr::BlockIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("blockIdx.{dim}"))
            },
            Expr::Error {e, ..} => {
                let (env, e) = e.pprint(env);
                (env, format!("{e}"))
            },
            Expr::GetLastError {..} => {
                (env, format!("cudaGetLastError()"))
            },
            Expr::FuncSetAttribute {func, attr, value, ..} => {
                let (env, func) = func.pprint(env);
                let (env, attr) = attr.pprint(env);
                let (env, value) = value.pprint(env);
                (env, format!("cudaFuncSetAttribute({func}, {attr}, {value})"))
            },
            Expr::MallocAsync {id, elem_ty, sz, stream, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                let (env, stream) = stream.pprint(env);
                (env, format!("cudaMallocAsync(&{id}, {sz} * sizeof({elem_ty}), {stream})"))
            },
            Expr::FreeAsync {id, stream, ..} => {
                let (env, id) = id.pprint(env);
                let (env, stream) = stream.pprint(env);
                (env, format!("cudaFreeAsync({id}, {stream})"))
            },
            Expr::StreamCreate {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("cudaStreamCreate(&{id})"))
            },
            Expr::StreamDestroy {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("cudaStreamDestroy({id})"))
            },
            Expr::StreamBeginCapture {stream, ..} => {
                let (env, stream) = stream.pprint(env);
                (env, format!("cudaStreamBeginCapture({stream}, cudaStreamCaptureModeGlobal)"))
            },
            Expr::StreamEndCapture {stream, graph, ..} => {
                let (env, stream) = stream.pprint(env);
                let (env, graph) = graph.pprint(env);
                (env, format!("cudaStreamEndCapture({stream}, &{graph})"))
            },
            Expr::GraphDestroy {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("cudaGraphDestroy({id})"))
            },
            Expr::GraphExecInstantiate {exec_graph, graph, ..} => {
                let (env, exec_graph) = exec_graph.pprint(env);
                let (env, graph) = graph.pprint(env);
                (env, format!("cudaGraphInstantiate(&{exec_graph}, {graph})"))
            },
            Expr::GraphExecDestroy {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("cudaGraphExecDestroy({id})"))
            },
            Expr::GraphExecUpdate {exec_graph, graph, update, ..} => {
                let (env, exec_graph) = exec_graph.pprint(env);
                let (env, graph) = graph.pprint(env);
                let (env, update) = update.pprint(env);
                (env, format!("cudaGraphExecUpdate({exec_graph}, {graph}, &{update})"))
            },
            Expr::GraphExecLaunch {id, ..} => {
                let (env, id) = id.pprint(env);
                let (env, s) = Stream::Default.pprint(env);
                (env, format!("cudaGraphLaunch({id}, {s})"))
            },
        }
    }
}

impl PrettyPrint for Stream {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Stream::Default => (env, "0".to_string()),
            Stream::Id(n) => n.pprint(env),
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
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                match expr {
                    Some(e) => {
                        let (env, e) = e.pprint(env);
                        (env, format!("{indent}{ty} {id} = {e};"))
                    },
                    None => (env, format!("{indent}{ty} {id};"))
                }
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
            Stmt::Synchronize {scope} => {
                let s = match scope {
                    SyncScope::Block => "__syncthreads()",
                    SyncScope::Cluster => "this_cluster().sync()",
                };
                (env, format!("{indent}{s};"))
            },
            Stmt::KernelLaunch {id, blocks, threads, args, stream} => {
                let (env, id) = id.pprint(env);
                let (env, blocks) = blocks.pprint(env);
                let (env, threads) = threads.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let (env, stream) = stream.pprint(env);
                (env, format!("{indent}{id}<<<dim3({blocks}), dim3({threads}), 0, {stream}>>>({args});"))
            },
            Stmt::AllocShared {ty, id, sz} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                (env, format!("{indent}__shared__ {ty} {id}[{sz}];"))
            },
            Stmt::CheckError {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}prickle_cuda_check_error({e});"))
            },
        }
    }
}

impl PrettyPrint for Attribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Attribute::Global => "__global__",
            Attribute::Device => "__device__",
            Attribute::Entry => "extern \"C\"",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Field {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Field {id, ty} = self;
        let (env, ty) = ty.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{ty} {id};"))
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty} = self;
        let (env, id) = id.pprint(env);
        let restrict_str = if let Type::Pointer {..} = &ty {
            " __restrict__"
        } else {
            ""
        };
        let (env, ty) = ty.pprint(env);
        (env, format!("{ty}{restrict_str} {id}"))
    }
}

impl PrettyPrint for KernelAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            KernelAttribute::LaunchBounds {threads} => {
                (env, format!("__launch_bounds__({threads})"))
            },
            KernelAttribute::ClusterDims {dims} => {
                let (env, dims) = dims.pprint(env);
                (env, format!("__cluster_dims__({dims})"))
            },
        }
    }
}

fn pprint_attrs(
    attrs: &Vec<KernelAttribute>,
    env: PrettyPrintEnv
) -> (PrettyPrintEnv, String) {
    if attrs.len() == 0 {
        (env, " ".to_string())
    } else {
        let (env, attrs) = pprint_iter(attrs.iter(), env, "\n");
        (env, format!("\n{attrs}\n"))
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::Include {header} => {
                (env, format!("#include {header}"))
            },
            Top::Namespace {ns, alias} => {
                if let Some(a) = alias {
                    (env, format!("using namespace {ns} = {a};"))
                } else {
                    (env, format!("using namespace {ns};"))
                }
            },
            Top::ExtDecl {ret_ty, id, params} => {
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                (env, format!("extern __device__ {ret_ty} {id}({params});"))
            },
            Top::StructDef {id, fields} => {
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let (env, fields) = pprint_iter(fields.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("struct {id} {{\n{fields}\n}};"))
            },
            Top::VarDef {ty, id, init} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                if let Some(e) = init {
                    let (env, e) = e.pprint(env);
                    (env, format!("{ty} {id} = {e};"))
                } else {
                    (env, format!("{ty} {id};"))
                }
            },
            Top::FunDef {dev_attr, ret_ty, attrs, id, params, body} => {
                let (env, dev_attr) = dev_attr.pprint(env);
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, attrs) = pprint_attrs(attrs, env);
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{dev_attr}\n{ret_ty}{attrs}{id}({params}) {{\n{body}\n}}"))
            },
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_iter(self.iter(), env, "\n")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cuda::ast_builder::*;
    use crate::utils::info::Info;
    use crate::utils::name::Name;
    use crate::utils::pprint;

    use strum::IntoEnumIterator;

    fn uvar(id: &str) -> Expr {
        var(id, Type::Scalar {sz: ElemSize::Bool})
    }

    #[test]
    fn unop_print() {
        for op in UnOp::iter() {
            for sz in ElemSize::iter() {
                match op {
                    UnOp::Sub | UnOp::Not | UnOp::BitNeg | UnOp::Addressof | UnOp::Abs => {
                        assert!(Expr::print_unop(&op, &scalar(sz)).is_some());
                    },
                    _ if sz.is_floating_point() => {
                        assert!(Expr::print_unop(&op, &scalar(sz)).is_some());
                    },
                    _ => {
                        assert!(Expr::print_unop(&op, &scalar(sz)).is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn binop_print() {
        for op in BinOp::iter() {
            for sz in ElemSize::iter() {
                let ty = scalar(sz.clone());
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::FloorDiv |
                    BinOp::Div | BinOp::Rem | BinOp::And | BinOp::Or |
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor |
                    BinOp::BitShl | BinOp::BitShr | BinOp::Eq | BinOp::Neq |
                    BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt | BinOp::Max |
                    BinOp::Min | BinOp::Atan2 => {
                        assert!(Expr::print_binop(&op, &ty, &ty).is_some());
                    },
                    BinOp::Pow if sz.is_floating_point() => {
                        assert!(Expr::print_binop(&op, &ty, &ty).is_some());
                    },
                    _ => {
                        assert!(Expr::print_binop(&op, &ty, &ty).is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn pprint_precedence_same_level_with_paren() {
        let s = add(
            uvar("x"),
            add(uvar("y"), uvar("z"), scalar(ElemSize::I64)),
            scalar(ElemSize::I64)
        ).pprint_default();
        assert_eq!(&s, "x + (y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_omit_paren() {
        let s = add(
            add(uvar("x"), uvar("y"), scalar(ElemSize::I64)),
            uvar("z"),
            scalar(ElemSize::I64)
        ).pprint_default();
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_print_paren() {
        let s = mul(
            add(uvar("x"), uvar("y"), scalar(ElemSize::I64)),
            add(uvar("y"), uvar("z"), scalar(ElemSize::I64)),
            scalar(ElemSize::I64)
        ).pprint_default();
        assert_eq!(&s, "(x + y) * (y + z)");
    }

    #[test]
    fn pprint_precedence_rhs_paren() {
        let s = add(
            uvar("x"),
            add(
                mul(uvar("y"), uvar("y"), scalar(ElemSize::I64)),
                uvar("z"),
                scalar(ElemSize::I64)
            ),
            scalar(ElemSize::I64)
        ).pprint_default();
        assert_eq!(&s, "x + (y * y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_paren() {
        let s = mul(
            uvar("x"),
            rem(uvar("y"), uvar("z"), scalar(ElemSize::I64)),
            scalar(ElemSize::I64)
        ).pprint_default();
        assert_eq!(&s, "x * (y % z)");
    }

    #[test]
    fn pprint_dims() {
        assert_eq!(Dim::X.pprint_default(), "x");
        assert_eq!(Dim::Y.pprint_default(), "y");
        assert_eq!(Dim::Z.pprint_default(), "z");
    }

    #[test]
    fn pprint_thread_idx_x() {
        let s = Expr::ThreadIdx {dim: Dim::X, ty: i64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "threadIdx.x");
    }

    #[test]
    fn pprint_block_idx_y() {
        let s = Expr::BlockIdx {dim: Dim::Y, ty: i64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "blockIdx.y");
    }

    #[test]
    #[should_panic]
    fn pprint_exp_int_type_fails() {
        exp(var("x", scalar(ElemSize::I32)), scalar(ElemSize::I32)).pprint_default();
    }

    #[test]
    #[should_panic]
    fn pprint_exp_invalid_type_fails() {
        exp(uvar("x"), scalar(ElemSize::Bool)).pprint_default();
    }

    #[test]
    fn pprint_log_f64() {
        let ty = scalar(ElemSize::F64);
        let s = log(var("x", ty.clone()), ty).pprint_default();
        assert_eq!(&s, "log(x)");
    }

    #[test]
    fn pprint_max_f32() {
        let ty = scalar(ElemSize::F32);
        let s = max(var("x", ty.clone()), var("y", ty.clone()), ty).pprint_default();
        assert_eq!(&s, "fmaxf(x, y)");
    }

    #[test]
    fn pprint_max_i64() {
        let ty = scalar(ElemSize::I64);
        let s = max(var("x", ty.clone()), var("y", ty.clone()), ty).pprint_default();
        assert_eq!(&s, "max(x, y)");
    }

    #[test]
    fn pprint_struct_literal() {
        let s = Expr::Struct {
            id: Name::sym_str("id"),
            fields: vec![
                ("x".to_string(), int(5, ElemSize::I64)),
                ("y".to_string(), int(25, ElemSize::I64)),
                ("z".to_string(), uvar("q"))
            ],
            ty: Type::Struct {id: Name::sym_str("ty")},
            i: Info::default()
        };
        assert_eq!(&s.pprint_default(), "id {\n  x: 5,\n  y: 25,\n  z: q\n}");
    }

    fn convert(e: Expr, ty: Type) -> Expr {
        Expr::Convert {e: Box::new(e), ty}
    }

    #[test]
    fn pprint_var_conversion() {
        let s = convert(uvar("x"), scalar(ElemSize::F32));
        assert_eq!(&s.pprint_default(), "(float)x");
    }

    #[test]
    fn pprint_literal_conversion() {
        let s = convert(int(5, ElemSize::I64), scalar(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)5");
    }

    #[test]
    fn pprint_add_conversion() {
        let s = convert(add(uvar("x"), uvar("y"), scalar(ElemSize::I32)), scalar(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)(x + y)");
    }

    #[test]
    fn pprint_synchronize_block() {
        let s = Stmt::Synchronize {scope: SyncScope::Block}.pprint_default();
        assert_eq!(&s, "__syncthreads();");
    }

    #[test]
    fn pprint_synchronize_cluster() {
        let s = Stmt::Synchronize {scope: SyncScope::Cluster}.pprint_default();
        assert_eq!(&s, "this_cluster().sync();");
    }

    #[test]
    fn pprint_for_loop() {
        let i = Name::new("i".to_string());
        let ty = scalar(ElemSize::I64);
        let i_var = Expr::Var {id: i.clone(), ty: ty.clone(), i: Info::default()};
        let for_loop = Stmt::For {
            var_ty: ty.clone(),
            var: i,
            init: int(0, ElemSize::I64),
            cond: binop(i_var.clone(), BinOp::Lt, int(10, ElemSize::I64), ty.clone()),
            incr: binop(i_var.clone(), BinOp::Add, int(1, ElemSize::I64), ty),
            body: vec![Stmt::Assign {dst: uvar("x"), expr: uvar("y")}],
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "for (int64_t i = 0; i < 10; i = i + 1) {{\n{indent}x = y;\n}}"
        );
        assert_eq!(for_loop.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond() {
        let cond = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(uvar("x")),
                op: BinOp::Eq,
                rhs: Box::new(uvar("y")),
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: Info::default()
            },
            thn: vec![Stmt::Assign {dst: uvar("x"), expr: uvar("y")}],
            els: vec![Stmt::Assign {dst: uvar("y"), expr: uvar("x")}],
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
                lhs: Box::new(uvar("x")),
                op: BinOp::Eq,
                rhs: Box::new(uvar("y")),
                ty: Type::Scalar {sz: ElemSize::Bool},
                i: Info::default()
            },
            thn: vec![Stmt::Assign {dst: uvar("x"), expr: uvar("y")},],
            els: vec![],
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("if (x == y) {{\n{indent}x = y;\n}}");
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_elseif() {
        let cond = Stmt::If {
            cond: uvar("x"),
            thn: vec![Stmt::Assign {dst: uvar("y"), expr: uvar("z")}],
            els: vec![Stmt::If {
                    cond: uvar("y"),
                    thn: vec![Stmt::Assign {dst: uvar("x"), expr: uvar("z")}],
                    els: vec![Stmt::Assign {dst: uvar("z"), expr: uvar("x")}],
            }],
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "if (x) {{\n{0}y = z;\n}} else if (y) {{\n{0}x = z;\n}} else {{\n{0}z = x;\n}}",
            indent
        );
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_while() {
        let wh = Stmt::While {
            cond: uvar("x"),
            body: vec![Stmt::Assign {dst: uvar("y"), expr: uvar("z")}]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("while (x) {{\n{indent}y = z;\n}}");
        assert_eq!(wh.pprint_default(), expected);
    }

    #[test]
    fn pprint_kernel_launch() {
        let id = "kernel";
        let kernel = Stmt::KernelLaunch {
            id: Name::new(id.to_string()),
            blocks: Dim3::default()
                .with_dim(&Dim::X, 4)
                .with_dim(&Dim::Z, 2),
            threads: Dim3::default()
                .with_dim(&Dim::Y, 6)
                .with_dim(&Dim::Z, 7),
            stream: Stream::Default,
            args: vec![uvar("x"), uvar("y")],
        };
        let expected = format!("{id}<<<dim3(4, 1, 2), dim3(1, 6, 7), 0, 0>>>(x, y);");
        assert_eq!(kernel.pprint_default(), expected);
    }

    #[test]
    fn pprint_struct_def() {
        let def = Top::StructDef {
            id: Name::new("point".to_string()),
            fields: vec![
                Field {id: "x".to_string(), ty: scalar(ElemSize::F32)},
                Field {id: "y".to_string(), ty: scalar(ElemSize::F32)},
            ]
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
        assert_eq!(Attribute::Entry.pprint_default(), "extern \"C\"");
    }

    #[test]
    fn print_scalar_param() {
        let p = Param {id: id("x"), ty: scalar(ElemSize::F16)};
        assert_eq!(p.pprint_default(), "half x");
    }

    #[test]
    fn print_pointer_param() {
        let ty = Type::Pointer{ty: Box::new(scalar(ElemSize::F16))};
        let p = Param {id: id("x"), ty};
        assert_eq!(p.pprint_default(), "half* __restrict__ x");
    }

    #[test]
    fn pprint_fun_def() {
        let def = Top::FunDef {
            dev_attr: Attribute::Entry,
            ret_ty: Type::Void,
            attrs: vec![],
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![
                Stmt::Assign {dst: uvar("x"), expr: uvar("y")}
            ]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("extern \"C\"\nvoid f() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }

    #[test]
    fn pprint_cuda_kernel() {
        let def = Top::FunDef {
            dev_attr: Attribute::Global,
            ret_ty: Type::Void,
            attrs: vec![KernelAttribute::LaunchBounds {threads: 256}],
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![
                Stmt::Assign {dst: uvar("x"), expr: uvar("y")}
            ]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("__global__\nvoid\n__launch_bounds__(256)\nf() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }
}

