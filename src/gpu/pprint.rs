use super::ast::*;
use crate::utils::ast::ExprType;
use crate::utils::pprint::*;

use itertools::Itertools;

use std::collections::BTreeMap;

impl PrettyPrint for ElemSize {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            ElemSize::Bool => "bool",
            ElemSize::I8 => "int8_t",
            ElemSize::I16 => "int16_t",
            ElemSize::I32 => "int32_t",
            ElemSize::I64 => "int64_t",
            ElemSize::U8 => "uint8_t",
            ElemSize::U16 => "uint16_t",
            ElemSize::U32 => "uint32_t",
            ElemSize::U64 => "uint64_t",
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

impl PrettyPrint for MemSpace {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            MemSpace::Host => "host",
            MemSpace::Device => "device",
        };
        (env, s.to_string())
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

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Void => (env, "void".to_string()),
            Type::Scalar {sz} => sz.pprint(env),
            Type::Pointer {ty, mem} => {
                let (env, ty) = ty.pprint(env);
                let (env, mem) = mem.pprint(env);
                (env, format!("{mem} {ty}*"))
            },
            Type::Struct {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("{id}"))
            },
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
            UnOp::Sub | UnOp::Not | UnOp::BitNeg | UnOp::Addressof => false,
            UnOp::Exp | UnOp::Log | UnOp::Cos | UnOp::Sin | UnOp::Sqrt |
            UnOp::Tanh | UnOp::Abs => true,
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
            UnOp::Addressof => "&",
        };
        s.to_string()
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

    fn print_binop(op: &BinOp, _argty: &Type, _ty: &Type) -> String {
        let s = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::FloorDiv | BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Pow => "pow",
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
            BinOp::Atan2 => "atan2",
        };
        s.to_string()
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
            Expr::Float {v, ..} => print_float(env, v, "inf"),
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
            Expr::IfExpr {cond, thn, els, ..} => {
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
            Expr::Struct {id, fields, ..} => {
                let (env, id) = id.pprint(env);
                let (env, fields) = fields.iter()
                    .fold((env, vec![]), |(env, mut strs), (id, e)| {
                        let (env, e) = e.pprint(env);
                        strs.push(format!(".{id}: {e}"));
                        (env, strs)
                    });
                let outer_indent = env.print_indent();
                let env = env.incr_indent();
                let indent = env.print_indent();
                let fields = fields.into_iter().join(&format!(",\n{indent}"));
                let env = env.decr_indent();
                (env, format!("{id} {{\n{indent}{fields}\n{outer_indent}}}"))
            },
            Expr::ThreadIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("threadIdx.{dim}"))
            },
            Expr::BlockIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("blockIdx.{dim}"))
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
        (env, format!("{indent}{{blocks: ({blocks}), threads: ({threads})}};"))
    }
}

impl PrettyPrint for SyncScope {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            SyncScope::Block => format!("block"),
            SyncScope::Cluster => format!("cluster"),
        };
        (env, s)
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
            Stmt::While {cond, body, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!("{0}while ({1}) {{\n{2}\n{0}}}", indent, cond, body);
                (env, s)
            },
            Stmt::Return {value, ..} => {
                let (env, value) = value.pprint(env);
                (env, format!("{indent}return {value};"))
            },
            Stmt::Scope {body, ..} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}{{\n{1}\n{0}}}", indent, body))
            },
            Stmt::ParallelReduction {var_ty, var, init, cond, incr, body, nthreads, tpb, ..} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let threading = format!("(parallel reduction over {nthreads} threads with {tpb} threads per block)");
                let s = format!(
                    "{0}for ({1} {2} = {3}; {4}; {2} = {5}) {7}\n{{\n{6}\n{0}}}",
                    indent, var_ty, var, init, cond, incr, body, threading
                );
                (env, s)
            },
            Stmt::Synchronize {scope, ..} => {
                let (env, scope) = scope.pprint(env);
                (env, format!("{indent}sync({scope});"))
            },
            Stmt::WarpReduce {value, op, ..} => {
                let (env, value) = value.pprint(env);
                let (env, op) = op.pprint(env);
                (env, format!("{indent}{value} = warp_reduce({value}, {op});"))
            },
            Stmt::ClusterReduce {block_idx, shared_var, temp_var, op, ..} => {
                let (env, block_idx) = block_idx.pprint(env);
                let (env, shared_var) = shared_var.pprint(env);
                let (env, temp_var) = temp_var.pprint(env);
                let (env, op) = op.pprint(env);
                (env, format!("{indent}{temp_var} = cluster_reduce({shared_var}[{block_idx}], {op});"))
            },
            Stmt::KernelLaunch {id, args, grid, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let (env, grid) = grid.pprint(env);
                (env, format!("{indent}launch({id}, [{args}], {grid});"))
            },
            Stmt::AllocDevice {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                (env, format!("{indent}{id} = alloc_device({sz}, {elem_ty});"))
            },
            Stmt::AllocShared {id, elem_ty, sz, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                (env, format!("{indent}shared {elem_ty} {id}[{sz}];"))
            },
            Stmt::FreeDevice {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("{indent}free({id});"))
            },
            Stmt::CopyMemory {elem_ty, src, dst, sz, src_mem, dst_mem, ..} => {
                let (env, elem_ty) = elem_ty.pprint(env);
                let (env, src) = src.pprint(env);
                let (env, dst) = dst.pprint(env);
                let (env, src_mem) = src_mem.pprint(env);
                let (env, dst_mem) = dst_mem.pprint(env);
                (env, format!("{indent}memcpy({dst}: {dst_mem} <- {src}: {src_mem}, {sz}, {elem_ty});"))
            }
        }
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty, ..} = self;
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

impl PrettyPrint for Field {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Field {id, ty, ..} = self;
        let (env, ty) = ty.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{ty} {id};"))
    }
}

impl PrettyPrint for Target {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Target::Device => (env, format!("device")),
            Target::Host => (env, format!("host")),
        }
    }
}

impl PrettyPrint for KernelAttribute {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            KernelAttribute::LaunchBounds {threads} => (env, format!("threads({threads})")),
            KernelAttribute::ClusterDims {dims} => {
                let (env, dims) = dims.pprint(env);
                (env, format!("cluster_dims({dims})"))
            },
        }
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::KernelFunDef {attrs, id, params, body} => {
                let (env, attrs) = pprint_iter(attrs.iter(), env, "\n");
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{attrs}\nvoid {id}({params}) {{\n{body}\n}}"))
            },
            Top::FunDef {ret_ty, id, params, body, target} => {
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let (env, target) = target.pprint(env);
                (env, format!("[{target}] {ret_ty} {id}({params}) {{\n{body}\n}}"))
            },
            Top::StructDef {id, fields} => {
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let (env, fields) = pprint_iter(fields.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("struct {id} {{\n{fields}\n}};"))
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
    use crate::utils::info::Info;
    use crate::utils::name::Name;
    use crate::utils::pprint;

    fn var(s: &str) -> Expr {
        Expr::Var {id: Name::new(s.to_string()), ty: Type::Boolean, i: Info::default()}
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v: v as i128, ty: Type::Scalar {sz: ElemSize::I64}, i: Info::default()}
    }

    fn bop(lhs: Expr, op: BinOp, rhs: Expr, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(Type::Boolean);
        Expr::BinOp {
            lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: Info::default()
        }
    }

    fn unop(op: UnOp, arg: Expr, ty: Type) -> Expr {
        Expr::UnOp {op, arg: Box::new(arg), ty, i: Info::default()}
    }

    fn add(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Add, rhs, None)
    }

    fn mul(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Mul, rhs, None)
    }

    fn rem(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Rem, rhs, None)
    }

    fn scalar_ty(sz: ElemSize) -> Type {
        Type::Scalar {sz}
    }

    fn int64_ty() -> Type {
        scalar_ty(ElemSize::I64)
    }

    fn i() -> Info {
        Info::default()
    }

    #[test]
    fn pprint_precedence_same_level_with_paren() {
        let s = add(var("x"), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "x + (y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_omit_paren() {
        let s = add(add(var("x"), var("y")), var("z")).pprint_default();
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_print_paren() {
        let s = mul(add(var("x"), var("y")), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "(x + y) * (y + z)");
    }

    #[test]
    fn pprint_precedence_rhs_paren() {
        let s = add(var("x"), add(mul(var("y"), var("y")), var("z"))).pprint_default();
        assert_eq!(&s, "x + (y * y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_paren() {
        let s = mul(var("x"), rem(var("y"), var("z"))).pprint_default();
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
        let s = Expr::ThreadIdx {dim: Dim::X, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "threadIdx.x");
    }

    #[test]
    fn pprint_block_idx_y() {
        let s = Expr::BlockIdx {dim: Dim::Y, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "blockIdx.y");
    }

    fn exp(arg: Expr, ty: Type) -> Expr {
        unop(UnOp::Exp, arg, ty)
    }

    fn log(arg: Expr, ty: Type) -> Expr {
        unop(UnOp::Log, arg, ty)
    }

    fn max(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
        bop(lhs, BinOp::Max, rhs, Some(ty))
    }

    #[test]
    fn pprint_exp_f32() {
        let s = exp(var("x"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "exp(x)");
    }

    #[test]
    fn pprint_log_f64() {
        let s = log(var("x"), scalar_ty(ElemSize::F64)).pprint_default();
        assert_eq!(&s, "log(x)");
    }

    #[test]
    fn pprint_max_f32() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "max(x, y)");
    }

    #[test]
    fn pprint_max_i64() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::I64)).pprint_default();
        assert_eq!(&s, "max(x, y)");
    }

    fn convert(e: Expr, ty: Type) -> Expr {
        Expr::Convert {e: Box::new(e), ty}
    }

    #[test]
    fn pprint_var_conversion() {
        let s = convert(var("x"), scalar_ty(ElemSize::F32));
        assert_eq!(&s.pprint_default(), "(float)x");
    }

    #[test]
    fn pprint_literal_conversion() {
        let s = convert(int(5), scalar_ty(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)5");
    }

    #[test]
    fn pprint_add_conversion() {
        let s = convert(add(var("x"), var("y")), scalar_ty(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)(x + y)");
    }

    #[test]
    fn pprint_launch_args() {
        let args = LaunchArgs::default()
            .with_blocks_dim(&Dim::Y, 2)
            .with_blocks_dim(&Dim::Z, 3)
            .with_threads_dim(&Dim::X, 4)
            .with_threads_dim(&Dim::Y, 5)
            .with_threads_dim(&Dim::Z, 6);
        let s = args.pprint_default();
        assert_eq!(&s, "{blocks: (1, 2, 3), threads: (4, 5, 6)};");
    }

    #[test]
    fn pprint_syncblock() {
        let s = Stmt::Synchronize {scope: SyncScope::Block, i: i()}.pprint_default();
        assert_eq!(&s, "sync(block);");
    }

    #[test]
    fn pprint_for_loop() {
        let id = Name::new("i".to_string());
        let i_var = Expr::Var {id: id.clone(), ty: scalar_ty(ElemSize::I64), i: i()};
        let for_loop = Stmt::For {
            var_ty: scalar_ty(ElemSize::I64),
            var: id,
            init: int(0),
            cond: bop(i_var.clone(), BinOp::Lt, int(10), None),
            incr: bop(i_var.clone(), BinOp::Add, int(1), None),
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            i: i()
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
                lhs: Box::new(var("x")),
                op: BinOp::Eq,
                rhs: Box::new(var("y")),
                ty: Type::Boolean,
                i: i()
            },
            thn: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            els: vec![Stmt::Assign {dst: var("y"), expr: var("x"), i: i()}],
            i: i()
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
                i: i()
            },
            thn: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            els: vec![],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("if (x == y) {{\n{indent}x = y;\n}}");
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_elseif() {
        let cond = Stmt::If {
            cond: var("x"),
            thn: vec![Stmt::Assign {dst: var("y"), expr: var("z"), i: i()}],
            els: vec![Stmt::If {
                    cond: var("y"),
                    thn: vec![Stmt::Assign {dst: var("x"), expr: var("z"), i: i()}],
                    els: vec![Stmt::Assign {dst: var("z"), expr: var("x"), i: i()}],
                    i: i()
            }],
            i: i()
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
            cond: var("x"),
            body: vec![Stmt::Assign {dst: var("y"), expr: var("z"), i: i()}],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("while (x) {{\n{indent}y = z;\n}}");
        assert_eq!(wh.pprint_default(), expected);
    }

    #[test]
    fn pprint_warp_reduce() {
        let value = var("x");
        let op = BinOp::Add;
        let reduce = Stmt::WarpReduce {
            value: value.clone(),
            op: op.clone(),
            int_ty: Type::Scalar {sz: ElemSize::I64},
            res_ty: Type::Scalar {sz: ElemSize::F32},
            i: i()
        };
        let x = value.pprint_default();
        let op_str = op.pprint_default();
        let expected = format!("{x} = warp_reduce({x}, {op_str});");
        assert_eq!(reduce.pprint_default(), expected);
    }

    #[test]
    fn pprint_kernel_launch() {
        let id = "kernel";
        let kernel = Stmt::KernelLaunch {
            id: Name::new(id.to_string()),
            args: vec![var("x"), var("y")],
            grid: LaunchArgs::default(),
            i: i()
        };
        let grid_str = LaunchArgs::default().pprint_default();
        let expected = format!("launch({id}, [x, y], {grid_str});");
        assert_eq!(kernel.pprint_default(), expected);
    }

    #[test]
    fn pprint_kernel_fun_def() {
        let def = Top::KernelFunDef {
            attrs: vec![KernelAttribute::LaunchBounds {threads: 1024}],
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("threads(1024)\nvoid f() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }

    #[test]
    fn pprint_host_fun_def() {
        let def = Top::FunDef {
            ret_ty: Type::Void,
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            target: Target::Host
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("[host] void f() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }

    #[test]
    fn pprint_device_fun_def() {
        let def = Top::FunDef {
            ret_ty: Type::Void,
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            target: Target::Device
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("[device] void f() {{\n{0}x = y;\n}}", indent);
        assert_eq!(def.pprint_default(), expected);
    }
}

