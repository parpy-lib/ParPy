use crate::utils::ast::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::pprint::*;
use crate::utils::smap::SFold;

pub use crate::utils::ast::BinOp;
pub use crate::utils::ast::ElemSize;
pub use crate::utils::ast::UnOp;
pub use crate::gpu::ast::LaunchArgs;
pub use crate::gpu::ast::Dim3;

use itertools::Itertools;

#[derive(Clone, Debug)]
pub enum Type {
    Void,
    Scalar {sz: ElemSize},
    Pointer {ty: Box<Type>},
    Array {ty: Box<Type>, sz: Box<Expr>},
    Function {result: Box<Type>, args: Vec<Type>},
    String,

    // Types specific to the CUDA runtime
    CudaFunction,
    CudaResult,
    CudaStream,
}

impl Type {
    fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Scalar {sz} => Some(sz),
            _ => None
        }
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
        Type::Pointer {ty} => pprint_type(format!("(*{decl})"), ty, env),
        Type::Array {ty, sz} => {
            let (env, sz) = sz.pprint(env);
            let (env, ty) = pprint_type(format!("{decl}[{sz}]"), ty, env);
            (env, format!("{ty}"))
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
        Type::String => (env, format!("const char (*{decl})")),
        Type::CudaFunction => join_space("CUfunction".to_string(), decl, env),
        Type::CudaResult => join_space("CUresult".to_string(), decl, env),
        Type::CudaStream => join_space("CUstream".to_string(), decl, env),
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_type("".to_string(), self, env)
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i128, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    Assign {dst: Box<Expr>, expr: Box<Expr>, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    ArrayAccess {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Ternary {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    Call {id: Name, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type, i: Info},

    LoadKernel {path: Name, id: Name, ty: Type, i: Info},
    LaunchKernel {
        id: Name, grid: LaunchArgs, smem: Box<Expr>, stream: Box<Expr>,
        args: Box<Expr>, ty: Type, i: Info
    },
    AllocDevice {id: Name, sz: usize, stream: Name, ty: Type, i: Info},
    FreeDevice {id: Name, stream: Name, ty: Type, i: Info},
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} |
            Expr::Bool {i, ..} |
            Expr::Int {i, ..} |
            Expr::Float {i, ..} |
            Expr::Assign {i, ..} |
            Expr::UnOp {i, ..} |
            Expr::BinOp {i, ..} |
            Expr::ArrayAccess {i, ..} |
            Expr::Ternary {i, ..} |
            Expr::Call {i, ..} |
            Expr::Convert {i, ..} |
            Expr::LoadKernel {i, ..} |
            Expr::LaunchKernel {i, ..} |
            Expr::AllocDevice {i, ..} |
            Expr::FreeDevice {i, ..} => i.clone()
        }
    }
}

impl ExprType<Type> for Expr {
    fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} |
            Expr::Bool {ty, ..} |
            Expr::Int {ty, ..} |
            Expr::Float {ty, ..} |
            Expr::Assign {ty, ..} |
            Expr::UnOp {ty, ..} |
            Expr::BinOp {ty, ..} |
            Expr::ArrayAccess {ty, ..} |
            Expr::Ternary {ty, ..} |
            Expr::Call {ty, ..} |
            Expr::Convert {ty, ..} |
            Expr::LoadKernel {ty, ..} |
            Expr::LaunchKernel {ty, ..} |
            Expr::AllocDevice {ty, ..} |
            Expr::FreeDevice {ty, ..} => ty
        }
    }

    fn is_leaf_node(&self) -> bool {
        match self {
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::LoadKernel {..} |
            Expr::AllocDevice {..} |
            Expr::FreeDevice {..} => true,
            Expr::Assign {..} |
            Expr::UnOp {..} |
            Expr::BinOp {..} |
            Expr::ArrayAccess {..} |
            Expr::Ternary {..} |
            Expr::Call {..} |
            Expr::Convert {..} |
            Expr::LaunchKernel {..} => false
        }
    }
}

impl SFold<Expr> for Expr {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> where Self: Sized {
        match self {
            Expr::Assign {dst, expr, ..} => f(f(acc?, dst)?, expr),
            Expr::UnOp {arg, ..} => f(acc?, arg),
            Expr::BinOp {lhs, rhs, ..} => f(f(acc?, lhs)?, rhs),
            Expr::ArrayAccess {target, idx, ..} => f(f(acc?, target)?, idx),
            Expr::Ternary {cond, thn, els, ..} => f(f(f(acc?, cond)?, thn)?, els),
            Expr::Call {args, ..} => args.sfold_result(acc, &f),
            Expr::Convert {e, ..} => f(acc?, e),
            Expr::LaunchKernel {smem, stream, args, ..} => f(f(f(acc?, smem)?, stream)?, args),
            Expr::Var {..} |
            Expr::Bool {..} |
            Expr::Int {..} |
            Expr::Float {..} |
            Expr::LoadKernel {..} |
            Expr::AllocDevice {..} |
            Expr::FreeDevice {..} => acc,
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
            UnOp::Sub |
            UnOp::Not |
            UnOp::BitNeg |
            UnOp::Addressof => false,
            UnOp::Sqrt => true
        }
    }

    fn print_unop(op: &UnOp, _argty: &Type) -> Option<String> {
        let s = match op {
            UnOp::Sub => "-",
            UnOp::Not => "!",
            UnOp::BitNeg => "~",
            UnOp::Addressof => "&",
            UnOp::Sqrt => "sqrt",
        };
        Some(s.to_string())
    }
}

impl PrettyPrintBinOp<Type> for Expr {
    fn extract_binop<'a>(&'a self) -> Option<(&'a Expr, &'a BinOp, &'a Expr, &'a Type)> {
        if let Expr::BinOp {lhs, op, rhs, ty, i: _} = self {
            Some((lhs, op, rhs, ty))
        } else {
            None
        }
    }

    fn is_infix(op: &BinOp, _ty: &Type) -> bool {
        match op {
            BinOp::Pow |
            BinOp::Max |
            BinOp::Min => false,
            _ => true
        }
    }

    fn print_binop(op: &BinOp, _argty: &Type, ty: &Type) -> Option<String> {
        let o = match op {
            BinOp::Add => Some("+"),
            BinOp::Sub => Some("-"),
            BinOp::Mul => Some("*"),
            BinOp::FloorDiv |
            BinOp::Div => Some("/"),
            BinOp::Rem => Some("%"),
            BinOp::Pow => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("hpow"),
                Some(ElemSize::F32) => Some("powf"),
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
            BinOp::Eq => Some("=="),
            BinOp::Neq => Some("!="),
            BinOp::Leq => Some("<="),
            BinOp::Geq => Some(">="),
            BinOp::Lt => Some("<"),
            BinOp::Gt => Some(">"),
            BinOp::Max => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hmax"),
                Some(ElemSize::F32) => Some("fmaxf"),
                Some(ElemSize::F64) => Some("fmax"),
                Some(_) => Some("max"),
                None => None
            },
            BinOp::Min => match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) => Some("__hmin"),
                Some(ElemSize::F32) => Some("fminf"),
                Some(ElemSize::F64) => Some("fmin"),
                Some(_) => Some("min"),
                None => None
            },
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
            Expr::Var {id, ty: _, i: _} => id.pprint(env),
            Expr::Bool {v, ty: _, i: _} => (env, v.to_string()),
            Expr::Int {v, ty: _, i: _} => (env, v.to_string()),
            Expr::Float {v, ty, i: _} => {
                let s = match ty.get_scalar_elem_size() {
                    Some(ElemSize::F16) => "CUDART_INF_FP16",
                    Some(ElemSize::F32) => "HUGE_VALF",
                    Some(ElemSize::F64) => "HUGE_VAL",
                    _ => "INVALID"
                };
                print_float(env, v, s)
            },
            Expr::Assign {dst, expr, ty: _, i: _} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{dst} = {expr}"))
            },
            Expr::UnOp {..} => self.print_parenthesized_unop(env),
            Expr::BinOp {..} => self.print_parenthesized_binop(env),
            Expr::ArrayAccess {target, idx, ty: _, i: _} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::Ternary {cond, thn, els, ty: _, i: _} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("({cond} ? {thn} : {els})"))
            },
            Expr::Call {id, args, ty: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                (env, format!("{id}({args})"))
            },
            Expr::Convert {e, ty, i: _} => {
                let (env, e_str) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                let s = if e.is_leaf_node() {
                    format!("({ty}){e_str}")
                } else {
                    format!("({ty})({e_str})")
                };
                (env, s)
            },
            Expr::LoadKernel {path, id, ty: _, i: _} => {
                let (env, path) = path.pprint(env);
                let (env, id) = id.pprint(env);
                (env, format!("parpy_triton::load_kernel({path}, \"{id}\")"))
            },
            Expr::LaunchKernel {id, grid, smem, stream, args, ty: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, smem) = smem.pprint(env);
                let (env, stream) = stream.pprint(env);
                let (env, args) = args.pprint(env);
                let Dim3 {x: bx, y: by, z: bz} = grid.blocks;
                let Dim3 {x: tx, y: ty, z: tz} = grid.threads;
                (env, format!("cuLaunchKernel({id}, {bx}, {by}, {bz}, {tx}, {ty}, {tz}, {smem}, {stream}, {args}, NULL)"))
            },
            Expr::AllocDevice {id, sz, stream, ty: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, stream) = stream.pprint(env);
                (env, format!("cuMemAllocAsync((CUdeviceptr)&{id}, {sz}, {stream})"))
            },
            Expr::FreeDevice {id, stream, ty: _, i: _} => {
                let (env, id) = id.pprint(env);
                let (env, stream) = stream.pprint(env);
                (env, format!("cuMemFreeAsync({id}, {stream})"))
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Definition {ty: Type, dst: Name, expr: Option<Expr>},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>},
    For {
        var_ty: Type, var: Name, init: Expr, cond: Expr, incr: Expr, body: Vec<Stmt>
    },
    While {cond: Expr, body: Vec<Stmt>},
    Scope {body: Vec<Stmt>},
    Expr {e: Expr},
    Return {e: Expr},
    CheckError {e: Expr},
    CheckNonNull {e: Expr},
}

impl SFold<Expr> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Expr) -> Result<A, E>
    ) -> Result<A, E> where Self: Sized {
        match self {
            Stmt::Definition {expr, ..} => match expr {
                Some(e) => f(acc?, e),
                None => acc
            },
            Stmt::If {cond, ..} => f(acc?, cond),
            Stmt::For {init, cond, incr, ..} => f(f(f(acc?, init)?, cond)?, incr),
            Stmt::While {cond, ..} => f(acc?, cond),
            Stmt::Scope {..} => acc,
            Stmt::Expr {e, ..} => f(acc?, e),
            Stmt::Return {e, ..} => f(acc?, e),
            Stmt::CheckError {e, ..} => f(acc?, e),
            Stmt::CheckNonNull {e, ..} => f(acc?, e),
        }
    }
}

impl SFold<Stmt> for Stmt {
    fn sfold_result<A, E>(
        &self,
        acc: Result<A, E>,
        f: impl Fn(A, &Stmt) -> Result<A, E>
    ) -> Result<A, E> where Self: Sized {
        match self {
            Stmt::If {thn, els, ..} => els.sfold_result(thn.sfold_result(acc, &f), &f),
            Stmt::For {body, ..} |
            Stmt::While {body, ..} |
            Stmt::Scope {body, ..} => body.sfold_result(acc, &f),
            Stmt::Definition {..} |
            Stmt::Expr {..} |
            Stmt::Return {..} |
            Stmt::CheckError {..} |
            Stmt::CheckNonNull {..} => acc
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
            Stmt::Definition {ty, dst, expr} => {
                let (env, dst) = dst.pprint(env);
                let (env, decl) = pprint_type(dst, ty, env);
                match expr {
                    Some(e) => {
                        let (env, e) = e.pprint(env);
                        (env, format!("{indent}{decl} = {e};"))
                    },
                    None => (env, format!("{indent}{decl};"))
                }
            },
            Stmt::If {..} => self.print_cond(env),
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}for ({var_ty} {var} = {init}; {cond}; {var} = {incr}) {{\n{body}\n{0}}}", indent))
            },
            Stmt::While {cond, body} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}while ({cond}) {{\n{body}\n{0}}}", indent))
            },
            Stmt::Scope {body} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}{{\n{body}\n{0}}}", indent))
            },
            Stmt::Expr {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}{e};"))
            },
            Stmt::Return {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}return {e};"))
            },
            Stmt::CheckError {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}parpy_triton_check_error({e});"))
            },
            Stmt::CheckNonNull {e} => {
                let (env, e) = e.pprint(env);
                (env, format!("{indent}parpy_triton_check_non_null({e});"))
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub ty: Type,
    pub id: Name
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {ty, id} = self;
        let (env, id) = id.pprint(env);
        pprint_type(id, ty, env)
    }
}

#[derive(Clone, Debug)]
pub enum Top {
    Include {header: String},
    ExtDecl {id: Name, ext_id: String, params: Vec<Param>},
    VarDef {ty: Type, id: Name, value: Option<Expr>},
    FunDef {ret_ty: Type, id: Name, params: Vec<Param>, body: Vec<Stmt>},
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::Include {header} => (env, format!("#include {header}")),
            Top::ExtDecl {id, ext_id, params} => {
                let (env, id) = id.pprint(env);
                let param_ids = params.iter()
                    .map(|Param {id, ..}| id.get_str())
                    .join(", ");
                (env, format!("#define {id}({0}) {ext_id}({0})", param_ids))
            },
            Top::VarDef {ty, id, value} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                match value {
                    Some(v) => {
                        let (env, v) = v.pprint(env);
                        (env, format!("{ty} {id} = {v};"))
                    }
                    None => (env, format!("{ty} {id};"))
                }
            },
            Top::FunDef {ret_ty, id, params, body} => {
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("extern \"C\"\n{ret_ty} {id}({params}) {{\n{body}\n}}"))
            },
        }
    }
}

pub type Ast = Vec<Top>;    

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_iter(self.iter(), env, "\n")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::*;

    fn var(s: &str) -> Expr {
        Expr::Var {id: Name::sym_str(s), ty: Type::Void, i: i()}
    }

    fn int(v: i128) -> Expr {
        Expr::Int {v, ty: Type::Scalar {sz: ElemSize::I32}, i: i()}
    }

    #[test]
    fn print_void_type() {
        let ty = Type::Void;
        assert_eq!(ty.pprint_default(), "void");
    }

    #[test]
    fn print_int32_type() {
        let ty = Type::Scalar {sz: ElemSize::I32};
        assert_eq!(ty.pprint_default(), "int32_t");
    }

    #[test]
    fn print_pointer_type() {
        let ty = Type::Pointer {ty: Box::new(Type::Void)};
        assert_eq!(ty.pprint_default(), "void (*)");
    }

    #[test]
    fn print_array_pointer_type() {
        let ty = Type::Array {
            ty: Box::new(Type::Pointer {ty: Box::new(Type::Void)}),
            sz: Box::new(int(10)),
        };
        assert_eq!(ty.pprint_default(), "void (*[10])");
    }

    #[test]
    fn print_function_type() {
        let ty = Type::Function {
            result: Box::new(Type::Void),
            args: vec![
                Type::Scalar {sz: ElemSize::F32},
                Type::Scalar {sz: ElemSize::F64},
            ],
        };
        assert_eq!(ty.pprint_default(), "void (float, double)");
    }

    #[test]
    fn print_array_access_expr() {
        let e = Expr::ArrayAccess {
            target: Box::new(var("x")),
            idx: Box::new(var("y")),
            ty: Type::Void,
            i: i()
        };
        assert_eq!(e.pprint_default(), "x[y]");
    }

    #[test]
    fn print_array_pointer_declaration() {
        let s = Stmt::Definition {
            ty: Type::Array {
                ty: Box::new(Type::Pointer {ty: Box::new(Type::Void)}),
                sz: Box::new(int(10)),
            },
            dst: Name::sym_str("x"),
            expr: None
        };
        assert_eq!(s.pprint_default(), "void (*x[10]);");
    }

    #[test]
    fn print_check_error() {
        let s = Stmt::CheckError {e: int(0)};
        assert_eq!(s.pprint_default(), "parpy_triton_check_error(0);");
    }

    #[test]
    fn print_check_non_null() {
        let s = Stmt::CheckNonNull {e: int(0)};
        assert_eq!(s.pprint_default(), "parpy_triton_check_non_null(0);");
    }

    #[test]
    fn print_array_reassignment() {
        let s = Stmt::Expr {
            e: Expr::Assign {
                dst: Box::new(Expr::ArrayAccess {
                    target: Box::new(var("x")),
                    idx: Box::new(var("y")),
                    ty: Type::Void,
                    i: i()
                }),
                expr: Box::new(Expr::UnOp {
                    op: UnOp::Addressof,
                    arg: Box::new(var("z")),
                    ty: Type::Void,
                    i: i()
                }),
                ty: Type::Void,
                i: i()
            }
        };
        assert_eq!(s.pprint_default(), "x[y] = &z;");
    }

    #[test]
    fn print_fun_def() {
        let t = Top::FunDef {
            ret_ty: Type::Scalar {sz: ElemSize::I32},
            id: Name::sym_str("fn"),
            params: vec![
                Param {id: Name::sym_str("x"), ty: Type::Scalar {sz: ElemSize::F32}},
                Param {
                    id: Name::sym_str("y"),
                    ty: Type::Pointer {ty: Box::new(Type::Scalar {sz: ElemSize::F32})}
                },
            ],
            body: vec![
                Stmt::Return {e: int(0)}
            ]
        };
        assert_eq!(t.pprint_default(), "extern \"C\"\nint32_t fn(float x, float (*y)) {\n  return 0;\n}");
    }

    #[test]
    fn print_var_decl() {
        let t = Top::VarDef {
            ty: Type::CudaFunction,
            id: Name::sym_str("fn"),
            value: None
        };
        assert_eq!(t.pprint_default(), "CUfunction fn;");
    }

    #[test]
    fn print_var_def() {
        let t = Top::VarDef {
            ty: Type::Scalar {sz: ElemSize::I64},
            id: Name::sym_str("init"),
            value: Some(int(0))
        };
        assert_eq!(t.pprint_default(), "int64_t init = 0;");
    }

    #[test]
    fn print_include() {
        let t = Top::Include {header: "<cuda.h>".to_string()};
        assert_eq!(t.pprint_default(), "#include <cuda.h>");
    }
}
