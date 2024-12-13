#[derive(Clone, Copy, Debug)]
pub enum IntSize {
    I8, I16, I32, I64
}

#[derive(Clone, Copy, Debug)]
pub enum FloatSize {
    F16, F32, F64
}

#[derive(Clone, Debug)]
pub enum Type {
    Int(IntSize),
    Float(FloatSize),
    IntTensor(IntSize),
    FloatTensor(FloatSize)
}

#[derive(Clone, Copy, Debug)]
pub enum BinOp {
    Add, Mul
}

#[derive(Clone, Copy, Debug)]
pub enum Dim {
    X, Y, Z
}

#[derive(Clone, Debug)]
pub enum Expr {
    Var {id : String, ty : Type},
    Int {v : i64, ty : Type},
    Float {v : f64, ty : Type},
    Str {v : String},
    BinOp {lhs : Box<Expr>, op : BinOp, rhs : Box<Expr>, ty : Type},
    Subscript {target : Box<Expr>, idx : Box<Expr>, ty : Type},
    Call {fun : String, args : Vec<Expr>},
    ThreadIdx(Dim),
    BlockIdx(Dim)
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Expr {e : Expr},
    Defn {ty : Type, id : String, e : Expr},
    Assign {dst : Expr, e : Expr},
    For {var_ty : Type, var : String, init : Expr, cond : Expr, incr : Expr, body : Vec<Stmt>},
    KernelLaunch {
        threads : (i64, i64, i64),
        blocks : (i64, i64, i64),
        id : String,
        args : Vec<Expr>
    }
}

#[derive(Clone, Debug)]
pub struct TypedParam {
    pub id : String,
    pub ty : Type
}

#[derive(Clone, Debug)]
pub enum HostTop {
    FunDecl {id : String, params : Vec<TypedParam>},
    FunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>},
}

#[derive(Clone, Debug)]
pub enum DeviceTop {
    KernelFunDef {id : String, params : Vec<TypedParam>, body : Vec<Stmt>}
}

pub struct Ast {
    // Definitions of entry-points to the host code. These do sanity checking on
    // inputs before passing over control to a host stage function.
    pub host_entry : Vec<HostTop>,

    // Host stage functions that may launch device kernels.
    pub host_stage : Vec<HostTop>,

    // Kernel functions runnin on the GPU
    pub kernels : Vec<DeviceTop>
}

impl Ast {
    pub fn new() -> Self {
        Ast {host_entry : vec![], host_stage : vec![], kernels : vec![]}
    }
}

/////////////////////
// PRETTY PRINTING //
/////////////////////

#[derive(Clone)]
pub struct PrintConfig {
    indent : isize,
    indent_sz : isize,
    print_pointers : bool,
}

impl Default for PrintConfig {
    fn default() -> PrintConfig {
        PrintConfig {
            indent : 0,
            indent_sz : 2,
            print_pointers : false
        }
    }
}

fn pprint_indent(spaces : isize) -> String {
    " ".repeat(spaces as usize)
}

pub trait PrettyPrint {
    fn pprint(&self, cfg : &mut PrintConfig) -> String;
}

pub fn pprint_vec<T : PrettyPrint>(
    v : &Vec<T>,
    separator : &str,
    cfg : &mut PrintConfig
) -> String {
    v.iter().map(|x| x.pprint(cfg)).collect::<Vec<String>>().join(separator)
}

impl PrettyPrint for Type {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        match self {
            Type::Int(IntSize::I8) => "int8_t".to_string(),
            Type::Int(IntSize::I16) => "int16_t".to_string(),
            Type::Int(IntSize::I32) => "int32_t".to_string(),
            Type::Int(IntSize::I64) => "int64_t".to_string(),
            Type::Float(FloatSize::F16) => "half".to_string(),
            Type::Float(FloatSize::F32) => "float".to_string(),
            Type::Float(FloatSize::F64) => "double".to_string(),
            Type::IntTensor(sz) => {
                if cfg.print_pointers {
                    let elem_ty = Type::Int(*sz);
                    let ty = elem_ty.pprint(cfg);
                    format!("{ty}*")
                } else {
                    format!("torch::Tensor")
                }
            },
            Type::FloatTensor(sz) => {
                if cfg.print_pointers {
                    let elem_ty = Type::Float(*sz);
                    let ty = elem_ty.pprint(cfg);
                    format!("{ty}*")
                } else {
                    format!("torch::Tensor")
                }
            }
        }
    }
}

impl PrettyPrint for BinOp {
    fn pprint(&self, _ : &mut PrintConfig) -> String {
        match self {
            BinOp::Add => "+".to_string(),
            BinOp::Mul => "*".to_string()
        }
    }
}

impl PrettyPrint for Dim {
    fn pprint(&self, _ : &mut PrintConfig) -> String {
        match self {
            Dim::X => "x".to_string(),
            Dim::Y => "y".to_string(),
            Dim::Z => "z".to_string(),
        }
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        match self {
            Expr::Var {id, ..} => format!("{id}"),
            Expr::Int {v, ..} => format!("{v}"),
            Expr::Float {v, ..} => format!("{v}"),
            Expr::Str {v} => format!("{v}"),
            Expr::BinOp {lhs, op, rhs, ..} => {
                let lhs = lhs.pprint(cfg);
                let op = op.pprint(cfg);
                let rhs = rhs.pprint(cfg);
                format!("({lhs} {op} {rhs})")
            }
            Expr::Subscript {target, idx, ..} => {
                let target = target.pprint(cfg);
                let idx = idx.pprint(cfg);
                format!("{target}[{idx}]")
            },
            Expr::Call {fun, args} => {
                let args = pprint_vec(args, ", ", cfg);
                format!("{fun}({args})")
            }
            Expr::ThreadIdx(dim) => {
                let dim = dim.pprint(cfg);
                format!("threadIdx.{dim}")
            },
            Expr::BlockIdx(dim) => {
                let dim = dim.pprint(cfg);
                format!("blockIdx.{dim}")
            }
        }
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        let ii = pprint_indent(cfg.indent);
        match self {
            Stmt::Expr {e} => {
                format!("{ii}{0};", e.pprint(cfg))
            },
            Stmt::Defn {ty, id, e} => {
                let ty = ty.pprint(cfg);
                let e = e.pprint(cfg);
                format!("{ii}{ty} {id} = {e};")
            },
            Stmt::Assign {dst, e} => {
                let dst = dst.pprint(cfg);
                let e = e.pprint(cfg);
                format!("{ii}{dst} = {e};")
            },
            Stmt::For {var_ty, var, init, cond, incr, body} => {
                let var_ty = var_ty.pprint(cfg);
                let init = init.pprint(cfg);
                let cond = cond.pprint(cfg);
                let incr = incr.pprint(cfg);
                cfg.indent = cfg.indent + cfg.indent_sz;
                let body = pprint_vec(body, "\n", cfg);
                cfg.indent = cfg.indent - cfg.indent_sz;
                format!("{ii}for ({var_ty} {var} = {init}; {var} < {cond}; {var} += {incr}) {{\n{body}\n{ii}}}")
            },
            Stmt::KernelLaunch {threads, blocks, id, args} => {
                let args = pprint_vec(args, ", ", cfg);
                let threads = format!("({0}, {1}, {2})", threads.0, threads.1, threads.2);
                let blocks = format!("({0}, {1}, {2})", blocks.0, blocks.1, blocks.2);
                format!("{ii}{id}<<<{blocks}, {threads}>>>({args});")
            },
        }
    }
}

impl PrettyPrint for TypedParam {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        let TypedParam {id, ty} = self;
        let ty = ty.pprint(cfg);
        format!("{ty} {id}")
    }
}

impl PrettyPrint for HostTop {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        match self {
            HostTop::FunDecl {id, params} => {
                cfg.print_pointers = false;
                let params = pprint_vec(params, ", ", cfg);
                format!("void {id}({params});")
            },
            HostTop::FunDef {id, params, body} => {
                cfg.print_pointers = false;
                let params = pprint_vec(params, ", ", cfg);
                cfg.indent = cfg.indent + cfg.indent_sz;
                let body = pprint_vec(body, "\n", cfg);
                cfg.indent = cfg.indent - cfg.indent_sz;
                format!("void {id}({params}) {{\n{body}\n}}")
            }
        }
    }
}

impl PrettyPrint for DeviceTop {
    fn pprint(&self, cfg : &mut PrintConfig) -> String {
        match self {
            DeviceTop::KernelFunDef {id, params, body} => {
                cfg.print_pointers = true;
                let params = pprint_vec(params, ", ", cfg);
                let body = pprint_vec(body, "\n", cfg);
                format!("__global__ void {id}({params}) {{\n{body}\n}}")
            }
        }
    }
}

const MACROS : &'static str = "
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x \" must be a CUDA tensor\")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x \" must be contiguous\")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)
";

fn declare_torch_modules(host_entry : &Vec<HostTop>) -> String {
    let pre = "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {";
    let post = "}";
    let mid = host_entry.iter()
        .flat_map(|entry| match entry {
            HostTop::FunDef {id, ..} => {
                vec![format!("  m.def(\"{0}\", &{0}, \"Parir {0}\")", id)]
            },
            _ => vec![]
        })
        .collect::<Vec<String>>()
        .join("\n");
    format!("{pre}\n{mid}\n{post}")
}

pub fn pprint_ast(ast : &Ast) -> (String, String) {
    let cfg = PrintConfig::default();
    let cpp = pprint_vec(&ast.host_entry, "\n", &mut cfg.clone());
    let cu_dev = pprint_vec(&ast.kernels, "\n", &mut cfg.clone());
    let cu_host = pprint_vec(&ast.host_stage, "\n", &mut cfg.clone());
    let module_decls = declare_torch_modules(&ast.host_entry);
    ( format!("{MACROS}\n{cpp}\n{module_decls}")
    , format!("{cu_dev}\n{cu_host}") )
}
