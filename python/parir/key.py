import hashlib

def generate_function_key(code, opts):
    s = f"{code}+{opts.includes}+{opts.libs}+{opts.extra_flags}"
    h = hashlib.new("sha256")
    h.update(s.encode("ascii"))
    return h.hexdigest()

def print_argument_key(arg):
    from .buffer import Buffer
    from .buffer import print_type
    if isinstance(arg, Buffer):
        if len(arg.shape) == 0:
            v = arg.numpy()
            return f"{v};{v.dtype}"
        else:
            shape_strs = [f"{s}" for s in arg.shape]
            return f"""[{",".join(shape_strs)};{arg.dtype}]"""
    else:
        return f"{arg}"

def print_arguments_key(args):
    return "-".join([print_argument_key(arg) for arg in args])

def print_compile_options_key(opts):
    return str(opts)

def generate_fast_cache_key(ir_ast, args, opts):
    from .parir import print_ir_ast
    ir_key = print_ir_ast(ir_ast)
    args_key = print_arguments_key(args)
    opts_key = print_compile_options_key(opts)
    return f"{ir_key}+{args_key}+{opts_key}"
