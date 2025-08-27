import hashlib

def _generate_function_key(code, opts):
    s = f"{code}+{opts.includes}+{opts.libs}+{opts.extra_flags}"
    h = hashlib.new("sha256")
    h.update(s.encode("ascii"))
    return h.hexdigest()

def _print_argument_key(arg):
    from .buffer import Buffer
    if isinstance(arg, dict):
        s = []
        for k, v in sorted(arg.items()):
            s.append(f"{k}: {print_argument_key(v)}")
        return """{{{", ".join(s)}}}"""
    elif isinstance(arg, Buffer):
        if len(arg.shape) == 0:
            v = arg.numpy()
            return f"{v};{v.dtype}"
        else:
            shape_strs = [f"{s}" for s in arg.shape]
            return f"""[{",".join(shape_strs)};{arg.dtype}]"""
    else:
        return f"{arg}"

def _print_arguments_key(args):
    return "-".join([_print_argument_key(arg) for arg in args])

def _print_compile_options_key(opts):
    return str(opts)

def _generate_fast_cache_key(ir_ast, args, opts):
    from .parpy import print_ast
    ir_key = _print_ast(ir_ast)
    args_key = _print_arguments_key(args)
    opts_key = _print_compile_options_key(opts)
    return f"{ir_key}+{args_key}+{opts_key}"
