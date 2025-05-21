import parir

# In this function, we define short-hand functions for specifying the compile
# options to be passed to the JIT compiler. The 'seq_opts' function ensures the
# code runs sequentially in the Python interpreter, while the 'par_opts'
# function runs with the given parallelization specification and (importantly)
# disables caching to prevent bugs in tests.

def seq_opts():
    opts = parir.CompileOptions()
    opts.seq = True
    return opts

def par_opts(p):
    opts = parir.CompileOptions()
    opts.parallelize = p
    opts.cache = False
    return opts
