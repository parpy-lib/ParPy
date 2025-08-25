import parpy
import torch

@parpy.jit
def kernel_wrap(ex, ey, hz, _fict_, TMAX):
    for t in range(TMAX):
        parpy.label('j')
        ey[0, :] = _fict_[t]

        parpy.label('i')
        parpy.label('j')
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])

        parpy.label('i')
        parpy.label('j')
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])

        parpy.label('i')
        parpy.label('j')
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])

def fdtd2d(TMAX, ex, ey, hz, _fict_, opts, compile_only=False):
    NX, NY = ex.shape
    if compile_only:
        args = [ex, ey, hz, _fict_, TMAX]
        return parpy.print_compiled(kernel_wrap, args, opts)
    kernel_wrap(ex, ey, hz, _fict_, TMAX, opts=opts)

