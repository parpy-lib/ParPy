import prickle
import torch

@prickle.jit
def kernel_wrap(ex, ey, hz, _fict_, TMAX):
    for t in range(TMAX):
        prickle.label('j')
        ey[0, :] = _fict_[t]

        prickle.label('i')
        prickle.label('j')
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])

        prickle.label('i')
        prickle.label('j')
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])

        prickle.label('i')
        prickle.label('j')
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])

def fdtd2d(TMAX, ex, ey, hz, _fict_, opts):
    NX, NY = ex.shape
    kernel_wrap(ex, ey, hz, _fict_, TMAX, opts=opts)

