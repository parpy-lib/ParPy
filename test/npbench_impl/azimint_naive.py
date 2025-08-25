# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import parpy
import pytest
import torch

@parpy.jit
def parpy_kernel_32bit(data, radius, res, rmax, npt, N):
    parpy.label('ix')
    for i in range(N):
        rmax[0] = parpy.max(rmax[0], radius[i])
    parpy.label('i')
    for i in range(npt):
        r1 = rmax[0] * parpy.float32(i) / parpy.float32(npt)
        r2 = rmax[0] * parpy.float32(i+1) / parpy.float32(npt)
        c = 0.0
        parpy.label('j')
        for j in range(N):
            c = c + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0)
        s = 0.0
        parpy.label('j')
        for j in range(N):
            s = s + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0) * data[j]
        # Modified to work with smaller values
        if c == 0.0:
            res[i] = 0.0
        else:
            res[i] = s / c

@parpy.jit
def parpy_kernel(data, radius, res, rmax, npt, N):
    parpy.label('ix')
    for i in range(N):
        rmax[0] = parpy.max(rmax[0], radius[i])
    parpy.label('i')
    for i in range(npt):
        r1 = rmax[0] * parpy.float64(i) / parpy.float64(npt)
        r2 = rmax[0] * parpy.float64(i+1) / parpy.float64(npt)
        c = 0.0
        parpy.label('j')
        for j in range(N):
            c = c + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0)
        s = 0.0
        parpy.label('j')
        for j in range(N):
            s = s + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0) * data[j]
        # Modified to work with smaller values
        if c == 0.0:
            res[i] = 0.0
        else:
            res[i] = s / c

def azimint_naive(data, radius, npt, opts, compile_only=False):
    rmax = torch.empty((1,), dtype=data.dtype)
    res = torch.zeros(npt, dtype=data.dtype)
    N, = data.shape
    args = [data, radius, res, rmax, npt, N]
    if data.dtype == torch.float32:
        if compile_only:
            return parpy.print_compiled(parpy_kernel_32bit, args, opts)
        parpy_kernel_32bit(data, radius, res, rmax, npt, N, opts=opts)
    else:
        if compile_only:
            return parpy.print_compiled(parpy_kernel, args, opts)
        parpy_kernel(data, radius, res, rmax, npt, N, opts=opts)
    return res
