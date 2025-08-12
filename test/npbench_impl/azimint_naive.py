# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import prickle
import pytest
import torch

@prickle.jit
def prickle_kernel_32bit(data, radius, res, rmax, npt, N):
    prickle.label('ix')
    for i in range(N):
        rmax[0] = prickle.max(rmax[0], radius[i])
    prickle.label('i')
    for i in range(npt):
        r1 = rmax[0] * prickle.float32(i) / prickle.float32(npt)
        r2 = rmax[0] * prickle.float32(i+1) / prickle.float32(npt)
        c = 0.0
        prickle.label('j')
        for j in range(N):
            c = c + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0)
        s = 0.0
        prickle.label('j')
        for j in range(N):
            s = s + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0) * data[j]
        # Modified to work with smaller values
        if c == 0.0:
            res[i] = 0.0
        else:
            res[i] = s / c

@prickle.jit
def prickle_kernel(data, radius, res, rmax, npt, N):
    prickle.label('ix')
    for i in range(N):
        rmax[0] = prickle.max(rmax[0], radius[i])
    prickle.label('i')
    for i in range(npt):
        r1 = rmax[0] * prickle.float64(i) / prickle.float64(npt)
        r2 = rmax[0] * prickle.float64(i+1) / prickle.float64(npt)
        c = 0.0
        prickle.label('j')
        for j in range(N):
            c = c + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0)
        s = 0.0
        prickle.label('j')
        for j in range(N):
            s = s + (1.0 if r1 <= radius[j] and radius[j] < r2 else 0.0) * data[j]
        # Modified to work with smaller values
        if c == 0.0:
            res[i] = 0.0
        else:
            res[i] = s / c

def azimint_naive(data, radius, npt, opts):
    rmax = torch.empty((1,), dtype=data.dtype)
    res = torch.zeros(npt, dtype=data.dtype)
    N, = data.shape
    print(data)
    print(radius)
    print(npt)
    if data.dtype == torch.float32:
        prickle_kernel_32bit(data, radius, res, rmax, npt, N, opts=opts)
    else:
        prickle_kernel(data, radius, res, rmax, npt, N, opts=opts)
    print(res)
    print(data.dtype)
    return res
