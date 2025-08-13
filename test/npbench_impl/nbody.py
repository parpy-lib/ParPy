# Adapted from https://github.com/pmocz/nbody-python/blob/master/nbody.py
# TODO: Add GPL-3.0 License

import prickle
import torch
"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

@prickle.jit
def getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r3, a, N):
    # matrix that stores all pairwise particle separations: r_j - r_i
    prickle.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    prickle.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r3[i,j] = dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0 + softening**2.0
        if inv_r3[i,j] > 0.0:
            inv_r3[i,j] **= (-1.5)

    # Compute and pack together the acceleration components in one go
    prickle.label('N')
    for i in range(0, N):
        prickle.label('reduce')
        a[i,0] = prickle.sum(G * (dx[i,:] * inv_r3[i,:]) * mass[:,0])
        prickle.label('reduce')
        a[i,1] = prickle.sum(G * (dy[i,:] * inv_r3[i,:]) * mass[:,0])
        prickle.label('reduce')
        a[i,2] = prickle.sum(G * (dz[i,:] * inv_r3[i,:]) * mass[:,0])

@prickle.jit
def getEnergy_kernel(pos, vel, mass, G, KE, PE, dx, dy, dz, inv_r, tmp, N):
    with prickle.gpu:
        prickle.label('i')
        KE[0] = prickle.sum(mass[:,0] * (vel[:,0]**2.0 + vel[:,1]**2.0 + vel[:,2]**2.0))
        KE[0] *= 0.5

    prickle.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        dx[i,j] = pos[j,0] - pos[i,0]
        dy[i,j] = pos[j,1] - pos[i,1]
        dz[i,j] = pos[j,2] - pos[i,2]

    prickle.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        inv_r[i,j] = prickle.sqrt(dx[i,j]**2.0 + dy[i,j]**2.0 + dz[i,j]**2.0)
        if inv_r[i,j] > 0.0:
            inv_r[i,j] = 1.0 / inv_r[i,j]

    prickle.label('N2')
    for ij in range(N*N):
        i = ij // N
        j = ij % N
        tmp[i,j] = -(mass[i,0] * mass[j,0]) * inv_r[i,j]

    with prickle.gpu:
        PE[0] = 0.0
        for i in range(N):
            prickle.label('reduce')
            for j in range(i+1, N):
                PE[0] += tmp[i,j]
        PE[0] *= G

@prickle.jit
def nbody_kernel(mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, acc, inv_r, tmp):
    getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N)
    getEnergy_kernel(pos, vel, mass, G, KE[0], PE[0], dx, dy, dz, inv_r, tmp, N)
    for i in range(1, Nt+1):
        # (1/2) kick
        prickle.label('N')
        prickle.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # drift
        prickle.label('N')
        prickle.label('_')
        pos[:,:] += vel[:,:] * dt

        # update accelerations
        getAcc_kernel(pos, mass, G, softening, dx, dy, dz, inv_r, acc, N)

        # (1/2) kick
        prickle.label('N')
        prickle.label('_')
        vel[:,:] += acc[:,:] * dt / 2.0

        # get energy of system
        getEnergy_kernel(pos, vel, mass, G, KE[i], PE[i], dx, dy, dz, inv_r, tmp, N)

def nbody(mass, pos, vel, N, Nt, dt, G, softening, opts, compile_only=False):
    # Convert to Center-of-Mass frame
    vel -= torch.mean(mass * vel, axis=0) / torch.mean(mass)

    # Allocate temporary data used within the megakernel
    N,_ = pos.shape
    # NOTE: We add a dummy dimension to KE and PE to ensure they are passed as
    # tensors to the underlying kernels, so that we can modify individual
    # elements within kernels.
    KE = torch.empty((Nt + 1, 1), dtype=pos.dtype)
    PE = torch.empty_like(KE)
    a = torch.empty((N, 3), dtype=pos.dtype)
    dx = torch.empty((N, N), dtype=pos.dtype)
    dy = torch.empty_like(dx)
    dz = torch.empty_like(dx)
    inv_r = torch.empty_like(dx)
    tmp = torch.empty_like(dx)

    if compile_only:
        args = [mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, a, inv_r, tmp]
        return prickle.print_compiled(nbody_kernel, args, opts)
    nbody_kernel(
        mass, pos, vel, N, Nt, dt, G, softening, KE, PE, dx, dy, dz, a, inv_r, tmp,
        opts=opts
    )
    return KE.reshape(Nt+1), PE.reshape(Nt+1)
