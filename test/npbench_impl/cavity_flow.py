# Barba, Lorena A., and Forsyth, Gilbert F. (2018).
# CFD Python: the 12 steps to Navier-Stokes equations.
# Journal of Open Source Education, 1(9), 21,
# https://doi.org/10.21105/jose.00021
# TODO: License
# (c) 2017 Lorena A. Barba, Gilbert F. Forsyth.
# All content is under Creative Commons Attribution CC-BY 4.0,
# and all code is under BSD-3 clause (previously under MIT, and changed on March 8, 2018).

import prickle
import torch

@prickle.jit
def build_up_b(b, rho, dt, u, v, dx, dy):
    prickle.label('ny')
    prickle.label('nx')
    b[1:-1,
      1:-1] = (rho * (1.0 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx) +
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy)) -
                      ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2.0 * dx))**2.0 - 2.0 *
                      ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2.0 * dy) *
                       (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2.0 * dx)) -
                      ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2.0 * dy))**2.0))

@prickle.jit
def pressure_poisson(nit, p, pn, dx, dy, b):
    for q in range(nit):
        prickle.label('ny')
        prickle.label('nx')
        pn[:,:] = p[:,:]
        prickle.label('ny')
        prickle.label('nx')
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2.0 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2.0) /
                         (2.0 * (dx**2.0 + dy**2.0)) - dx**2.0 * dy**2.0 /
                         (2.0 * (dx**2.0 + dy**2.0)) * b[1:-1, 1:-1])

        prickle.label('ny')
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        prickle.label('nx')
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        prickle.label('ny')
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        prickle.label('nx')
        p[-1, :] = 0.0  # p = 0 at y = 2

@prickle.jit
def cavity_flow_kernel(nx, ny, nt, nit, u, un, v, vn, b, dt, dx, dy, p, pn, rho, nu):
    build_up_b(b, rho, dt, u, v, dx, dy)
    pressure_poisson(nit, p, pn, dx, dy, b)

    prickle.label('ny')
    prickle.label('nx')
    u[1:-1,
      1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
               (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy *
               (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2.0 * rho * dx) *
               (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu *
               (dt / dx**2.0 *
                (un[1:-1, 2:] - 2.0 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                dt / dy**2.0 *
                (un[2:, 1:-1] - 2.0 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    prickle.label('ny')
    prickle.label('nx')
    v[1:-1,
      1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx *
               (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy *
               (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2.0 * rho * dy) *
               (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu *
               (dt / dx**2.0 *
                (vn[1:-1, 2:] - 2.0 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                dt / dy**2.0 *
                (vn[2:, 1:-1] - 2.0 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    prickle.label('nx')
    u[0, :] = 0.0
    prickle.label('ny')
    u[:, 0] = 0.0
    prickle.label('ny')
    u[:, -1] = 0.0
    prickle.label('nx')
    u[-1, :] = 1.0  # set velocity on cavity lid equal to 1
    prickle.label('nx')
    v[0, :] = 0.0
    prickle.label('nx')
    v[-1, :] = 0.0
    prickle.label('ny')
    v[:, 0] = 0.0
    prickle.label('ny')
    v[:, -1] = 0.0

def cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu, opts):
    un = torch.empty_like(u)
    vn = torch.empty_like(v)
    b = torch.zeros((ny, nx), device=u.device)

    for n in range(nt):
        un = u.detach().clone()
        vn = v.detach().clone()
        pn = p.detach().clone()

        cavity_flow_kernel(
            nx, ny, nt, nit, u, un, v, vn, b, dt, dx, dy, p, pn, rho, nu,
            opts=opts
        )
