import prickle
import torch

# Sample constants
BET_M = 0.5
BET_P = 0.5

@prickle.jit
def vadv_prickle(
    utens_stage, u_stage, wcon, u_pos, utens, dtr_stage, ccol, dcol, data_col, I, J, K,
    # Extra arguments we need to pre-allocate outside and pass along
    BET_M, BET_P, gav, gcv, as_, cs, acol, bcol, correction_term, divided, datacol
):
    for k in range(1):
        prickle.label('I')
        for i in range(I):
            prickle.label('J')
            for j in range(J):
                gcv[i,j] = 0.25 * (wcon[i+1, j, k + 1] + wcon[i, j, k + 1])
                cs[i,j] = gcv[i,j] * BET_M

                ccol[i, j, k] = gcv[i,j] * BET_P
                bcol[i,j] = dtr_stage - ccol[i, j, k]

                # update the d column
                correction_term[i,j] = -cs[i,j] * (u_stage[i, j, k + 1] - u_stage[i, j, k])
                dcol[i, j, k] = (dtr_stage * u_pos[i, j, k] + utens[i, j, k] +
                                 utens_stage[i, j, k] + correction_term[i,j])

                # Thomas forward
                divided[i,j] = 1.0 / bcol[i,j]
                ccol[i, j, k] = ccol[i, j, k] * divided[i,j]
                dcol[i, j, k] = dcol[i, j, k] * divided[i,j]

    for k in range(1, K - 1):
        prickle.label('I')
        for i in range(I):
            prickle.label('J')
            for j in range(J):
                gav[i,j] = -0.25 * (wcon[i+1, j, k] + wcon[i, j, k])
                gcv[i,j] = 0.25 * (wcon[i+1, j, k + 1] + wcon[i, j, k + 1])

                as_[i,j] = gav[i,j] * BET_M
                cs[i,j] = gcv[i,j] * BET_M

                acol[i,j] = gav[i,j] * BET_P
                ccol[i, j, k] = gcv[i,j] * BET_P
                bcol[i,j] = dtr_stage - acol[i,j] - ccol[i, j, k]

                # update the d column
                correction_term[i,j] = (
                    -as_[i,j] * (u_stage[i, j, k - 1] -
                                 u_stage[i, j, k]) - cs[i,j] * (
                                     u_stage[i, j, k + 1] - u_stage[i, j, k]))
                dcol[i, j, k] = (dtr_stage * u_pos[i, j, k] + utens[i, j, k] +
                                 utens_stage[i, j, k] + correction_term[i,j])

                # Thomas forward
                divided[i,j] = 1.0 / (bcol[i,j] - ccol[i, j, k - 1] * acol[i,j])
                ccol[i, j, k] = ccol[i, j, k] * divided[i,j]
                dcol[i, j, k] = (dcol[i, j, k] - (dcol[i, j, k - 1]) * acol[i,j]) * divided[i,j]

    for k in range(K - 1, K):
        prickle.label('I')
        for i in range(I):
            prickle.label('J')
            for j in range(J):
                gav[i,j] = -0.25 * (wcon[i+1, j, k] + wcon[i, j, k])
                as_[i,j] = gav[i,j] * BET_M
                acol[i,j] = gav[i,j] * BET_P
                bcol[i,j] = dtr_stage - acol[i,j]

                # update the d column
                correction_term[i,j] = -as_[i,j] * (u_stage[i, j, k - 1] - u_stage[i, j, k])
                dcol[i, j, k] = (dtr_stage * u_pos[i, j, k] + utens[i, j, k] +
                        utens_stage[i, j, k] + correction_term[i,j])

                # Thomas forward
                divided[i,j] = 1.0 / (bcol[i,j] - ccol[i, j, k - 1] * acol[i,j])
                dcol[i, j, k] = (dcol[i, j, k] - (dcol[i, j, k - 1]) * acol[i,j]) * divided[i,j]

    for k in range(K - 1, K - 2, -1):
        prickle.label('I')
        for i in range(I):
            prickle.label('J')
            for j in range(J):
                datacol[i,j] = dcol[i, j, k]
                data_col[i,j] = datacol[i,j]
                utens_stage[i, j, k] = dtr_stage * (datacol[i,j] - u_pos[i, j, k])

    for k in range(K - 2, -1, -1):
        prickle.label('I')
        for i in range(I):
            prickle.label('J')
            for j in range(J):
                datacol[i,j] = dcol[i, j, k] - ccol[i, j, k] * data_col[i, j]
                data_col[i,j] = datacol[i,j]
                utens_stage[i, j, k] = dtr_stage * (datacol[i,j] - u_pos[i, j, k])


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def vadv(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage, opts):
    I, J, K = utens_stage.shape
    ccol = torch.empty((I, J, K), dtype=utens_stage.dtype)
    dcol = torch.empty((I, J, K), dtype=utens_stage.dtype)
    data_col = torch.empty((I, J), dtype=utens_stage.dtype)

    # Extra allocations
    gav = torch.empty_like(data_col)
    gcv = torch.empty_like(gav)
    as_ = torch.empty_like(gav)
    cs = torch.empty_like(gav)
    acol = torch.empty_like(gav)
    bcol = torch.empty_like(gav)
    correction_term = torch.empty_like(gav)
    divided = torch.empty_like(gav)
    datacol = torch.empty_like(gav)

    vadv_prickle(
        utens_stage, u_stage, wcon, u_pos, utens, dtr_stage, ccol, dcol,
        data_col, I, J, K, BET_M, BET_P, gav, gcv, as_, cs, acol, bcol,
        correction_term, divided, datacol,
        opts=opts
    )
