import prickle
import torch

@prickle.jit
def prickle_kernel(a, b, c, img_in, img_out, y1, y2, W, H):
    prickle.label('i')
    for i in range(W):
        y1[i, 0] = a[0] * img_in[i, 0]
        y1[i, 1] = a[0] * img_in[i, 1] + a[1] * img_in[i, 0] + b[0] * y1[i, 0]
    for j in range(2, H):
        prickle.label('i')
        y1[:, j] = (a[0] * img_in[:, j] + a[1] * img_in[:, j-1] +
                    b[0] * y1[:, j-1] + b[1] * y1[:, j-2])

    prickle.label('i')
    for i in range(W):
        y2[i, -1] = 0.0
        y2[i, -2] = a[2] * img_in[i, -1]
    for j in range(H-3, -1, -1):
        prickle.label('i')
        y2[:, j] = (a[2] * img_in[:, j+1] + a[3] * img_in[:, j+2] +
                    b[0] * y2[:, j+1] + b[1] * y2[:, j+2])

    prickle.label('i')
    prickle.label('j')
    img_out[:, :] = c[0] * (y1[:, :] + y2[:, :])

    prickle.label('j')
    for j in range(H):
        y1[0, j] = a[4] * img_out[0, j]
        y1[1, j] = a[4] * img_out[1, j] + a[5] * img_out[0, j] + b[0] * y1[0, j]
    for i in range(2, W):
        prickle.label('j')
        y1[i, :] = (a[4] * img_out[i, :] + a[5] * img_out[i-1, :] +
                    b[0] * y1[i-1, :] + b[1] * y1[i-2, :])

    prickle.label('j')
    for j in range(H):
        y2[W-1, j] = 0.0
        y2[W-2, j] = a[6] * img_out[W-1, j]
    for i in range(W-3, -1, -1):
        prickle.label('j')
        y2[i, :] = (a[6] * img_out[i+1, :] + a[7] * img_out[i+2, :] +
                    b[0] * y2[i+1, :] + b[1] * y2[i+2, :])

    prickle.label('i')
    prickle.label('j')
    img_out[:, :] = c[1] * (y1[:, :] + y2[:, :])

def deriche(alpha, img_in, opts):
    y1 = torch.empty_like(img_in)
    y2 = torch.empty_like(img_in)
    img_out = torch.empty_like(img_in)
    W, H = img_in.shape
    k = (1.0 - prickle.exp(-alpha)) * (1.0 - prickle.exp(-alpha)) / (
        1.0 + alpha * prickle.exp(-alpha) - prickle.exp(2.0 * alpha))
    a = torch.empty(8, dtype=y1.dtype)
    b = torch.empty(2, dtype=y1.dtype)
    c = torch.empty_like(b)
    a[0] = a[4] = k
    a[1] = a[5] = k * prickle.exp(-alpha) * (alpha - 1.0)
    a[2] = a[6] = k * prickle.exp(-alpha) * (alpha + 1.0)
    a[3] = a[7] = -k * prickle.exp(-2.0 * alpha)
    b[0] = 2.0**(-alpha)
    b[1] = -prickle.exp(-2.0 * alpha)
    c[0] = c[1] = 1.0
    prickle_kernel(a, b, c, img_in, img_out, y1, y2, W, H, opts=opts)
    return img_out
