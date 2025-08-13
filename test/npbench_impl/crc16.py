import prickle
import torch

@prickle.jit
def crc16_kernel(data, poly, N, out):
    with prickle.gpu:
        crc = prickle.int32(0xFFFF)
        for j in range(N):
            b = data[j]
            cur_byte = 0xFF & b
            for _ in range(8):
                if (crc & 0x0001) ^ (cur_byte & 0x0001):
                    crc = (crc >> 1) ^ poly
                else:
                    crc = crc >> 1
                cur_byte = cur_byte >> 1
        crc = (~crc & 0xFFFF)
        crc = (crc << 8) | ((crc >> 8) & 0xFF)
        out[0] = crc & 0xFFFF

def crc16(data, opts, compile_only=False):
    poly = 0x8408
    N, = data.shape
    out = torch.empty(1, dtype=torch.int32)
    if compile_only:
        args = [data, poly, N, out]
        return prickle.print_compiled(crc16_kernel, args, opts)
    crc16_kernel(data, poly, N, out, opts=opts)
    return int(out[0])
