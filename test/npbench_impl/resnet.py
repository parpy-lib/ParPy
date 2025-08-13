import prickle
import torch

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

@prickle.jit
def conv2d_kernel(inputs, weights, output, H_out, W_out, N, C_in, C_out, K):
    prickle.label('i')
    for i in range(H_out):
        prickle.label('j')
        for j in range(W_out):
            output[:,i,j,:] = 0.0
            for a in range(N):
                for b in range(K):
                    for c in range(K):
                        for d in range(C_in):
                            for e in range(C_out):
                                output[a,i,j,e] += inputs[a,i+b,j+c,d] * weights[b,c,d,e]

# Deep learning convolutional operator (stride = 1)
def conv2d(input, weights, opts, compile_only=False):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_in = input.shape[3]
    C_out = weights.shape[3]
    output = torch.empty((N, H_out, W_out, C_out), dtype=torch.float32)
    if compile_only:
        args = [input, weights, output, H_out, W_out, N, C_in, C_out, K]
        return prickle.print_compiled(conv2d_kernel, args, opts)
    conv2d_kernel(input, weights, output, H_out, W_out, N, C_in, C_out, K, opts=opts)
    return output


# Batch normalization operator, as used in ResNet
def batchnorm2d(x, eps=1e-5, ):
    mean = torch.mean(x, axis=0, keepdims=True)
    std = torch.std(x, axis=0, unbiased=False, keepdims=True)
    return (x - mean) / torch.sqrt(std + eps)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet(input, conv1, conv2, conv3, opts, compile_only=False):
    # Pad output of first convolution for second convolution
    padded = torch.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))

    if compile_only:
        return conv2d(input, conv1, opts, True)
    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1, opts)
    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2, opts)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3, opts)
    x = batchnorm2d(x)
    return relu(x + input)
