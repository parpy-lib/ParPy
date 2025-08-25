# ParPy

ParPy is a Python library performing GPU acceleration of Python functions containing for-loops based on simple annotations provided by the user when calling an annotated function, with a high degree of control and customizability.

## Quick Start

### Requirements

The quick start guide described below assumes [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) has been installed and that the `conda` command is available.

### Setup

Clone the repository and create a conda environment by running the following command from the root of the repository
```bash
conda env create -f benchmarks/<backend>-env.yml
```
to set up an environment adapted for your target backend, `<backend>`. The currently supported backends are `cuda` and `metal`. The environment includes a recent installation of the Rust compiler (in which the majority of the ParPy compiler is implemented), and other libraries that we compare against in our benchmarks (both in the `benchmarks` directory of this repository and our modified [NPBench suite](https://github.com/larshum/npbench)).

An advanced user can install the `minimal` environment, which includes the minimal set of requirements needed to install ParPy. Installed in this way, ParPy can be used to generate readable source code (CUDA C++ or Metal C++), but it cannot run any functions. The test runner automatically skips tests that run code for unsupported backends.

After the setup finishes, activate the environment using
```bash
conda activate bench-env
```

The environment includes all dependencies needed to run all tests and benchmarks for the selected platform. To exit the environment, use `conda deactivate`. The remaining setup assumes commands run from within the Conda environment.

#### CUDA

Install ParPy by running
```bash
pip install ".[cuda]"
```
from the root of the repository. This installs additional packages required exclusively by the CUDA backend.

#### Metal

Install ParPy by running
```bash
pip install .
```
from the root of the repository. After installing ParPy, users have to download the [Metal-cpp](https://developer.apple.com/metal/cpp/) headers for their MacOS version. Following this, the user must set the `METAL_CPP_HEADER_PATH` to the path of the `metal-cpp` directory, so ParPy knows where to find it.

### Running tests

To run tests, ParPy must be installed using `pip install ".[test]"` (or `".[cuda,test]"` for the CUDA backend) to include testing dependencies. We run quick unit tests for the internal behavior of the native compiler using
```bash
cargo test
```

Further, we can run a more comprehensive set of integration tests using
```bash
pytest
```
These tests ensure the result of our compiler behaves correctly with respect to expected output. We either compare against a known output or against existing library functions in PyTorch. These tests take about 5-15 minutes depending on the target backend.

### Minimal Example

Below is a minimal example implementing AXPY (`a` plus `x` times `y`, where `x` and `y` are vectors). To parallelize using ParPy, we first decorate the target function using `@parpy.jit` to indicate that we want it to be just-in-time (JIT) compiled. Note that we explicitly provide the number of rows `N` of `x` and `y`, as the dimensions of arrays are not accessible within a ParPy function. A more comprehensive version of this example, including explainingcomments, is found at `examples/axpy.py`.

To provide control over parallelization, we annotate the function with a label `N`, by using the `parpy.label` statement. When we call the function, we provide a keyword argument `opts` specifying compilation options, including how to parallelize the function. The `parpy.par` function uses the default options with the a provided dicitionary specifying the parallelization in terms of labels of the function. In this case, we specify that the for-loop (labelled `N`) should be parallelized across `128` threads. ParPy automatically determines how to map the one-dimensional threads specified across any number of for-loops to a GPU grid.

```python
import numpy as np
import parpy

@parpy.jit
def axpy(a, x, y):
    parpy.label('N')
    for i in range(N):
        y[i] = a + x[i] * y[i]

N = 1024
x = np.ones(N)
y = np.zeros_like(x)
a = 2.5
axpy(a, x, y, N, opts=parpy.par({'N': parpy.threads(128)}))
assert np.allclose(y, a)
```

## Motivation

ParPy is a Python framework for automatic GPU parallelization of for-loops and slices. The key feature of ParPy is the ability to control not only what to parallelize but how to parallelize it.

ParPy can operate on arrays from typical frameworks (e.g., NumPy, PyTorch, and CuPy). If provided an array allocated on the GPU (providing the `__cuda_array_interface__` attribute), its memory will be automatically reused on the GPU without copying data. Otherwise, if an array is allocated in CPU memory, it is automatically copied to and from memory allocated on the backend device. To avoid this overhead, users can explicitly copy memory to the device

ParPy can operate on any arrays or tensors, and is highly interoperable with existing libraries. If an argument is an array that implements the `__cuda_array_interface__` attribute (e.g., a Torch tensor allocated on the GPU or CuPy NDArrays), ParPy will reuse the existing GPU memory without copying it. We can also provide an array that implements the `__array_interface__` or `__array__` attributes (e.g., a Torch tensor on the CPU or a NumPy array), in which case ParPy will copy this to a temporary buffer which will be allocated for the duration of the called function.

Importantly, the aim of ParPy is not to replace such libraries but to be a complement. Users can easily write specialized GPU kernels for algorithms that are not supported in libraries to avoid the Python interpreter overhead and gain better control over parallelization.

The ParPy compiler produces CUDA C++ code. The mapping from a ParPy function to the CUDA C++ is intentionally relatively straightforward; the aim is that a user should be able to identify how the for-loops of the original function are mapped to the GPU by looking at the CUDA C++ code. Advanced users can retrieve the CUDA C++ code and manually adjust it to use features inaccessible from Python (e.g., CUDA intrinsics or external libraries) and re-compile it using the ParPy API (assuming no changes are made to the generated C API).

## Example

We start with a regular Python function for computing the row-wise sum of a two-dimensional Torch tensor `x` and storing the result in `y`:
```python
def sum_rows(x, y, N, M):
  for i in range(N):
    for j in range(M):
      y[i] += x[i,j]
```

The iterations of the outer for-loop over `i` are independent because we write the summation result to distinct indices of the output tensor `y`. Therefore, this loop can be safely parallelized. Parallelization in ParPy is performed in two separate steps. The first step is to annotate the function, to indicate it should be JIT compiled and to associate labels with the statements we wish to parallelize:
```python
import parpy

@parpy.jit
def sum_rows(x, y, N, M):
  parpy.label('N')
  for i in range(N):
    for j in range(M):
      y[i] += x[i,j]
```

We use the decorator `@parpy.jit` to indicate the function should be JIT compiled. When the Python interpreter processes this definition, the ParPy compiler translates the Python code to an untyped Python-like representation. We use the statement `parpy.label('N')` to associate the label `N` with the outer for-loop. The second step of parallelization occurs when calling the function. At this point, we can determine how to parallelize the function, based on the arguments we provide. For instance, if the original call to `sum_rows` is:
```python
sum_rows(x, y, N, M)
```
we update it by providing an options object via the `opts` keyword. We can conveniently construct an option using the `parpy.par` function, which provides the default options along with the provided parallel specification (in the form of a Python dict).
```python
p = {'N': parpy.threads(N)}
sum_rows(x, y, N, M, opts=parpy.par(p))
```

When the `sum_rows` function is called, the ParPy compiler will generate GPU code based on the untyped Python-like representation from the first stage and the provided arguments. Our provided options specify that the compiler should generate CUDA code (the default target), and that any statement associated with the label `N` is parallelized across `N` threads. The [Parallelization](#parallelization) section describes how the compiler maps these threads to the GPU grid.

### Parallel Reduction

The summation performed in the inner loop of the `sum_rows` is an example of a [reduction](https://en.wikipedia.org/wiki/Fold_%28higher-order_function%29), which can be parallelized. To do this in ParPy, we first annotate the inner for-loop:
```python
@parpy.jit
def sum_rows(x, y, N, M):
  parpy.label('N')
  for i in range(N):
    parpy.label('M')
    for j in range(M):
      y[i] += x[i,j]
```

Second, we specify how to parallelize the reduction. The compiler currently does not automatically identify reductions, so we need to specify that we want it to perform a parallel reduction in the parallelization specification:
```python
p = {'N': parpy.threads(N), 'M': parpy.threads(M).reduce()}
sum_rows(x, y, N, M, opts=parpy.par(p))
```

Alternatively, we can use the reduction operator in combination with slicing (described in the following section), in which case the `parpy.reduce()` specifier is inferred by the compiler.

### More Examples

The [test](/test) directory contains several programs using parallelization. The above example is based on the code in [test_reduce.py](test/test_reduce.py).

## Slicing

The ParPy compiler supports a restricted form of slicing, where each slice dimension of a statement is mapped to a separate for-loop, starting from the left. For example, we can use slices to define the `sum_rows` more succinctly as
```python
@parpy.jit
def sum_rows(x, y, N, M):
  parpy.label('N')
  parpy.label('M')
  y[:] = parpy.sum(x[:,:], axis=1)
```

The `parpy.sum` operation is an example of a slice reduction operation, which reduces the number of dimensions. In this case, we use the `axis` keyword argument to specify a dimension to reduce over. When the `axis` argument is omitted, the compiler generates a reduction over all dimensions. For example,
```python
y[0] = parpy.sum(x[:,:])
```
would reduce over both dimensions of `x` and store the result in `y`. Note that all tensor dimensions must be explicitly listed in a slice operation.

The ParPy compiler will automatically translate slice operations to semantically equivalent explicit for-loops. This happens quite early in the compiler pipeline, allowing us to drastically simplify later stages of the compiler.

### Limitations

Slice operations that require materialization of a new tensor are not supported:
```
a = x[i,:]
```

Instead, we can achieve this by pre-allocating `a` outside the function, passing it as an argument, and writing to it with an explicit slice pattern:
```
a[:] = x[i,:]
```

## Arguments

The arguments passed to a Python function can be Python `int`s or `float`s, a form of GPU array kind, or a dict with string keys mapping to non-dict values (which must be among the supported values). A Python `int` is interpreted as a 64-bit integer, and similarly, a `float` is interpreted as a 64-bit floating-point number. A GPU array with an empty shape (e.g., `torch.tensor(10, dtype=torch.int16)`) is treated as a scalar value but with a custom type specified via its `dtype`.

The ParPy compiler automatically specializes the generated code based on the provided scalar values and on the shapes of all provided (non-empty) tensors. In situations where we do not want the compiler to specialize on an argument, we wrap it in a tensor of shape `(1,)` (e.g., `torch.tensor([10], dtype=torch.int16)`). For instance, such a situation can occur when we want to avoid excessive JIT compilation overhead when calling a ParPy function from a for-loop (due to specializing on the iteration variable, which changes in each iteration).

See the [test_dict_args.py](test/test_dict_args.py) and [test_forward.py](test/test_forward.py) files for examples of how to use dictionary arguments.

## Debugging

All functions provided via ParPy should behave equivalently when executed by the Python interpreter and when compiled to GPU code (barring compiler bugs and missing reduction annotations). To avoid having to remove the `@parpy.jit` decorator when a user wants to debug using the Python interpreter, we can set the `seq` keyword to `True` in the call to the function, as in:
```python
sum_rows(x, y, N, M, seq=True)
```

An advanced or curious user can set the `debug` keyword argument to `True` in a call. When debugging is enabled in the compiler, it will output the AST (abstract syntax tree, a representation of the code) after all major transformations within the compiler, along with the time spent in the compiler up to that stage.

## Parallelization

The approach used to map parallelism to CUDA is relatively straightforward. The innermost level of parallelism is mapped to CUDA threads (up to 1024) or threads and blocks. Any parallel outer for-loops are always mapped to CUDA blocks. The outermost parallel for-loop maps to a CUDA kernel and any sequential loops outside it result in a for-loop running on the CPU side and launching one or more CUDA kernels.

Implementing reductions efficiently require all 32 threads of a GPU warp to be participating. If a user requests `n` threads in the innermost loop, the compiler will automatically increase this to the next number divisible by 32 (by setting it to `((n + 31) / 32) * 32`). The parallelism requested in outer parallel loops will never be adjusted by the compiler.

Generally, the compiler assumes all statements on the same level of nesting (in terms of parallelism) have the same amount of parallelism. Assume we have written code such as:
```python
parpy.label('N')
for i in range(N):
  parpy.label('M1')
  for j in range(M):
    ...
  parpy.label('M2')
  for j in range(M):
    ...
```

The compiler trivially supports the case where the `M1` and `M2` labels are mapped to the same number of threads. However, if they map to different numbers of threads, the current implementation of the compiler automatically transforms the code to
```python
parpy.label('N')
for i in range(N):
  parpy.label('M1')
  for j in range(M):
    ...
parpy.label('N')
for i in range(N):
  parpy.label('M2')
  for j in range(M):
    ...
```
such that we end up with two separate kernels with a consistent amount of parallelism.

Note that the current compiler version does not do the above transformation when `M1` and `M2` are not equal, and both of them map to at most 1024 threads (i.e., they map to a single CUDA block). In this case, the compiler would either have to generate code where some threads are idle for one of the loops, or it could split them up into two parallel loops, both of which introduce extra overhead in a situation where this may have a significant performance impact.

### Sequential Code on the GPU

As running sequential code on the GPU is significantly slower than running it on the CPU, sequential for-loops outside of the first parallel for-loop on the CPU. As arguments are allocated on the GPU, we disallow any non-loop statements outside parallel loops. While supporting operations on the CPU would be possible, this would be inefficient as it may require copying data between the CPU and the GPU. Therefore, ParPy disallows this altogether.

Consider the below attempt at a one-dimensional summation function in ParPy:
```python
import parpy
@parpy.jit
def sum(x, y, N):
  y[0] = 0.0
  parpy.label('N')
  for i in range(N):
    y[0] += x[i]
```

The compiler rejects this code because the initial assignment to `y[0]` occurs outside a parallel for-loop. In this case, we are willing to accept the (minimal) overhead of running this statement sequentially on the GPU. To achieve this, we could wrap the whole function body in a for-loop with one iteration, annotate it, and specify that its label should map to one thread. While this would work, it is verbose and requires modifying the code in multiple places. As a short-hand for this, we can wrap the function body in the `parpy.gpu` context:
```
import parpy
@parpy.jit
def sum(x, y, N):
  with parpy.gpu:
    y[0] = 0.0
    parpy.label('N')
    for i in range(N):
      y[0] += x[i]
```

## Working Around Limitations

Many parts of a typical Python program are inherently sequential or include code that is difficult to parallelize efficiently. For this reason, ParPy (intentionally) does not support many of Python's features. ParPy is designed to be used in small functions with much potential parallelism, and its performance is critical. As ParPy can operate on, for instance, Torch tensors, we can opt to use the Torch library for problems where it already provides a highly efficient implementation.

There are many low-level concepts that are critical to achieving high performance, but that are not immediately accessible from ParPy, such as [tensor cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) and [shared memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)) in CUDA. In such situations, we can retrieve the generated code from ParPy, manually modify the code, and pass the updated code to ParPy to produce a function usable from Python. The ParPy API exposes two functions for this purpose.

For instance, consider the `sum_rows` function presented as an example above. To print the resulting CUDA code from compiling this function, we can use
```python
p = {'N': parpy.threads(32), 'M': parpy.threads(128).reduce()}
print(parpy.print_compiled(sum_rows, [x, y, N, M], opts=parpy.par(p)))
```

We pass a function to be compiled (this function does _not_ have to be decorated with `@parpy.jit`), a list of the arguments to be passed to the function, and the parallelization dictionary (which we would pass to the `parallelize` keyword argument in a regular function call). The resulting string is printed to stdout using `print`.

An advanced user can store the output code in a file and modify it manually. This is useful for performing operations not supported by the ParPy compiler (as discussed above) or debugging the generated code. Assuming the modified code of the function `sum_rows` is stored as a string `code`, we can compile it using
```python
fn = parpy.compile_string("sum_rows", code, opts=parpy.par({}))
```

where we can provide include paths, library paths, and other flags to be passed to the underlying target compiler via the options object. The resulting function `fn` is a callable Python function expecting the same type of arguments as the original `sum_rows` function. The two required arguments are the name of the function and the CUDA C++ code (as a string).
