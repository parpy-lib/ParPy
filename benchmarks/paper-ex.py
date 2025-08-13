# Assumes this script runs from the root of the repository
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import prickle
import statistics
import torch

@prickle.jit
def sum_rows(x, out, N):
    prickle.label('outer')
    for i in range(N):
        prickle.label('inner')
        out[i] = prickle.sum(x[i,:])

def sum_rows_wrap(x, p):
    N, M = x.shape
    y = torch.zeros(N, dtype=torch.float32, device='cuda')
    sum_rows(x, y, N, opts=prickle.par(p))
    return y

def print_sum_rows(args, p):
    opts = prickle.par(p)
    opts.backend = prickle.CompileBackend.Cuda
    print(prickle.print_compiled(sum_rows, args, opts))

def print_versions(N, M):
    x = torch.randn((N, M), dtype=torch.float32)
    y = torch.tensor((N,), dtype=torch.float32)
    args = [x, y, N]

    p1 = { 'outer': prickle.threads(N) }
    print("Version #1:")
    print_sum_rows(args, p1)
    p2 = {
        'outer': prickle.threads(N),
        'inner': prickle.threads(1024),
    }
    print("\nVersion #2:")
    print_sum_rows(args, p2)
    p3 = {
        'outer': prickle.threads(N),
        'inner': prickle.threads(32 * 1024),
    }
    print("\nVersion #3:")
    print_sum_rows(args, p3)

def bench(fn, arg):
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)

    # Warmup
    r = fn(arg)
    torch.cuda.synchronize()
    # Skip if the function does not support the provided input.
    if r is None:
        return []
    else:
        expected = torch.sum(arg, dim=1)
        assert torch.allclose(expected, r, atol=1e-1, rtol=1e-2), f"{expected}\n{r}"

    # Benchmarking
    times = []
    for i in range(100):
        e1.record()
        fn(arg)
        e2.record()
        torch.cuda.synchronize()
        times += [e1.elapsed_time(e2)]
    torch.cuda.empty_cache()
    return times

# Parallelization of the outer loop, should only run each on a separate thread.
def version1(x):
    N, M = x.shape
    p = { 'outer': prickle.threads(N) }
    if N <= 1024:
        return sum_rows_wrap(x, p)
    else:
        return None

# Parallelize both loops, using at most one full block for the inner loop.
def version2(x):
    N, M = x.shape
    p = { 'outer': prickle.threads(N), 'inner': prickle.threads(1024) }
    return sum_rows_wrap(x, p)

# Parallelize both loops so that each thread processes one element.
def version3(x):
    N, M = x.shape
    p = { 'outer': prickle.threads(N), 'inner': prickle.threads(32 * 1024) }
    return sum_rows_wrap(x, p)

def print_times(times):
    if len(times) > 0:
        m = statistics.mean(times)
        s = statistics.stdev(times)
        print(f"{m:2f} ± {s:2f}")
    else:
        print("-")

def entry(version, N, M, t):
    return {'version': version, 'N': N, 'M': M, 'time': t}

def run_test(N, M):
    print(f"{N} {M}")
    x = torch.randn((N, M), dtype=torch.float32, device='cuda')
    result = []
    t1 = bench(version1, x)
    print_times(t1)
    result += [entry(1, N, M, t) for t in t1]
    t2 = bench(version2, x)
    print_times(t2)
    result += [entry(2, N, M, t) for t in t2]
    t3 = bench(version3, x)
    print_times(t3)
    result += [entry(3, N, M, t) for t in t3]
    return result

# Print the code of each version for N=128 and M=1024
print_versions(128, 1024)

# Run the benchmarks for many combinations of N and M.
fig, axs = plt.subplots(layout="constrained")
N = 10
M_values = [10**M for M in range(1, 9)]
results = []
for M in M_values:
    results += run_test(N, M)
ofs = np.arange(len(M_values))

def version_to_label(v):
    if v == 1:
        return "(b)"
    elif v == 2:
        return "(c)"
    elif v == 3:
        return "(d)"
    else:
        print(f"Unknown version: {v}")
        exit(1)

# Produce a plot for the results using a particular value of N
df = pd.DataFrame(results)
#pd.to_csv("example.csv", index=False)
n_res = df[df["N"] == N]
width = 0.25
for idx, version in enumerate([1, 2, 3]):
    m_res = n_res[n_res["version"] == version].groupby("M")
    rts = m_res["time"].mean()
    std = m_res["time"].std()
    label = version_to_label(version)
    rects = axs.bar(ofs + width * idx, rts, width, label=label)
    axs.errorbar(ofs + width * idx, rts, yerr=std, fmt="o", capsize=3, color="black")
m_labels = [f"$10^{int(math.log10(M))}$" for M in M_values]
axs.set_xlabel("M", fontsize=16)
axs.set_xticks(ofs + width, m_labels)
axs.set_yscale("log")
axs.set_ylabel("Execution time (ms)", fontsize=16)
axs.tick_params(axis='both', which='major', labelsize=16)
axs.tick_params(axis='both', which='minor', labelsize=14)
axs.legend(loc="upper left", fontsize=16)
fig.savefig(f"mot-ex-perf.pdf", bbox_inches="tight", pad_inches=0.05)
