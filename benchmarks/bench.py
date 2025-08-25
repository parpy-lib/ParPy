import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import contextlib
import io
import numpy as np
import os
import pandas as pd
import ssgetpy
import subprocess
import sys
from tqdm import tqdm

import common

class obj(object):
    pass

# Record the number of times each framework failed due to running out of memory
# (OOM) or failing in another way (e.g., due to a floating-point exception).
bench_ooms = {}
bench_fails = {}

def incr(dst, key):
    if key in dst:
        dst[key] += 1
    else:
        dst[key] = 1

def launch_bench(benchmark, config_args, timeout_s=1800, fn=None):
    cmds = ["python3", f"{benchmark}.py"] + config_args
    err_msgs = []
    if fn is not None:
        _stdout = io.StringIO()
        _stderr = io.StringIO()
        with contextlib.redirect_stdout(_stdout), contextlib.redirect_stderr(_stderr):
            code = fn()
            res = obj()
            out = _stdout.getvalue()
            err = _stderr.getvalue()
            r = code
    else:
        try:
            res = subprocess.run(cmds, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            err_msgs += [f"Benchmark {benchmark} timed out"]
            return
        out = res.stdout.decode('ascii')
        err = res.stderr.decode('ascii')
        r = res.returncode
    if r != 0:
        framework = config_args[0]
        if r == 34:
            incr(bench_ooms, framework)
        else:
            err_msgs += [f"Benchmark {benchmark} failed with exit code {r}"]
            incr(bench_fails, framework)
    if len(out) != 0:
        with open(f"{benchmark}.stdout", "a") as f:
            f.write(f"{config_args}\n{out}\n\n")
    if len(err) != 0:
        with open(f"{benchmark}.stderr", "a") as f:
            err = ''.join([e + "\n" for e in err_msgs]) + err
            f.write(f"{config_args}\n{err}\n\n")

def clear_log_output(benchmark):
    with open(f"{benchmark}.stdout", "w+") as f:
        pass
    with open(f"{benchmark}.stderr", "w+") as f:
        pass

def print_mean_pm_stddev(times):
    if len(times) == 0:
        return "-"
    else:
        t = np.mean(times)
        s = np.std(times)
        return f"{t:.1f} ± {s:.1f}"

def forward_config_id(config):
    if config == 1:
        return "single-block"
    elif config == 2:
        return "one-to-one"
    elif config == 3:
        return "tuned"

def find_col_maxlen(rows, col):
    return np.max([len(row[col]) for row in rows])

def print_aligned_columns(rows):
    nrows = len(rows)
    ncols = len(rows[0])
    maxlens = [find_col_maxlen(rows, i) for i in range(ncols)]
    for row in rows:
        print(" ".join(col.ljust(maxlen) for col, maxlen in zip(row, maxlens)))

def print_forward_output(csv_file, frameworks, configurations, kmer):
    results_df = pd.read_csv(csv_file)
    kmer_header = [f"{k}mer" for k in kmer]
    rows = []
    rows.append(["Model"] + kmer_header)
    for framework in frameworks:
        results_fw = results_df[results_df["framework"] == framework]
        # Trellis only runs one configuration
        if framework == "trellis":
            row = []
            row.append(f"{framework.capitalize()}")
            for k in kmer:
                times = results_fw[results_fw["k"] == k]["time"]
                row.append(f"{print_mean_pm_stddev(list(times))}")
            rows.append(row)
        else:
            for configuration in configurations:
                row = []
                results_config = results_fw[results_fw["configuration"] == configuration]
                row.append(f"{framework.capitalize()} ({configuration})")
                for k in kmer:
                    times = results_config[results_config["k"] == k]["time"]
                    row.append(f"{print_mean_pm_stddev(list(times))}")
                rows.append(row)
    print_aligned_columns(rows)

def compile_trellis_shared_libs():
    r = subprocess.run(["make"], capture_output=True)
    if r.returncode != 0:
        out = r.stdout.decode("ascii")
        err = r.stderr.decode("ascii")
        print(f"Failed to compile shared libraries: {out} | {err}")
        exit(r.returncode)

def run_forward_benchmark():
    frameworks = ["parpy", "triton", "trellis"]
    configurations = [1, 2, 3]
    kmer = [5, 7]
    csv_file = f"{common.FORWARD_NAME}.csv"

    # Only run the benchmarks if the CSV data file does not exist.
    if not os.path.isfile(csv_file):
        # Compile the shared libraries for the generated Trellis code if they do
        # not yet exist.
        compile_trellis_shared_libs()

        clear_log_output("forward")

        # Run the benchmarks over all possible configurations.
        with tqdm(total=len(kmer) * len(frameworks)) as pbar:
            for k in kmer:
                for framework in frameworks:
                    if framework == "trellis":
                        launch_bench("forward", [framework, str(k)])
                    else:
                        for configuration in configurations:
                            launch_bench("forward", [framework, str(k), str(configuration)])
                    pbar.update(1)
    else:
        print("CSV results found - skipping benchmarks and printing results")

    # Print the results in a readable format
    print_forward_output(csv_file, frameworks, configurations, kmer)

def produce_sddmm_output(csv_file, frameworks, k):
    def set_colors(bp, colors):
        for i, c in enumerate(colors):
            plt.setp(bp["boxes"][i], color=c)
            bp["boxes"][i].set_facecolor("#FFFFFF")
            plt.setp(bp["whiskers"][2*i], color=c)
            plt.setp(bp["whiskers"][2*i+1], color=c)
            plt.setp(bp["caps"][2*i], color=c)
            plt.setp(bp["caps"][2*i+1], color=c)
            plt.setp(bp["medians"][i], color="#000000")
    results_df = pd.read_csv(csv_file)
    fig, axs = plt.subplots(layout="constrained")
    times = [[] for i in range(9)]
    colors = ["#0072B2", "#D55E00", "#E69F00"]
    for i, framework in enumerate(frameworks):
        fw_res = results_df[results_df["framework"] == framework]
        # Group by number of non-zeros within an exponential range, and
        # produce a box plot for each interval.
        for e in range(1, 10):
            nnz_range = fw_res["nnz"].between(10**(e-1), 10**e)
            times[e-1].append(fw_res[nnz_range]["time"])
    for i in range(9):
        bp = axs.boxplot(times[i], positions=[i*4+1, i*4+2, i*4+3], whis=[0, 100], patch_artist=True, widths=0.6)
        set_colors(bp, colors)
    axs.set_xticks([i for i in np.arange(0, 36, 4)])
    axs.set_xticklabels([f"$10^{i}$" for i in range(9)])
    axs.set_yscale("log")
    axs.set_xlabel("Number of non-zero values", fontsize=16)
    axs.set_ylabel("Execution time (ms)", fontsize=16)
    axs.set_axisbelow(True)
    axs.yaxis.grid(color="gray", which="major", alpha=.5)
    axs.yaxis.grid(color="gray", which="minor", alpha=.2)
    axs.tick_params(axis="both", which="major", labelsize=16)
    axs.tick_params(axis="both", which="minor", labelsize=14)

    # Plot legend with each framework associated with a color
    for c, fw in zip(colors, frameworks):
        plt.plot([], c=c, label=fw)
    plt.legend()

    axs.legend(loc="upper left", fontsize=16)
    fig.savefig(f"sddmm-{k}.pdf", bbox_inches="tight", pad_inches=0.05)

def run_sddmm_benchmark(k, limit=2892):
    frameworks = ["cuSPARSE", "ParPy-CSR", "ParPy-COO"]
    csv_file = f"{common.SDDMM_NAME}-{k}.csv"

    if not os.path.isfile(csv_file):
        # 1. Download all matrices
        matrices = ssgetpy.search(limit=limit, nzbounds=(None, 10**9))
        for matrix in tqdm(matrices, desc="Downloading SuiteSparse matrices (this may take a while)"):
            common.download_matrix(matrix)

        # 2. Run the benchmark on each matrix for each of the frameworks and store
        # the results in a file.
        import sddmm
        clear_log_output("sddmm")
        niters = len(matrices) * len(frameworks)
        for idx, matrix in enumerate(tqdm(matrices, desc=f"Running benchmarks")):
            for framework in frameworks:
                fn = lambda: sddmm.run_sddmm(framework, matrix.name, k)
                launch_bench("sddmm", [framework, matrix.name], fn=fn)

        # 3. After running all benchmarks, we report the number of benchmarks
        # failed by the respective framework and whether it failed due to OOM
        # or another reason.
        for framework in frameworks:
            if framework in bench_ooms:
                print(f"Framework {framework} ran out of memory in {bench_ooms[framework]} benchmarks.")
            if framework in bench_fails:
                print(f"Framework {framework} failed {bench_fails[framework]} benchmarks.")
    else:
        print("CSV results found - skipping benchmarks and plotting results")

    # 4. Generate a plot based on the results.
    produce_sddmm_output(csv_file, frameworks, k)

benchmark_id = sys.argv[1]
if benchmark_id == "all":
    run_forward_benchmark()
    run_sddmm_benchmark(64)
    run_sddmm_benchmark(1024)
if benchmark_id == "forward":
    run_forward_benchmark()
elif benchmark_id == "sddmm":
    k = int(sys.argv[2])
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
        run_sddmm_benchmark(k, limit)
    else:
        run_sddmm_benchmark(k)
