import numpy as np
import pandas as pd
import os
import requests
import sys
import tarfile
import time
import torch
from tqdm import tqdm
from scipy.io import mmread
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta state.*")

URL_BASE = "https://sparse.tamu.edu"
SUITESPARSE_PATH = os.environ.get("SUITESPARSE_PATH", "/src/suitesparse")

FORWARD_CSV = "forward-results.csv"
SDDMM_CSV = "sddmm-results.csv"

BATCH_SIZE = 16384

# Custom download function because ssgetpy uses a throttled approach.
def download_extracted_to(url, dst_path):
    response = requests.get(url, stream=True)
    content_length = int(response.headers["content-length"])
    download_file = "/tmp/data.tar.gz"
    with open(download_file, "wb") as outfile, tqdm(
        total=content_length, unit="B"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=BATCH_SIZE):
            outfile.write(chunk)
            pbar.update(BATCH_SIZE)
    with tarfile.open(download_file) as f:
        f.extractall(SUITESPARSE_PATH)
    os.unlink(download_file)

def download_matrix(matrix):
    matrix_path = Path(f"{SUITESPARSE_PATH}/{matrix.name}/{matrix.name}.mtx")
    if not matrix_path.exists():
        url = f"{URL_BASE}/MM/{matrix.group}/{matrix.name}.tar.gz"
        download_extracted_to(url, matrix_path)

def ssgetpy_matrix_to_csr(matrix_name):
    matrix_path = Path(f"{SUITESPARSE_PATH}/{matrix_name}/{matrix_name}.mtx")
    matrix = mmread(matrix_path.resolve())
    matrix = matrix.tocsr()
    crow_indices = torch.tensor(matrix.indptr, dtype=torch.int64)
    col_indices = torch.tensor(matrix.indices, dtype=torch.int64)
    values = torch.tensor(matrix.data, dtype=torch.float32)
    shape = matrix.shape
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(shape), device='cuda')

def bench(id, f, expected=None, nruns=10, nwarmup=10, atol=1e-5, rtol=1e-5):
    def unwrap_value(v):
        if isinstance(v, torch.Tensor):
            if v.layout == torch.sparse_csr:
                v = v.values()
            return np.asarray(v.cpu())
        else:
            return v
    # Initial warmup with output validation
    v = f()
    torch.cuda.synchronize()
    if expected is not None:
        expected = unwrap_value(expected)
        v = unwrap_value(v)
        if not np.allclose(v, expected, atol=atol, rtol=rtol):
            lhs = np.abs(v.flatten() - expected.flatten())
            rhs = atol + rtol * np.abs(expected.flatten())
            nfails = len(v.flatten()[lhs > rhs])
            sys.stderr.write(f"{id} | {nfails}\n")
            sys.stderr.write(f"lhs: {expected}\nrhs: {v}\n\n")

    # Benchmarking with warmups
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(nruns+nwarmup):
        start.record()
        f()
        end.record()
        torch.cuda.synchronize()
        if i < nwarmup:
            continue
        times += [start.elapsed_time(end)]
    return times

def append_csv(csv_file, results):
    df = pd.DataFrame(results)
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=header)
