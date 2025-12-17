import h5py
import numpy as np
from math import inf
import os
import pandas as pd
import parpy
import parpy.builtin as pb
import parpy.math as pm
import parpy.types as pt
import pytest
import statistics
import sys
import torch
import triton
import triton.language as tl

import common

DATA_PATH = "../test/data"

# I/O handling for the model and the observation sequences
def generate_init_probs(k):
    init_probs = np.zeros((16, 4**k), dtype=np.float32)
    for kmer in range(0, 4**k):
        init_probs[0][kmer] = np.log(1.0 / float(4**k))
    for layer in range(1, 16):
        for kmer in range(0, 4**k):
            init_probs[layer][kmer] = -inf
    return init_probs

def reverse_index(i, k):
    return sum([(i // 4**x) % 4 * (4**(k-x-1)) for x in range(k)])

def transform_output_probs(obs, k):
    output_probs = np.zeros((4**k, 101), dtype=np.float32)
    for i in range(4**k):
        idx = reverse_index(i, k)
        for j in range(101):
            output_probs[i][j] = obs[j][idx]
    return output_probs.transpose()

def read_trellis_inputs(model_filename, signals_filename):
    model_path = f"{DATA_PATH}/{model_filename}"
    signals_path = f"{DATA_PATH}/{signals_filename}"
    with h5py.File(model_path, "r") as f:
        with np.errstate(divide="ignore"):
            obs = np.log(f['Tables']['ObservationProbabilities'][:])
        trans1 = np.log(f['Tables']['TransitionProbabilities'][:])
        duration = np.log(f['Tables']['DurationProbabilities'][:])
        tail_factor = np.log(f['Tables']['DurationProbabilities'].attrs['TailFactor'])
        k = f['Parameters'].attrs['KMerLength']
        init_probs = generate_init_probs(k)
        trans1 = trans1.reshape(4, 4**k).transpose(1, 0)
        out_prob = transform_output_probs(obs, k)
    with h5py.File(signals_path, "r") as f:
        keys = list(f.keys())
        signals = [f[k]['Raw']['Signal'][:].tolist() for k in keys]
    synthetic_248 = np.log(np.exp(0.) - np.exp(tail_factor))
    num_states = 16 * 4**k

    # Convert data to torch compatible format allocated on the GPU.
    trans1 = torch.tensor(trans1, dtype=torch.float32, device='cuda')
    trans2 = torch.tensor(duration, dtype=torch.float32, device='cuda')
    out_prob = torch.tensor(out_prob, dtype=torch.float32, device='cuda')
    init_prob = torch.tensor(init_probs, dtype=torch.float32, device='cuda')
    hmm = {
        'gamma': torch.tensor(tail_factor, dtype=torch.float32, device='cuda'),
        'trans1': trans1.contiguous(),
        'trans2': trans2,
        'output_prob': out_prob.contiguous(),
        'initial_prob': init_prob.flatten(),
        'synthetic_248': torch.tensor(synthetic_248, dtype=torch.float32, device='cuda'),
        'num_states': torch.tensor(num_states, dtype=torch.int64, device='cuda')
    }

    signal_lengths = [len(s) for s in signals]
    maxlen = max(signal_lengths)
    torch_signals = torch.empty((len(signals), maxlen), dtype=torch.int8, device='cuda')
    for i, s in enumerate(signals):
        torch_signals[i, 0:len(s)] = torch.tensor(s, dtype=torch.int8, device='cuda')
    lens = torch.tensor(signal_lengths, dtype=torch.int64, device='cuda')
    num_instances = len(lens)
    seqs = {
        'data': torch_signals,
        'lens': lens,
        'maxlen': torch.tensor(maxlen, dtype=torch.int64, device='cuda'),
        'num_instances': torch.tensor(num_instances, dtype=torch.int64, device='cuda')
    }
    return hmm, seqs

def read_expected_output(fname):
    path = f"{DATA_PATH}/{fname}"
    with open(path) as f:
        return np.asarray([float(l) for l in f.readlines()], dtype=np.float32)

@parpy.jit
def forward_init_inst(hmm, seqs, alpha_src, inst):
    o = seqs["data"][inst, 0]
    parpy.label('state')
    for state in range(hmm["num_states"]):
        num_kmers = hmm["num_states"] // 16
        alpha_src[inst, state] = hmm["initial_prob"][state] + \
                                 hmm["output_prob"][o, state % num_kmers]

@parpy.jit
def forward_step_inst(hmm, seqs, alpha1, alpha2, inst, t):
    alpha_src = alpha2
    alpha_dst = alpha1
    if t & 1:
        alpha_src = alpha1
        alpha_dst = alpha2
    o = seqs["data"][inst, t]
    parpy.label('state')
    for state in range(hmm["num_states"]):
        if t < seqs["lens"][inst]:
            # Transitively inlined version of forward_prob_predecessors.
            num_kmers = hmm["num_states"] // 16

            pred1 = pb.convert((state // 4) % (hmm["num_states"] // 64), pt.I32)
            pred2 = pb.convert(hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64), pt.I32)
            pred3 = pb.convert(2 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64), pt.I32)
            pred4 = pb.convert(3 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64), pt.I32)
            t11 = hmm["trans1"][pred1 % num_kmers, state % 4]
            t12 = hmm["trans1"][pred2 % num_kmers, state % 4]
            t13 = hmm["trans1"][pred3 % num_kmers, state % 4]
            t14 = hmm["trans1"][pred4 % num_kmers, state % 4]
            t2 = hmm["trans2"][state // num_kmers]
            p1 = t11 + t2 + alpha_src[inst, pred1]
            p2 = t12 + t2 + alpha_src[inst, pred2]
            p3 = t13 + t2 + alpha_src[inst, pred3]
            p4 = t14 + t2 + alpha_src[inst, pred4]

            pred5 = pb.convert(0, pt.I32)
            p5 = pb.convert(0.0, pt.F32)
            if state // num_kmers == 15:
                pred5 = state
                p5 = hmm["gamma"]
            elif state // num_kmers == 14:
                pred5 = ((state // num_kmers) + 1) * num_kmers + state % num_kmers
                p5 = hmm["synthetic_248"]
            else:
                pred5 = ((state // num_kmers) + 1) * num_kmers + state % num_kmers
                p5 = 0.0
            p5 = p5 + alpha_src[inst, pred5]

            # Inlined version of log_sum_exp.
            maxp = pb.maximum(p1, p2)
            maxp = pb.maximum(maxp, p3)
            maxp = pb.maximum(maxp, p4)
            maxp = pb.maximum(maxp, p5)
            lsexp = maxp + pm.log(
                pm.exp(p1 - maxp) +
                pm.exp(p2 - maxp) +
                pm.exp(p3 - maxp) +
                pm.exp(p4 - maxp) +
                pm.exp(p5 - maxp)
            )
            lsexp = pb.maximum(lsexp, pb.convert(-pb.inf, pt.F32))

            alpha_dst[inst, state] = lsexp + hmm["output_prob"][o, state % num_kmers]

@parpy.jit
def forward_lse_inst(hmm, seqs, result, alpha1, alpha2, inst):
    # Summation of final alpha values
    alpha = alpha2
    if seqs["lens"][inst] & 1 != 0:
        alpha = alpha1

    parpy.label('state')
    maxp = parpy.reduce.max(alpha[inst, :])

    parpy.label('state')
    psum = parpy.reduce.sum(pm.exp(alpha[inst, :] - maxp))

    result[inst] = maxp + pm.log(psum)

@parpy.jit
def forward_kernel(hmm, seqs, result, alpha1, alpha2):
    parpy.label('inst')
    for inst in range(seqs["num_instances"]):
        # Initialization
        pb.inline(forward_init_inst(hmm, seqs, alpha1, inst))

        # Forward steps for t = 1, 2, .. maxlen
        for t in range(1, seqs["maxlen"]):
            pb.inline(forward_step_inst(hmm, seqs, alpha1, alpha2, inst, t))

        # Summation of final alpha values
        pb.inline(forward_lse_inst(hmm, seqs, result, alpha1, alpha2, inst))

def forward_parpy(hmm, seqs, nthreads):
    result = torch.zeros(seqs["num_instances"], dtype=torch.float32, device='cuda')
    alpha1 = torch.zeros((seqs["num_instances"], hmm["num_states"]), dtype=torch.float32, device='cuda')
    alpha2 = torch.zeros_like(alpha1)
    p = {
        'inst': parpy.threads(seqs["num_instances"]),
        'state': parpy.threads(nthreads),
    }
    opts = parpy.par(p)
    opts.force_int_size = pt.I64
    forward_kernel(hmm, seqs, result, alpha1, alpha2, opts=opts)
    return result

class ParPyTuned:
    def __init__(self, hmm, seqs):
        best_time = float('inf')
        best_nthreads = 0
        for nthreads in [2**n for n in range(10, 19) if 2**n <= hmm["num_states"]]:
            times = common.bench("parpy-tuning", lambda: forward_parpy(hmm, seqs, nthreads), nruns=5, nwarmup=1)
            avg = statistics.mean(times)
            if avg < best_time:
                best_time = avg
                best_nthreads = nthreads
        self.hmm = hmm
        self.seqs = seqs
        self.nthreads = best_nthreads

    def __call__(self):
        return forward_parpy(self.hmm, self.seqs, self.nthreads)

    def __name__(self):
        return "ParPyTuned"

@triton.jit
def forward_triton_init(
        hmm_initial_prob, hmm_output_prob, hmm_num_states : tl.constexpr, seqs_data,
        seqs_maxlen : tl.constexpr, alpha_src):
  instance = tl.program_id(axis=0)
  o = tl.load(seqs_data + instance * seqs_maxlen)
  state_idx = tl.arange(0, hmm_num_states)
  init_prob = tl.load(hmm_initial_prob + state_idx)
  num_kmers = hmm_num_states // 16
  out_prob = tl.load(hmm_output_prob + o * num_kmers + state_idx % num_kmers)
  tl.store(alpha_src + instance * hmm_num_states + state_idx, init_prob + out_prob)

@triton.jit
def forward_triton_step(hmm_output_prob, hmm_trans1, hmm_trans2, hmm_gamma: tl.float32, hmm_synthetic_248: tl.float32, hmm_num_states: tl.constexpr, seqs_data, seqs_lens, seqs_maxlen: tl.constexpr, alpha1, alpha2, t, BLOCK_SIZE: tl.constexpr):
  instance = tl.program_id(axis=0)
  if t & 1:
    alpha_src = alpha1
    alpha_dst = alpha2
  else:
    alpha_src = alpha2
    alpha_dst = alpha1
  state_ofs = BLOCK_SIZE * tl.program_id(axis=1)
  state_idx = state_ofs + tl.arange(0, BLOCK_SIZE)
  idx = instance * hmm_num_states + state_idx
  seq_len = tl.load(seqs_lens + instance)
  if t < seq_len:
    o = tl.load(seqs_data + instance * seqs_maxlen + t)
    num_kmers = hmm_num_states // 16

    # Transitively inlined version of forward_prob_predecessors. The loop is
    # unrolled to allow writing to distinct tensors.
    pred = (state_idx // 4) % (hmm_num_states // 64)
    t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
    t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
    p0 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

    pred = hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
    t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
    t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
    p1 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

    pred = 2 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
    t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
    t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
    p2 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

    pred = 3 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
    t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
    t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
    p3 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

    # For this part, we have three possible cases, but as Triton reasons about
    # the whole block, we cannot have if-conditions here. We express it using
    # the "tl.where" function, which conditionally chooses values between two
    # tensors based on whether the provided booleans are true or false.

    # if state // 64 == 15
    pred_fst = state_idx
    p_fst = tl.full((BLOCK_SIZE,), hmm_gamma, dtype=tl.float32)

    # else if state // 64 == 14
    pred_snd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
    p_snd = tl.full((BLOCK_SIZE,), hmm_synthetic_248, dtype=tl.float32)

    # else (if state // 64 != 14 && state // 64 != 15)
    pred_trd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
    p_trd = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Combination of the three above cases...
    pred = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    pred = tl.where(state_idx // num_kmers == 15, pred_fst, pred)
    pred = tl.where(state_idx // num_kmers == 14, pred_snd, pred)
    pred = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, pred_trd, pred)
    p = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    p = tl.where(state_idx // num_kmers == 15, p_fst, p)
    p = tl.where(state_idx // num_kmers == 14, p_snd, p)
    p = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, p_trd, p)
    p4 = p + tl.load(alpha_src + instance * hmm_num_states + pred)

    # Inlined version of log_sum_exp
    maxp = tl.maximum(p0, p1)
    maxp = tl.maximum(maxp, p2)
    maxp = tl.maximum(maxp, p3)
    maxp = tl.maximum(maxp, p4)
    lsexp = maxp + tl.log(tl.exp(p0-maxp) + tl.exp(p1-maxp) + tl.exp(p2-maxp) + tl.exp(p3-maxp) + tl.exp(p4-maxp))
    neginfs = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    lsexp = tl.maximum(lsexp, neginfs)

    outp = tl.load(hmm_output_prob + o * num_kmers + state_idx % num_kmers)
    tl.store(alpha_dst + idx, lsexp + outp)
  elif seq_len == t:
    alpha_val = tl.load(alpha_src + idx)
    tl.store(alpha_dst + idx, alpha_val)

@triton.jit
def forward_triton_steps(hmm_output_prob, hmm_trans1, hmm_trans2, hmm_gamma : tl.float32, hmm_synthetic_248 : tl.float32, hmm_num_states : tl.constexpr, seqs_data, seqs_lens, seqs_maxlen : tl.constexpr, alpha1, alpha2):
  instance = tl.program_id(axis=0)
  for t in range(1, seqs_maxlen):
    if t & 1:
      alpha_src = alpha1
      alpha_dst = alpha2
    else:
      alpha_src = alpha2
      alpha_dst = alpha1
    state_idx = tl.arange(0, hmm_num_states)
    idx = instance * hmm_num_states + state_idx
    seq_len = tl.load(seqs_lens + instance)
    if t < seq_len:
      o = tl.load(seqs_data + instance * seqs_maxlen + t)
      num_kmers = hmm_num_states // 16

      # Transitively inlined version of forward_prob_predecessors. The loop is
      # unrolled to allow writing to distinct tensors.
      pred = (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p0 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p1 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = 2 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p2 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = 3 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p3 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      # For this part, we have three possible cases, but as Triton reasons about
      # the whole block, we cannot have if-conditions here. We express it using
      # the "tl.where" function, which conditionally chooses values between two
      # tensors based on whether the provided booleans are true or false.

      # if state // 64 == 15
      pred_fst = state_idx
      p_fst = tl.full((hmm_num_states,), hmm_gamma, dtype=tl.float32)

      # else if state // 64 == 14
      pred_snd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
      p_snd = tl.full((hmm_num_states,), hmm_synthetic_248, dtype=tl.float32)

      # else (if state // 64 != 14 && state // 64 != 15)
      pred_trd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
      p_trd = tl.zeros((hmm_num_states,), dtype=tl.float32)

      # Combination of the three above cases...
      pred = tl.zeros((hmm_num_states,), dtype=tl.int32)
      pred = tl.where(state_idx // num_kmers == 15, pred_fst, pred)
      pred = tl.where(state_idx // num_kmers == 14, pred_snd, pred)
      pred = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, pred_trd, pred)
      p = tl.full((hmm_num_states,), float('-inf'), dtype=tl.float32)
      p = tl.where(state_idx // num_kmers == 15, p_fst, p)
      p = tl.where(state_idx // num_kmers == 14, p_snd, p)
      p = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, p_trd, p)
      p4 = p + tl.load(alpha_src + instance * hmm_num_states + pred)

      # Inlined version of log_sum_exp
      maxp = tl.maximum(p0, p1)
      maxp = tl.maximum(maxp, p2)
      maxp = tl.maximum(maxp, p3)
      maxp = tl.maximum(maxp, p4)
      lsexp = maxp + tl.log(tl.exp(p0-maxp) + tl.exp(p1-maxp) + tl.exp(p2-maxp) + tl.exp(p3-maxp) + tl.exp(p4-maxp))
      neginfs = tl.full((hmm_num_states,), float('-inf'), dtype=tl.float32)
      lsexp = tl.maximum(lsexp, neginfs)

      outp = tl.load(hmm_output_prob + o * num_kmers + state_idx % num_kmers)
      tl.store(alpha_dst + idx, lsexp + outp)
    elif seq_len == t:
      alpha_val = tl.load(alpha_src + idx)
      tl.store(alpha_dst + idx, alpha_val)
    tl.debug_barrier()

@triton.jit
def forward_triton_lse(hmm_num_states : tl.constexpr, seqs_maxlen : tl.constexpr, alpha1, alpha2, result):
  instance = tl.program_id(axis=0)
  if seqs_maxlen & 1:
    alpha = alpha1
  else:
    alpha = alpha2
  state_idx = tl.arange(0, hmm_num_states)
  alpha_vals = tl.load(alpha + instance * hmm_num_states + state_idx)
  maxp = tl.max(alpha_vals, axis=0)
  psum = tl.sum(tl.exp(alpha_vals - maxp), axis=0)
  tl.store(result + instance, maxp + tl.log(psum))

def forward_triton(hmm, seqs, nthreads):
    result = torch.empty(seqs["num_instances"], dtype=torch.float32, device='cuda')
    alpha1 = torch.empty(seqs["num_instances"] * hmm["num_states"], dtype=torch.float32, device='cuda')
    alpha2 = torch.empty_like(alpha1)

    def grid_sz(meta):
      if "BLOCK_SIZE" in meta:
        return (seqs["num_instances"], hmm["num_states"] // meta["BLOCK_SIZE"])
      else:
        return (seqs["num_instances"], )
    forward_triton_init[grid_sz](
        hmm["initial_prob"], hmm["output_prob"], int(hmm["num_states"]),
        seqs["data"].flatten(), int(seqs["maxlen"]), alpha1
    )
    if nthreads <= 1024:
        forward_triton_steps[grid_sz](
            hmm["output_prob"], hmm["trans1"], hmm["trans2"],
            float(hmm["gamma"]), float(hmm["synthetic_248"]), int(hmm["num_states"]),
            seqs["data"], seqs["lens"], int(seqs["maxlen"]), alpha1, alpha2
        )
    else:
        # If the user requests more than 1024 threads, multiple blocks have to
        # run the code in each time step. As Triton has no way of multi-block
        # synchronization, we have to break it up into multiple separate calls
        # to Triton kernels.
        for t in range(1, seqs["maxlen"]):
            forward_triton_step[grid_sz](
              hmm["output_prob"], hmm["trans1"], hmm["trans2"],
              float(hmm["gamma"]), float(hmm["synthetic_248"]), int(hmm["num_states"]),
              seqs["data"], seqs["lens"], int(seqs["maxlen"]), alpha1, alpha2, t,
              BLOCK_SIZE=1024
            )
    forward_triton_lse[grid_sz](
        int(hmm["num_states"]), int(seqs["maxlen"]), alpha1, alpha2, result
    )
    return result

class TritonTuned:
    def __init__(self, hmm, seqs):
        best_time = float('inf')
        best_nthreads = 0
        for nthreads in [2**n for n in range(10, 19) if 2**n <= hmm["num_states"]]:
            times = common.bench("triton3-tune", lambda: forward_triton(hmm, seqs, nthreads))
            avg = statistics.mean(times)
            if avg < best_time:
                best_time = avg
                best_nthreads = nthreads
        self.hmm = hmm
        self.seqs = seqs
        self.nthreads = best_nthreads

    def __call__(self):
        return forward_triton(self.hmm, self.seqs, self.nthreads)

    def __name__(self):
        return "TritonTuned"

def forward_trellis(hmm, signals, k):
    return hmm.forward(signals)

def run_forward(framework, k, config_id):
    if framework == "parpy":
        base_fn = forward_parpy
    elif framework == "triton":
        base_fn = forward_triton
    elif framework == "trellis":
        pass
    else:
        sys.stderr.write(f"Unknown framework: {framework}\n")
        return 1

    hmm, seqs = read_trellis_inputs(f"{k}mer-model.hdf5", "signals.hdf5")
    expected = read_expected_output(f"{k}mer-expected.txt")

    if framework == "trellis":
        cwd = os.getcwd()
        if k == 5:
            path = f"{cwd}/5mer"
        elif k == 7:
            path = f"{cwd}/7mer"
        sys.path.append(path)
        from trellis import HMM
        tables = {
            'gamma': hmm['gamma'].cpu().numpy(),
            'trans1': hmm['trans1'].cpu().numpy(),
            'trans2': hmm['trans2'].cpu().numpy(),
            'outputProb': hmm['output_prob'].cpu().numpy(),
            'initialProb': hmm['initial_prob'].cpu().numpy()
        }
        hmm = HMM(tables)
        signals = len(seqs['lens']) * [None]
        for i in range(len(signals)):
            signals[i] = seqs['data'][i, :seqs['lens'][i]].cpu().numpy()
        fn = lambda: forward_trellis(hmm, signals, k)
    elif config_id == 1:
        fn = lambda: base_fn(hmm, seqs, 1024)
    elif config_id == 2:
        fn = lambda: base_fn(hmm, seqs, hmm["num_states"])
    elif config_id == 3:
        if framework == "parpy":
            fn = ParPyTuned(hmm, seqs)
        else:
            fn = TritonTuned(hmm, seqs)

    def mk_framework_entry(fw, config, k, t):
        return {"framework": fw, "configuration": config, "k": k, "time": t}

    config = f"{framework.capitalize()}-{config_id}-{k}"
    times = common.bench(config, fn, expected)
    results = [mk_framework_entry(framework, config_id, k, t) for t in times]
    common.append_csv(f"{common.FORWARD_NAME}.csv", results)
    return 0

def print_gpu(gpu_id, idx, n):
    # Format the columns differently for the rightmost GPU column
    col_format = "c|" if idx < n - 1 else "c"
    return f"\\multicolumn{{2}}{{{col_format}}}{{{gpu_id}}}"

def print_mean_pm_stddev(times):
    if len(times) == 0:
        return "-"
    else:
        return f"{np.mean(times):.1f} \\pm {np.std(times):.1f}"

def print_configuration(files, framework, configuration):
    kmer = [5, 7]
    if configuration is not None:
        print(f"{framework.capitalize()} ({configuration})", end="")
    else:
        print(f"{framework.capitalize()}", end="")

    for file in files:
        results_df = pd.read_csv(file)
        results_fw = results_df[results_df["framework"] == framework]
        if configuration is not None:
            results_fw = results_fw[results_fw["configuration"] == configuration]
        for k in kmer:
            times = results_fw[results_fw["k"] == k]["time"]
            print(f" & ${print_mean_pm_stddev(list(times))}$", end="")
    print(r"\\")

def print_table(gpus, files):
    frameworks = ["parpy", "triton", "trellis"]
    configurations = [1, 2, 3]
    kmer = [5, 7]

    tab_format = "|".join(["cc" for _ in gpus])
    print(f"\\begin{{tabular}}{{l|{tab_format}}}")
    gpu_format = " & ".join([print_gpu(gpu, i, len(gpus)) for i, gpu in enumerate(gpus)])
    print(f"GPU & {gpu_format}\\\\")
    print(r"\hline")
    model_format = " & ".join([" & ".join([f"{k}mer" for k in kmer]) for _ in gpus])
    print(f"Model & {model_format}\\\\")
    print(r"\hline")

    # Print ParPy configurations
    framework = "parpy"
    for configuration in configurations:
        print_configuration(files, framework, configuration)

    framework = "triton"
    for configuration in configurations:
        print_configuration(files, framework, configuration)

    framework = "trellis"
    print_configuration(files, framework, None)

    print(r"\end{tabular}")

if __name__ == "__main__":
    if sys.argv[1] == "plot":
        args = sys.argv[2:]
        assert len(args) % 2 == 0
        gpus = args[0:len(args):2]
        files = args[1:len(args):2]
        print_table(gpus, files)
    else:
        framework = sys.argv[1]
        k = int(sys.argv[2])
        config_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        run_forward(framework, k, config_id)
