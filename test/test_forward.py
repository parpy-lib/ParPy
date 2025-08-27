import importlib
import numpy as np
import parpy
from parpy.operators import int16, exp, float32, inf, log, max, sum
import pytest
import torch

from common import *

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

def read_trellis_inputs(model_path, signals_path):
    import h5py
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

    # Convert data to torch compatible format
    trans1 = torch.tensor(trans1, dtype=torch.float32)
    trans2 = torch.tensor(duration, dtype=torch.float32)
    out_prob = torch.tensor(out_prob, dtype=torch.float32)
    init_prob = torch.tensor(init_probs, dtype=torch.float32)
    hmm = {
        'gamma': torch.tensor(tail_factor, dtype=torch.float32),
        'trans1': trans1.contiguous(),
        'trans2': trans2,
        'output_prob': out_prob.contiguous(),
        'initial_prob': init_prob.flatten(),
        'synthetic_248': torch.tensor(synthetic_248, dtype=torch.float32),
        'num_states': torch.tensor(num_states, dtype=torch.int64)
    }

    signal_lengths = [len(s) for s in signals]
    maxlen = max(signal_lengths)
    torch_signals = torch.zeros((len(signals), maxlen), dtype=torch.int8)
    for i, s in enumerate(signals):
        torch_signals[i, 0:len(s)] = torch.tensor(s, dtype=torch.int8)
    lens = torch.tensor(signal_lengths, dtype=torch.int64)
    num_instances = len(lens)
    seqs = {
        'data': torch_signals,
        'lens': lens,
        'maxlen': torch.tensor(maxlen, dtype=torch.int64),
        'num_instances': torch.tensor(num_instances, dtype=torch.int64)
    }
    return hmm, seqs

def read_expected_output(fname):
  with open(fname) as f:
    return torch.tensor([float(l) for l in f.readlines()], dtype=torch.float32)

@parpy.jit
def forward_kernel(hmm, seqs, alpha1, alpha2, result):
    parpy.label('inst')
    for inst in range(seqs["num_instances"]):
        # Initialization
        o = seqs["data"][inst, 0]
        parpy.label('state')
        for state in range(hmm["num_states"]):
            alpha1[inst, state] = hmm["initial_prob"][state] + hmm["output_prob"][o, state % 64]

        # Forward steps (t = 1, .., maxlen-1)
        for t in range(1, seqs["maxlen"]):
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

                    pred1 = int16((state // 4) % (hmm["num_states"] // 64))
                    pred2 = int16(hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64))
                    pred3 = int16(2 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64))
                    pred4 = int16(3 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64))
                    t11 = hmm["trans1"][pred1 % num_kmers, state % 4]
                    t12 = hmm["trans1"][pred2 % num_kmers, state % 4]
                    t13 = hmm["trans1"][pred3 % num_kmers, state % 4]
                    t14 = hmm["trans1"][pred4 % num_kmers, state % 4]
                    t2 = hmm["trans2"][state // num_kmers]
                    p1 = t11 + t2 + alpha_src[inst, pred1]
                    p2 = t12 + t2 + alpha_src[inst, pred2]
                    p3 = t13 + t2 + alpha_src[inst, pred3]
                    p4 = t14 + t2 + alpha_src[inst, pred4]

                    pred5 = int16(0)
                    p5 = float32(0.0)
                    if state // num_kmers == 15:
                        pred5 = state
                        p5 = hmm["gamma"]
                    elif state // num_kmers == 14:
                        pred5 = ((state // num_kmers) + 1) * num_kmers + state % num_kmers
                        p5 = hmm["synthetic_248"]
                    else:
                        pred5 = ((state // num_kmers) + 1) * num_kmers + state % num_kmers
                        p5 = float32(0.0)
                    p5 = p5 + alpha_src[inst, pred5]

                    # Inlined version of log_sum_exp.
                    maxp = max(p1, p2)
                    maxp = max(maxp, p3)
                    maxp = max(maxp, p4)
                    maxp = max(maxp, p5)
                    lsexp = maxp + log(
                        exp(p1 - maxp) + exp(p2 - maxp) + exp(p3 - maxp) +
                        exp(p4 - maxp) + exp(p5 - maxp)
                    )
                    lsexp = max(lsexp, float32(-inf))

                    alpha_dst[inst, state] = lsexp + hmm["output_prob"][o, state % num_kmers]
                elif t == seqs["lens"][inst]:
                    alpha_dst[inst, state] = alpha_src[inst, state]

        # Summation of final alpha values
        alpha = alpha2
        if seqs["maxlen"] & 1:
            alpha = alpha1

        parpy.label('state')
        maxp = max(alpha[inst, :])

        parpy.label('state')
        psum = sum(exp(alpha[inst, :] - maxp))

        result[inst] = maxp + log(psum)

def forward(hmm, seqs, opts):
    alpha1 = torch.zeros((seqs["num_instances"], hmm["num_states"]), dtype=torch.float32)
    alpha2 = torch.zeros_like(alpha1)
    result = torch.zeros(seqs["num_instances"], dtype=torch.float32)
    forward_kernel(hmm, seqs, alpha1, alpha2, result, opts=opts)
    return result

def read_test_data():
    model = "test/data/3mer-model.hdf5"
    signals = "test/data/signals.hdf5"
    hmm, seqs = read_trellis_inputs(model, signals)
    expected = read_expected_output("test/data/3mer-expected.txt")
    return hmm, seqs, expected

def run_forw_test(hmm, seqs, expected, opts):
    probs = forward(hmm, seqs, opts)
    assert torch.allclose(probs, expected, atol=1e-5), f"{probs}\n{expected}"

@pytest.mark.skipif(importlib.util.find_spec('h5py') is None, reason="Test requires h5py")
@pytest.mark.parametrize('backend', compiler_backends)
def test_forward_single_block(backend):
    def helper():
        hmm, seqs, expected = read_test_data()
        p = {
            'inst': parpy.threads(seqs["num_instances"]),
            'state': parpy.threads(512),
        }
        run_forw_test(hmm, seqs, expected, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.skipif(importlib.util.find_spec('h5py') is None, reason="Test requires h5py")
@pytest.mark.parametrize('backend', compiler_backends)
def test_forward_multi_block(backend):
    def helper():
        hmm, seqs, expected = read_test_data()
        p = {
            'inst': parpy.threads(seqs["num_instances"]),
            'state': parpy.threads(2048),
        }
        if backend == parpy.CompileBackend.Metal:
            with pytest.raises(TypeError) as e_info:
                run_forw_test(hmm, seqs, expected, par_opts(backend, p))
            assert e_info.match(r".*nested pointer in generated code.*")
        else:
            run_forw_test(hmm, seqs, expected, par_opts(backend, p))
    run_if_backend_is_enabled(backend, helper)

@pytest.mark.skipif(importlib.util.find_spec('h5py') is None, reason="Test requires h5py")
@pytest.mark.parametrize('backend', compiler_backends)
def test_forward_compiles(backend):
    hmm, seqs, _ = read_test_data()
    result = torch.zeros(seqs["num_instances"], dtype=torch.float32)
    alpha1 = torch.zeros((seqs["num_instances"], hmm["num_states"]), dtype=torch.float32)
    alpha2 = torch.zeros_like(alpha1)
    p = {
        'inst': parpy.threads(seqs["num_instances"]),
        'state': parpy.threads(hmm["num_states"])
    }
    s = parpy.print_compiled(forward_kernel, [hmm, seqs, alpha1, alpha2, result], par_opts(backend, p))
    assert len(s) != 0
