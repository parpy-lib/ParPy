import parir
from parir import ParKind, ParSpec
import torch

@parir.jit
def forward_init(hmm, seqs, alpha_src):
    for inst in range(seqs["num_instances"]):
        o = seqs["data"][inst, 0]
        for state in range(hmm["num_states"]):
            alpha_src[inst, state] = hmm["initial_prob"][state] + hmm["output_prob"][o, state % 64]

@parir.jit
def forward_steps(hmm, seqs, alpha1, alpha2):
    for inst in range(seqs["num_instances"]):
        for t in range(1, seqs["maxlen"]):
            alpha_src = alpha2
            alpha_dst = alpha1
            if t & 1:
                alpha_src = alpha1
                alpha_dst = alpha2
            o = seqs["data"][inst, t]
            for state in range(hmm["num_states"]):
                if t < seqs["lens"][inst]:
                    # Transitively inlined version of forward_prob_predecessors.
                    num_kmers = hmm["num_states"] // 16

                    pred1 = (state // 4) % (hmm["num_states"] // 64)
                    pred2 = hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64)
                    pred3 = 2 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64)
                    pred4 = 3 * hmm["num_states"] // 64 + (state // 4) % (hmm["num_states"] // 64)
                    t11 = hmm["trans1"][pred1 % num_kmers * 4, state % 4]
                    t12 = hmm["trans1"][pred2 % num_kmers * 4, state % 4]
                    t13 = hmm["trans1"][pred3 % num_kmers * 4, state % 4]
                    t14 = hmm["trans1"][pred4 % num_kmers * 4, state % 4]
                    t2 = hmm["trans2"][state // num_kmers]
                    p1 = t11 + t2 + alpha_src[inst, pred1]
                    p2 = t12 + t2 + alpha_src[inst, pred2]
                    p3 = t13 + t2 + alpha_src[inst, pred3]
                    p4 = t14 + t2 + alpha_src[inst, pred4]

                    pred5 = 0
                    p5 = 0.0
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
                    maxp = max(p1, p2)
                    maxp = max(maxp, p3)
                    maxp = max(maxp, p4)
                    maxp = max(maxp, p5)
                    lsexp = maxp + parir.log(parir.exp(p1 - maxp) + parir.exp(p2 - maxp) + parir.exp(p3 - maxp) + parir.exp(p4 - maxp) + parir.exp(p5 - maxp))
                    lsexp = max(lsexp, -parir.inf)

                    alpha_dst[inst, state] = lsexp + hmm["output_prob"][o, state % num_kmers]
                elif t == seqs["lens"][inst]:
                    alpha_dst[inst, state] = alpha_src[inst, state]

@parir.jit
def forward_lse(hmm, seqs, result, alpha1, alpha2):
    for inst in range(seqs["num_instances"]):
        # Summation of final alpha values
        alpha = alpha2
        if seqs["maxlen"] & 1:
            alpha = alpha1

        maxp = -parir.inf
        for state in range(hmm["num_states"]):
            maxp = max(maxp, alpha[inst, state])

        psum = 0.0
        for state in range(hmm["num_states"]):
            psum = psum + parir.exp(alpha[inst, state] - maxp)

        result[inst] = maxp + parir.log(psum)

def forward(hmm, seqs):
    # NOTE: The arguments 'hmm' and 'seqs' are Python records. By annotating
    # functions with declared structs, we can send these to Parir functions.
    result = torch.empty(seqs["num_instances"], dtype=torch.float32, device='cuda')
    alpha1 = torch.empty((seqs["num_instances"], hmm["num_states"]), dtype=torch.float32, device='cuda')
    alpha2 = torch.empty_like(alpha1)

    par = {
        'inst': ParSpec(ParKind.GpuThreads(seqs["num_instances"])),
        'state': ParSpec(ParKind.GpuThreads(hmm["num_states"]))
    }
    forward_init(hmm, seqs, alpha1, parallelize=par, cache=False)

    par = {
        'inst': ParSpec(ParKind.GpuThreads(seqs["num_instances"])),
        'state': ParSpec(ParKind.GpuThreads(hmm["num_states"]))
    }
    forward_steps(hmm, seqs, alpha1, alpha2, parallelize=par, cache=False)

    par = {
        'inst': ParSpec(ParKind.GpuThreads(seqs["num_instances"])),
        'state': ParSpec(ParKind.GpuThreads(hmm["num_states"]))
    }
    forward_lse(hmm, seqs, result, alpha1, alpha2, parallelize=par, cache=False)

    return result

# Tests that we can run the Forward algorithm with dummy values
def test_forward():
    hmm = {
        'gamma': torch.tensor(0.5, dtype=torch.float32),
        'trans1': torch.randn((1024, 4), dtype=torch.float32),
        'trans2': torch.randn(16, dtype=torch.float32),
        'output_prob': torch.randn((101, 1024), dtype=torch.float32),
        'initial_prob': torch.randn(1024, dtype=torch.float32),
        'synthetic_248': torch.tensor(0.5, dtype=torch.float32),
        'num_states': torch.tensor(1024, dtype=torch.int64)
    }
    seqs = {
        'data': torch.zeros((8885, 100), dtype=torch.uint8),
        'lens': torch.zeros(100, dtype=torch.int64),
        'maxlen': torch.tensor(0, dtype=torch.int64),
        'num_instances': torch.tensor(1, dtype=torch.int64)
    }
    forward(hmm, seqs)

test_forward()
