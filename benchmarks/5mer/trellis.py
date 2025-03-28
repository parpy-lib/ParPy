import numpy as np
import math
import ctypes
import time

class HMM:
    def _forward(self, obs, obs_lens, print_times):
        maxlen = int(max(obs_lens))
        num_instances = len(obs_lens)

        if print_times:
            t0 = time.time()
        ptr = self.lib.forward(
            obs, obs_lens, maxlen, num_instances
        )
        ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(self.prob_ctype))
        result = np.ctypeslib.as_array(ctypes_ptr, shape=(num_instances,)).copy()
        self.clib.free(ptr)
        if print_times:
            t1 = time.time()
            print(t1-t0)

        return result

    def _viterbi(self, obs, obs_lens, padded_lens, num_parallel, print_times):
        maxlen = int(max(padded_lens))
        num_instances = len(obs_lens)

        # If we run Viterbi on more than one sequence in parallel, we order
        # them by length to reduce the amount of padding. This can
        # significantly reduce the number of kernels we launch, especially if
        # there is a huge difference in the lengths of the observations,
        # thereby improving performance. For the kmer example, this results in
        # a 30% reduction in the number of kernels we launch.
        if num_parallel > 1 and num_parallel < num_instances:
            idxobs = [(i, x, y, z) for (i, x), y, z in zip(enumerate(obs), obs_lens, padded_lens)]
            ordered_idxobs = sorted(idxobs, key=lambda x: x[2])
            permutation, obs, obs_lens, padded_lens, = zip(*ordered_idxobs)
            obs_lens = np.array(obs_lens, dtype=np.int32)
            padded_lens = np.array(padded_lens, dtype=np.int32)

        # Flatten the observations after potentially reordering them based on
        # length.
        obs = np.array(obs).flatten()

        if print_times:
            t0 = time.time()
        ptr = self.lib.viterbi(
            obs, obs_lens, maxlen, num_parallel, num_instances
        )
        ctypes_ptr = ctypes.cast(ptr, ctypes.POINTER(self.state_ctype))
        result = np.ctypeslib.as_array(ctypes_ptr, shape=(num_instances, maxlen)).copy()
        self.clib.free(ptr)
        if print_times:
            t1 = time.time()
            print(t1-t0)

        # Remove padding of result
        result = [r[:obs_lens[i]] for i, r in enumerate(result)]

        # If we ran more than one instance in parallel, we restore the original
        # order here.
        if num_parallel > 1 and num_parallel < num_instances:
            result_tmp = result.copy()
            for i, p in enumerate(permutation):
                result_tmp[p] = result[i]
            return result_tmp

        return result

    def pad_signals(self, signals, lens):
        n = max(lens)
        ps = np.zeros((len(signals), n), dtype=self.obs_type)
        for i, s in enumerate(signals):
            ps[i][:len(s)] = s
        return ps

    def viterbi(self, signals, num_parallel=1, print_times=False):
        bos = self.batch_size - self.batch_overlap
        lens = np.array([len(x) for x in signals], dtype=np.int32)
        plens = np.array([(n + bos - 1) // bos * bos + self.batch_overlap for n in lens], dtype=np.int32)
        padded_obs = self.pad_signals(signals, plens)
        return self._viterbi(padded_obs, lens, plens, num_parallel, print_times)

    def forward(self, signals, print_times=False):
        lens = np.array([len(x) for x in signals], dtype=np.int32)
        if all([n == lens[0] for n in lens]):
            padded_signals = np.array(signals, dtype=self.obs_type)
        else:
            padded_signals = self.pad_signals(signals, lens)
        return self._forward(padded_signals.flatten(), lens, print_times)

    def setup_library(self):
        self.lib = ctypes.cdll.LoadLibrary("./5mer/libhmm.so")
        self.clib = ctypes.cdll.LoadLibrary("libc.so.6")

        # Declare the argument types of the Forward and Viterbi function calls
        self.lib.forward.argtypes = [
            np.ctypeslib.ndpointer(dtype=self.obs_type, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.forward.restype = np.ctypeslib.ndpointer(dtype=self.prob_type, ndim=1, flags="C_CONTIGUOUS")
        self.lib.viterbi.argtypes = [
            np.ctypeslib.ndpointer(dtype=self.obs_type, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.viterbi.restype = np.ctypeslib.ndpointer(dtype=self.state_type, ndim=1, flags="C_CONTIGUOUS")

    def __init__(self, args):
        self.precompute_predecessors = False
        self.num_states = 16384
        self.num_preds = 5
        self.batch_size = 1024
        self.batch_overlap = 128
        self.state_type = np.uint16
        self.state_ctype = ctypes.c_uint16
        self.prob_type = np.float32
        self.prob_ctype = ctypes.c_float
        self.obs_type = np.uint8
        self.obs_ctype = ctypes.c_uint8
        self.setup_library()
        self.copy_args(args)

    def copy_args(self, args):
        self.lib.init.argtypes = [
            self.prob_ctype,
            np.ctypeslib.ndpointer(dtype=self.prob_type, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=self.prob_type, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=self.prob_type, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=self.prob_type, ndim=1, flags='C_CONTIGUOUS'),
            self.prob_ctype
        ]
        self.lib.init(float(args['gamma']), np.array(args['trans1'].flatten(), dtype=self.prob_type), np.array(args['trans2'].flatten(), dtype=self.prob_type), np.array(args['outputProb'].flatten(), dtype=self.prob_type), np.array(args['initialProb'].flatten(), dtype=self.prob_type), float(np.log(np.exp(0.) - np.exp(args['gamma']))))
