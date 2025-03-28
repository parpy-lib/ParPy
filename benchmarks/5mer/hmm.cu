#include <algorithm>
#include <cstdint>
#include <cstdio>
typedef uint16_t state_t;
typedef uint8_t obs_t;
typedef float prob_t;
#define BATCH_SIZE 1024
#define BATCH_OVERLAP 128
#define NUM_STATES 16384
#define NUM_OBS 101
#define NUM_PREDS 5
#define trans1_SIZE 4096
#define trans2_SIZE 16
#define outputProb_SIZE 103424
#define initialProb_SIZE 16384
__device__
prob_t t_gamma;
__device__
prob_t t_trans1[4096];
__device__
prob_t t_trans2[16];
__device__
prob_t t_outputProb[103424];
__device__
prob_t t_initialProb[16384];
__device__
prob_t t_synthetic_247;
extern "C"
void init(prob_t gamma_arg, prob_t (*trans1_arg), prob_t (*trans2_arg), prob_t (*outputProb_arg), prob_t (*initialProb_arg), prob_t synthetic_247_arg) {
  cudaMemcpyToSymbol(t_gamma, (&gamma_arg), (sizeof(prob_t)));
  cudaMemcpyToSymbol(t_trans1, trans1_arg, (trans1_SIZE * (sizeof(prob_t))));
  cudaMemcpyToSymbol(t_trans2, trans2_arg, (trans2_SIZE * (sizeof(prob_t))));
  cudaMemcpyToSymbol(t_outputProb, outputProb_arg, (outputProb_SIZE * (sizeof(prob_t))));
  cudaMemcpyToSymbol(t_initialProb, initialProb_arg, (initialProb_SIZE * (sizeof(prob_t))));
  cudaMemcpyToSymbol(t_synthetic_247, (&synthetic_247_arg), (sizeof(prob_t)));
}
__device__
prob_t init_prob(state_t x) {
  return (t_initialProb[x]);
}
__device__
prob_t output_prob(state_t x, obs_t o) {
  return (t_outputProb[((o * 1024) + (x % 1024))]);
}
__device__
prob_t transition_prob(state_t x, state_t y) {
  return ((t_trans1[(((x % 1024) * 4) + (y % 4))]) + (t_trans2[(y / 1024)]));
}
__device__
prob_t transition_prob1(state_t x, state_t y) {
  return t_gamma;
}
__device__
prob_t transition_prob2(state_t x, state_t y) {
  return t_synthetic_247;
}
__device__
prob_t transition_prob3(state_t x, state_t y) {
  return 0.;
}
__device__
int forward_prob_predecessors(const prob_t (*alpha_prev), int instance, state_t state, prob_t (*probs)) {
  int pidx = 0;
  state_t pred;
  prob_t p;
  {
    state_t x1 = 0;
    while ((x1 < 4)) {
      {
        (pred = (((0 * 1024) + (x1 * 256)) + ((state / 4) % 256)));
        (p = transition_prob(pred, state));
        {
          ((probs[pidx]) = (p + (alpha_prev[((instance * NUM_STATES) + pred)])));
          (pidx = (pidx + 1));
        }
      }
      (x1 = (x1 + 1));
    }
  }
  {
    if (((state / 1024) == 15)) {
      {
        (pred = state);
        (p = transition_prob1(pred, state));
        ;
      }
    } else {
      
    }
    if (((state / 1024) == 14)) {
      {
        (pred = ((((state / 1024) + 1) * 1024) + (state % 1024)));
        (p = transition_prob2(pred, state));
        ;
      }
    } else {
      
    }
    if ((((state / 1024) != 14) && ((state / 1024) != 15))) {
      {
        (pred = ((((state / 1024) + 1) * 1024) + (state % 1024)));
        (p = transition_prob3(pred, state));
        ;
      }
    } else {
      
    }
    {
      ((probs[pidx]) = (p + (alpha_prev[((instance * NUM_STATES) + pred)])));
      (pidx = (pidx + 1));
    }
  }
  return pidx;
}
__device__
void viterbi_max_predecessor(const prob_t (*chi_prev), int instance1, state_t state, state_t (*maxs), prob_t (*maxp)) {
  state_t pred;
  prob_t p;
  {
    state_t x1 = 0;
    while ((x1 < 4)) {
      {
        (pred = (((0 * 1024) + (x1 * 256)) + ((state / 4) % 256)));
        (p = transition_prob(pred, state));
        {
          (p = (p + (chi_prev[((instance1 * NUM_STATES) + pred)])));
          if ((p > (*maxp))) {
            ((*maxs) = pred);
            ((*maxp) = p);
          } else {
            
          }
        }
      }
      (x1 = (x1 + 1));
    }
  }
  {
    if (((state / 1024) == 15)) {
      {
        (pred = state);
        (p = transition_prob1(pred, state));
        ;
      }
    } else {
      
    }
    if (((state / 1024) == 14)) {
      {
        (pred = ((((state / 1024) + 1) * 1024) + (state % 1024)));
        (p = transition_prob2(pred, state));
        ;
      }
    } else {
      
    }
    if ((((state / 1024) != 14) && ((state / 1024) != 15))) {
      {
        (pred = ((((state / 1024) + 1) * 1024) + (state % 1024)));
        (p = transition_prob3(pred, state));
        ;
      }
    } else {
      
    }
    {
      (p = (p + (chi_prev[((instance1 * NUM_STATES) + pred)])));
      if ((p > (*maxp))) {
        ((*maxs) = pred);
        ((*maxp) = p);
      } else {
        
      }
    }
  }
}
////////////////////////////
// GENERAL IMPLEMENTATION //
////////////////////////////

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "[%s:%d] CUDA error: %s", file, line, cudaGetErrorString(code));
    exit(code);
  }
}

__global__
void forward_init(
    const obs_t* __restrict__ obs, int maxlen, prob_t* __restrict__ alpha_zero) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    obs_t x = obs[instance * maxlen];
    alpha_zero[instance * NUM_STATES + state] =
      init_prob(state) + output_prob(state, x);
  }
}

__device__
prob_t log_sum_exp(const prob_t* probs, const prob_t neginf) {
  prob_t maxp = probs[0];
  for (int i = 1; i < NUM_PREDS; i++) {
    if (probs[i] > maxp) maxp = probs[i];
  }
  if (maxp == neginf) return maxp;
  prob_t sum = 0.0;
  for (int i = 0; i < NUM_PREDS; i++) {
    sum += expf(probs[i] - maxp);
  }
  return maxp + logf(sum);
}

__global__ void forward_step(
    const obs_t* __restrict__ obs, const int* __restrict__ obs_lens, int maxlen,
    const prob_t* __restrict__ alpha_prev, prob_t* __restrict__ alpha_curr,
    int t, const prob_t neginf) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    size_t idx = instance * NUM_STATES + state;
    if (t < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t];
      prob_t probs[NUM_PREDS];
      int pidx = forward_prob_predecessors(alpha_prev, instance, state, probs);
      while (pidx < NUM_PREDS) probs[pidx++] = neginf;
      alpha_curr[idx] = log_sum_exp(probs, neginf) + output_prob(state, x);
    } else if (t == obs_lens[instance]) {
      // We only need to copy the alpha data once - past this point, both alpha
      // vectors will contain the same data.
      alpha_curr[idx] = alpha_prev[idx];
    }
  }
}

__device__
void forward_max_warp_reduce(volatile prob_t *maxp, unsigned int tid) {
  if (maxp[tid + 32] > maxp[tid]) {
    maxp[tid] = maxp[tid + 32];
  }
  if (maxp[tid + 16] > maxp[tid]) {
    maxp[tid] = maxp[tid + 16];
  }
  if (maxp[tid + 8] > maxp[tid]) {
    maxp[tid] = maxp[tid + 8];
  }
  if (maxp[tid + 4] > maxp[tid]) {
    maxp[tid] = maxp[tid + 4];
  }
  if (maxp[tid + 2] > maxp[tid]) {
    maxp[tid] = maxp[tid + 2];
  }
  if (maxp[tid + 1] > maxp[tid]) {
    maxp[tid] = maxp[tid + 1];
  }
}

__global__
void forward_max(
    const prob_t* __restrict__ alpha, prob_t* __restrict__ result,
    const prob_t neginf) {
  unsigned int idx = threadIdx.x;
  unsigned int instance = blockIdx.x;
  unsigned int lo = instance * NUM_STATES;

  __shared__ prob_t maxp[512];
  if (idx < NUM_STATES) {
    maxp[idx] = alpha[lo + idx];
  } else {
    maxp[idx] = neginf;
  }
  for (int i = lo + idx + 512; i < lo + NUM_STATES; i += 512) {
    if (alpha[i] > maxp[idx]) {
      maxp[idx] = alpha[i];
    }
  }
  __syncthreads();
  
  if (idx < 256) {
    if (maxp[idx + 256] > maxp[idx]) {
      maxp[idx] = maxp[idx + 256];
    }
  }
  __syncthreads();
  if (idx < 128) {
    if (maxp[idx + 128] > maxp[idx]) {
      maxp[idx] = maxp[idx + 128];
    }
  }
  __syncthreads();
  if (idx < 64) {
    if (maxp[idx + 64] > maxp[idx]) {
      maxp[idx] = maxp[idx + 64];
    }
  }
  __syncthreads();
  if (idx < 32) forward_max_warp_reduce(maxp, idx);

  if (idx == 0) {
    result[instance] = maxp[0];
  }
}

__device__
void forward_sum_warp_reduce(volatile prob_t *psum, unsigned int tid) {
  psum[tid] = psum[tid] + psum[tid + 32];
  psum[tid] = psum[tid] + psum[tid + 16];
  psum[tid] = psum[tid] + psum[tid + 8];
  psum[tid] = psum[tid] + psum[tid + 4];
  psum[tid] = psum[tid] + psum[tid + 2];
  psum[tid] = psum[tid] + psum[tid + 1];
}

__global__
void forward_log_sum_exp(
    const prob_t* __restrict__ alpha, prob_t* __restrict__ result) {
  unsigned int idx = threadIdx.x;
  unsigned int instance = blockIdx.x;
  unsigned int lo = instance * NUM_STATES;

  // Retrieve the maximum value for the current instance, as computed in the
  // max kernel.
  prob_t maxp = result[instance];

  __shared__ prob_t psum[512];
  if (idx < NUM_STATES) {
    psum[idx] = expf(alpha[lo + idx] - maxp);
  } else {
    psum[idx] = 0.0;
  }
  for (int i = lo + idx + 512; i < lo + NUM_STATES; i += 512) {
    psum[idx] = psum[idx] + expf(alpha[i] - maxp);
  }
  __syncthreads();

  // Compute the sum of all these exponents
  if (idx < 256) psum[idx] = psum[idx] + psum[idx + 256];
  __syncthreads();
  if (idx < 128) psum[idx] = psum[idx] + psum[idx + 128];
  __syncthreads();
  if (idx < 64) psum[idx] = psum[idx] + psum[idx + 64];
  __syncthreads();
  if (idx < 32) forward_sum_warp_reduce(psum, idx);

  // The first thread of each block writes the result
  if (idx == 0) {
    result[instance] = maxp + logf(psum[0]);
  }
}

__global__
void viterbi_init(
    const obs_t* __restrict__ obs, int maxlen, prob_t* __restrict__ chi_zero) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    obs_t x = obs[instance * maxlen];
    chi_zero[instance * NUM_STATES + state] =
      init_prob(state) + output_prob(state, x);
  }
}

__global__
void viterbi_init_batch(
    const obs_t* __restrict__ obs, const int* __restrict__ obs_lens, int maxlen,
    const state_t* __restrict__ seq, prob_t* __restrict__ chi_zero, int t,
    const prob_t neginf) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    if (t < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t];
      state_t last_state = seq[instance * maxlen + t - 1];
      if (state == last_state) {
        chi_zero[instance * NUM_STATES + state] = output_prob(state, x);
      } else {
        chi_zero[instance * NUM_STATES + state] = neginf;
      }
    }
  }
}

__global__
void viterbi_forward(
    const obs_t* __restrict__ obs, const int* __restrict__ obs_lens, int maxlen,
    prob_t* __restrict__ chi1, prob_t* __restrict__ chi2,
    state_t* __restrict__ zeta, int t, int k, const prob_t neginf) {
  state_t state = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int instance = blockIdx.y;
  if (state < NUM_STATES) {
    prob_t *chi_prev, *chi_curr;
    if (k % 2 == 0) {
      chi_prev = chi2;
      chi_curr = chi1;
    } else {
      chi_prev = chi1;
      chi_curr = chi2;
    }
    size_t idx = instance * NUM_STATES + state;
    size_t zeta_idx = instance * BATCH_SIZE * NUM_STATES + (k-1) * NUM_STATES + state;
    if (t+k < obs_lens[instance]) {
      obs_t x = obs[instance * maxlen + t + k];
      state_t maxs;
      prob_t maxp = neginf;
      viterbi_max_predecessor(chi_prev, instance, state, &maxs, &maxp);
      chi_curr[idx] = maxp + output_prob(state, x);
      zeta[zeta_idx] = maxs;
    } else if (t+k == obs_lens[instance]) {
      // We only need to copy over chi data once - past this point, we know
      // both chi vectors will contain identical information. We continue
      // setting the zeta matrix as below to ensure we backtrack through it
      // correctly.
      chi_curr[idx] = chi_prev[idx];
      zeta[zeta_idx] = state;
    } else {
      zeta[zeta_idx] = state;
    }
  }
}

__device__
void viterbi_backward_warp_reduce(volatile prob_t *maxp, volatile state_t *maxs, unsigned int tid) {
  if (maxp[tid + 32] > maxp[tid]) {
    maxp[tid] = maxp[tid + 32];
    maxs[tid] = maxs[tid + 32];
  }
  if (maxp[tid + 16] > maxp[tid]) {
    maxp[tid] = maxp[tid + 16];
    maxs[tid] = maxs[tid + 16];
  }
  if (maxp[tid + 8] > maxp[tid]) {
    maxp[tid] = maxp[tid + 8];
    maxs[tid] = maxs[tid + 8];
  }
  if (maxp[tid + 4] > maxp[tid]) {
    maxp[tid] = maxp[tid + 4];
    maxs[tid] = maxs[tid + 4];
  }
  if (maxp[tid + 2] > maxp[tid]) {
    maxp[tid] = maxp[tid + 2];
    maxs[tid] = maxs[tid + 2];
  }
  if (maxp[tid + 1] > maxp[tid]) {
    maxp[tid] = maxp[tid + 1];
    maxs[tid] = maxs[tid + 1];
  }
}

__global__
void viterbi_backward(
    const prob_t* __restrict__ chi, const state_t* __restrict__ zeta,
    state_t* __restrict__ out, int maxlen, int T, const prob_t neginf) {
  size_t idx = threadIdx.x;
  size_t instance = blockIdx.x;
  size_t lo = instance * NUM_STATES;

  __shared__ state_t maxs[512];
  __shared__ prob_t maxp[512];
  maxs[idx] = idx;
  if (idx < NUM_STATES) {
    maxp[idx] = chi[lo + idx];
  } else {
    maxp[idx] = neginf;
  }
  for (int i = lo + idx + 512; i < lo + NUM_STATES; i += 512) {
    if (chi[i] > maxp[idx]) {
      maxp[idx] = chi[i];
      maxs[idx] = i - lo;
    }
  }
  __syncthreads();

  if (idx < 256) {
    if (maxp[idx + 256] > maxp[idx]) {
      maxp[idx] = maxp[idx + 256];
      maxs[idx] = maxs[idx + 256];
    }
  }
  __syncthreads();
  if (idx < 128) {
    if (maxp[idx + 128] > maxp[idx]) {
      maxp[idx] = maxp[idx + 128];
      maxs[idx] = maxs[idx + 128];
    }
  }
  __syncthreads();
  if (idx < 64) {
    if (maxp[idx + 64] > maxp[idx]) {
      maxp[idx] = maxp[idx + 64];
      maxs[idx] = maxs[idx + 64];
    }
  }
  __syncthreads();
  if (idx < 32) viterbi_backward_warp_reduce(maxp, maxs, idx);

  // Run the backtracking sequentially from the maximum state using the first
  // thread for each instance.
  if (idx == 0) {
    state_t max_state = maxs[0];
    state_t *outptr = out + instance * maxlen + T;
    const state_t *zetaptr = zeta + instance * BATCH_SIZE * NUM_STATES;
    outptr[BATCH_SIZE-1] = max_state;
    for (int t = BATCH_SIZE-2; t >= 0; t--) {
      outptr[t] = zetaptr[t * NUM_STATES + outptr[t+1]];
    }
  }
}

extern "C"
prob_t *forward(obs_t *obs, int *obs_lens, int maxlen, int num_instances) {
  // Copy data to the GPU
  int *cu_obs_lens;
  gpuErrchk(cudaMalloc(&cu_obs_lens, num_instances * sizeof(int)));
  gpuErrchk(cudaMemcpy(cu_obs_lens, obs_lens, num_instances * sizeof(int), cudaMemcpyHostToDevice));
  obs_t *cu_obs;
  gpuErrchk(cudaMalloc(&cu_obs, num_instances * maxlen * sizeof(obs_t)));
  gpuErrchk(cudaMemcpy(cu_obs, obs, num_instances * maxlen * sizeof(obs_t), cudaMemcpyHostToDevice));

  // Allocate working data on the GPU
  size_t alpha_size = num_instances * NUM_STATES * sizeof(prob_t);
  prob_t *alpha_src, *alpha_dst;
  gpuErrchk(cudaMalloc(&alpha_src, alpha_size));
  gpuErrchk(cudaMalloc(&alpha_dst, alpha_size));
  prob_t *cu_result;
  gpuErrchk(cudaMalloc(&cu_result, num_instances * sizeof(prob_t)));

  // Run the Forward algorithm using the kernels defined above
  int tpb = 256;
  int xblocks = (NUM_STATES + tpb - 1) / tpb;
  dim3 blockdim(xblocks, num_instances, 1);
  dim3 threaddim(tpb, 1, 1);
  const prob_t neginf = -1.0/0.0;

  // Intialization step
  forward_init<<<blockdim, threaddim>>>(cu_obs, maxlen, alpha_src);
  gpuErrchk(cudaPeekAtLastError());

  // Forward step
  for (int t = 1; t < maxlen; t++) {
    forward_step<<<blockdim, threaddim>>>(cu_obs, cu_obs_lens, maxlen, alpha_src, alpha_dst, t, neginf);
    gpuErrchk(cudaPeekAtLastError());
    std::swap(alpha_src, alpha_dst);
  }

  // LogSumExp step
  dim3 reduce_blockdim(num_instances, 1, 1);
  dim3 reduce_threaddim(512, 1, 1);
  forward_max<<<reduce_blockdim, reduce_threaddim>>>(alpha_src, cu_result, neginf);
  gpuErrchk(cudaPeekAtLastError());
  forward_log_sum_exp<<<reduce_blockdim, reduce_threaddim>>>(alpha_src, cu_result);
  gpuErrchk(cudaPeekAtLastError());

  // Copy result from the GPU
  prob_t *result = (prob_t*)malloc(num_instances * sizeof(prob_t));
  gpuErrchk(cudaMemcpy(result, cu_result, num_instances * sizeof(prob_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaDeviceSynchronize());

  // Free allocated data on the GPU
  cudaFree(cu_obs_lens);
  cudaFree(cu_obs);
  cudaFree(alpha_src);
  cudaFree(alpha_dst);
  cudaFree(cu_result);

  return result;
}

extern "C"
state_t *viterbi(
    obs_t *obs, int *obs_lens, int maxlen, int num_parallel, int num_instances) {
  // Allocate observation data on the GPU
  int *cu_obs_lens;
  gpuErrchk(cudaMalloc(&cu_obs_lens, num_parallel * sizeof(int)));
  gpuErrchk(cudaMemcpy(cu_obs_lens, obs_lens, num_parallel * sizeof(int), cudaMemcpyHostToDevice));
  obs_t *cu_obs;
  gpuErrchk(cudaMalloc(&cu_obs, num_parallel * maxlen * sizeof(obs_t)));
  gpuErrchk(cudaMemcpy(cu_obs, obs, num_parallel * maxlen * sizeof(obs_t), cudaMemcpyHostToDevice));

  // Allocate working data on the GPU
  size_t chi_sz = num_parallel * NUM_STATES * sizeof(prob_t);
  prob_t *chi_src, *chi_dst;
  gpuErrchk(cudaMalloc(&chi_src, chi_sz));
  gpuErrchk(cudaMalloc(&chi_dst, chi_sz));
  state_t *zeta;
  gpuErrchk(cudaMalloc(&zeta, num_parallel * BATCH_SIZE * NUM_STATES * sizeof(state_t)));
  state_t *cu_result;
  gpuErrchk(cudaMalloc(&cu_result, num_parallel * maxlen * sizeof(state_t)));

  // Pre-allocate result storage, and pin host memory to enable asynchronous
  // copying of data.
  state_t *result = (state_t*)malloc(num_instances * maxlen * sizeof(state_t));
  gpuErrchk(cudaHostRegister(obs, num_instances * maxlen * sizeof(obs_t), 0));
  gpuErrchk(cudaHostRegister(obs_lens, num_instances * sizeof(int), 0));

  // Run all the Viterbi instances.
  for (int i = 0; i < num_instances; i += num_parallel) {
    // Copy data to the GPU for the current parallel slice
    size_t slicesz = std::min(num_instances - i, num_parallel);
    gpuErrchk(cudaMemcpyAsync(cu_obs_lens, obs_lens + i, slicesz * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyAsync(cu_obs, obs + i * maxlen, slicesz * maxlen * sizeof(obs_t), cudaMemcpyHostToDevice));

    // Run the Viterbi algorithm using the kernels defined above
    int tpb = 256;
    int xblocks = (NUM_STATES + tpb - 1) / tpb;
    dim3 blockdim(xblocks, slicesz, 1);
    dim3 threaddim(tpb, 1, 1);
    const prob_t neginf = -1.0/0.0;

    int bos = BATCH_SIZE - BATCH_OVERLAP;
    int slice_maxlen = *std::max_element(obs_lens + i, obs_lens + i + slicesz);
    int nbatches = (int)std::ceil((double)slice_maxlen / (double)bos);
    for (int b = 0; b < nbatches; b++) {
      int t = b * bos;
      // Initialize the batch differently depending on whether it is the first
      // batch or not.
      if (b == 0) {
        viterbi_init<<<blockdim, threaddim>>>(cu_obs, maxlen, chi_src);
      } else {
        viterbi_init_batch<<<blockdim, threaddim>>>(cu_obs, cu_obs_lens, maxlen, cu_result, chi_src, t, neginf);
      }
      gpuErrchk(cudaPeekAtLastError());

      // Run the forward pass
      for (int k = 1; k < BATCH_SIZE; k++) {
        viterbi_forward<<<blockdim, threaddim>>>(cu_obs, cu_obs_lens, maxlen, chi_src, chi_dst, zeta, t, k, neginf);
        gpuErrchk(cudaPeekAtLastError());
      }

      // Run the backward pass
      dim3 backw_blockdim(slicesz, 1, 1);
      dim3 backw_threaddim(512, 1, 1);
      viterbi_backward<<<backw_blockdim, backw_threaddim>>>(chi_src, zeta, cu_result, maxlen, t, neginf);
      gpuErrchk(cudaPeekAtLastError());
    }

    // Copy result from the GPU to the result container
    gpuErrchk(cudaMemcpyAsync(result + i * maxlen, cu_result, slicesz * maxlen * sizeof(state_t), cudaMemcpyDeviceToHost));
  }

  // Synchronize after launching all kernels
  gpuErrchk(cudaDeviceSynchronize());

  // Free up allocated data on the GPU
  gpuErrchk(cudaFree(cu_obs_lens));
  gpuErrchk(cudaFree(cu_obs));
  gpuErrchk(cudaFree(chi_src));
  gpuErrchk(cudaFree(chi_dst));
  gpuErrchk(cudaFree(zeta));
  gpuErrchk(cudaFree(cu_result));

  // Unpin the CPU data
  gpuErrchk(cudaHostUnregister(obs));
  gpuErrchk(cudaHostUnregister(obs_lens));

  return result;
}
