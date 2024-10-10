#include "bicg_kernel.hu"
__global__ void kernel0(float *A, float *r, float *s)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_r[32];
    float private_s[1];

    {
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        shared_r[t0] = r[t0 + c1];
        __syncthreads();
        if (c1 == 0)
          private_s[0] = 0;
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_s[0] = (private_s[0] + (shared_r[c3] * A[(c1 + c3) * 1024 + (32 * b0 + t0)]));
        __syncthreads();
      }
      s[32 * b0 + t0] = private_s[0];
    }
}
__global__ void kernel1(float *A, float *p, float *q)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_p[32];
    float private_q[1];

    {
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        for (int c2 = 0; c2 <= 31; c2 += 1)
          shared_A[c2][t0] = A[(32 * b0 + c2) * 1024 + (t0 + c1)];
        shared_p[t0] = p[t0 + c1];
        __syncthreads();
        if (c1 == 0)
          private_q[0] = 0;
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_q[0] = (private_q[0] + (shared_A[t0][c3] * shared_p[c3]));
        __syncthreads();
      }
      q[32 * b0 + t0] = private_q[0];
    }
}
