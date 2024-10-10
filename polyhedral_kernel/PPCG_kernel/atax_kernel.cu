#include "atax_kernel.hu"
__global__ void kernel0(float *A, float *tmp, float *x)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    float private_tmp[1];
    __shared__ float shared_x[32];

    {
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        for (int c2 = 0; c2 <= 31; c2 += 1)
          shared_A[c2][t0] = A[(32 * b0 + c2) * 1024 + (t0 + c1)];
        shared_x[t0] = x[t0 + c1];
        __syncthreads();
        if (c1 == 0)
          private_tmp[0] = 0;
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_tmp[0] = (private_tmp[0] + (shared_A[t0][c3] * shared_x[c3]));
        __syncthreads();
      }
      tmp[32 * b0 + t0] = private_tmp[0];
    }
}
__global__ void kernel1(float *A, float *tmp, float *y)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_tmp[32];
    float private_y[1];

    {
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        shared_tmp[t0] = tmp[t0 + c1];
        __syncthreads();
        if (c1 == 0)
          private_y[0] = 0;
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_y[0] = (private_y[0] + (A[(c1 + c3) * 1024 + (32 * b0 + t0)] * shared_tmp[c3]));
        __syncthreads();
      }
      y[32 * b0 + t0] = private_y[0];
    }
}
