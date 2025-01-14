#include "mvt_kernel.hu"
__global__ void kernel0(float *A, float *x1, float *y_1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    float private_x1[1];
    __shared__ float shared_y_1[32];

    {
      private_x1[0] = x1[32 * b0 + t0];
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        for (int c2 = 0; c2 <= 31; c2 += 1)
          shared_A[c2][t0] = A[(32 * b0 + c2) * 1024 + (t0 + c1)];
        shared_y_1[t0] = y_1[t0 + c1];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_x1[0] = (private_x1[0] + (shared_A[t0][c3] * shared_y_1[c3]));
        __syncthreads();
      }
      x1[32 * b0 + t0] = private_x1[0];
    }
}
__global__ void kernel1(float *A, float *x2, float *y_2)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    float private_x2[1];
    __shared__ float shared_y_2[32];

    {
      private_x2[0] = x2[32 * b0 + t0];
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        shared_y_2[t0] = y_2[t0 + c1];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1)
          private_x2[0] = (private_x2[0] + (A[(c1 + c3) * 1024 + (32 * b0 + t0)] * shared_y_2[c3]));
        __syncthreads();
      }
      x2[32 * b0 + t0] = private_x2[0];
    }
}
