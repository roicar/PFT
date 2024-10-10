#include "2mm_kernel.hu"
__global__ void kernel0(float *A, float *B, float *C)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    float private_C[1][2];

    {
      private_C[0][0] = C[(32 * b0 + t0) * 1024 + (32 * b1 + t1)];
      private_C[0][1] = C[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)];
      for (int c2 = 0; c2 <= 1023; c2 += 32) {
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_A[t0][c4] = A[(32 * b0 + t0) * 1024 + (c2 + c4)];
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_B[t0][c4] = B[(t0 + c2) * 1024 + (32 * b1 + c4)];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_C[0][0] = (private_C[0][0] + (shared_A[t0][c3] * shared_B[c3][t1]));
          private_C[0][1] = (private_C[0][1] + (shared_A[t0][c3] * shared_B[c3][t1 + 16]));
        }
        __syncthreads();
      }
      C[(32 * b0 + t0) * 1024 + (32 * b1 + t1)] = private_C[0][0];
      C[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)] = private_C[0][1];
    }
}
__global__ void kernel1(float *C, float *D, float *E)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_C[32][32];
    float private_D[1][2];
    __shared__ float shared_E[32][32];

    {
      private_D[0][0] = D[(32 * b0 + t0) * 1024 + (32 * b1 + t1)];
      private_D[0][1] = D[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)];
      for (int c2 = 0; c2 <= 1023; c2 += 32) {
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_C[t0][c4] = C[(t0 + c2) * 1024 + (32 * b1 + c4)];
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_E[t0][c4] = E[(32 * b0 + t0) * 1024 + (c2 + c4)];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_D[0][0] = (private_D[0][0] + (shared_E[t0][c3] * shared_C[c3][t1]));
          private_D[0][1] = (private_D[0][1] + (shared_E[t0][c3] * shared_C[c3][t1 + 16]));
        }
        __syncthreads();
      }
      D[(32 * b0 + t0) * 1024 + (32 * b1 + t1)] = private_D[0][0];
      D[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)] = private_D[0][1];
    }
}
