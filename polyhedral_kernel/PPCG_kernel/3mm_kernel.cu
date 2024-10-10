#include "3mm_kernel.hu"
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
          private_C[0][0] += (shared_A[t0][c3] * shared_B[c3][t1]);
          private_C[0][1] += (shared_A[t0][c3] * shared_B[c3][t1 + 16]);
        }
        __syncthreads();
      }
      C[(32 * b0 + t0) * 1024 + (32 * b1 + t1)] = private_C[0][0];
      C[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)] = private_C[0][1];
    }
}
__global__ void kernel1(float *D, float *E, float *F)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_D[32][32];
    __shared__ float shared_E[32][32];
    float private_F[1][2];

    {
      private_F[0][0] = F[(32 * b0 + t0) * 1024 + (32 * b1 + t1)];
      private_F[0][1] = F[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)];
      for (int c2 = 0; c2 <= 1023; c2 += 32) {
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_D[t0][c4] = D[(32 * b0 + t0) * 1024 + (c2 + c4)];
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_E[t0][c4] = E[(t0 + c2) * 1024 + (32 * b1 + c4)];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_F[0][0] += (shared_D[t0][c3] * shared_E[c3][t1]);
          private_F[0][1] += (shared_D[t0][c3] * shared_E[c3][t1 + 16]);
        }
        __syncthreads();
      }
      F[(32 * b0 + t0) * 1024 + (32 * b1 + t1)] = private_F[0][0];
      F[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)] = private_F[0][1];
    }
}
__global__ void kernel2(float *C, float *F, float *G)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_C[32][32];
    __shared__ float shared_F[32][32];
    float private_G[1][2];

    {
      private_G[0][0] = G[(32 * b0 + t0) * 1024 + (32 * b1 + t1)];
      private_G[0][1] = G[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)];
      for (int c2 = 0; c2 <= 1023; c2 += 32) {
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_C[t0][c4] = C[(32 * b0 + t0) * 1024 + (c2 + c4)];
        for (int c4 = t1; c4 <= 31; c4 += 16)
          shared_F[t0][c4] = F[(t0 + c2) * 1024 + (32 * b1 + c4)];
        __syncthreads();
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_G[0][0] += (shared_C[t0][c3] * shared_F[c3][t1]);
          private_G[0][1] += (shared_C[t0][c3] * shared_F[c3][t1 + 16]);
        }
        __syncthreads();
      }
      G[(32 * b0 + t0) * 1024 + (32 * b1 + t1)] = private_G[0][0];
      G[(32 * b0 + t0) * 1024 + (32 * b1 + t1 + 16)] = private_G[0][1];
    }
}
