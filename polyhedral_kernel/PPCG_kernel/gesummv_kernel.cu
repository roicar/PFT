#include "gesummv_kernel.hu"
__global__ void kernel0(float *A, float *B, float *tmp, float *x, float *y)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    float private_tmp[1];
    float private_y[1];

    {
      for (int c1 = 0; c1 <= 1023; c1 += 32) {
        for (int c2 = 0; c2 <= 31; c2 += 1)
          shared_A[c2][t0] = A[(32 * b0 + c2) * 1024 + (t0 + c1)];
        for (int c2 = 0; c2 <= 31; c2 += 1)
          shared_B[c2][t0] = B[(32 * b0 + c2) * 1024 + (t0 + c1)];
        __syncthreads();
        if (c1 == 0) {
          private_y[0] = 0;
          private_tmp[0] = 0;
        }
        for (int c3 = 0; c3 <= 31; c3 += 1) {
          private_tmp[0] = ((shared_A[t0][c3] * x[c1 + c3]) + private_tmp[0]);
          private_y[0] = ((shared_B[t0][c3] * x[c1 + c3]) + private_y[0]);
        }
        if (c1 == 992)
          private_y[0] = ((32412 * private_tmp[0]) + (2123 * private_y[0]));
        __syncthreads();
      }
      y[32 * b0 + t0] = private_y[0];
      tmp[32 * b0 + t0] = private_tmp[0];
    }
}
