#include "polynomialmul_kernel.hu"
__global__ void kernel0(float *a, float *b, float *result)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_a[32];
    __shared__ float shared_b[63];
    float private_result[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    {
      if (b0 >= 33) {
        shared_a[t0] = a[t0];
        __syncthreads();
        if (32 * b0 + t0 <= 2048)
          private_result[0] = 0.0;
        __syncthreads();
      }
      for (int c1 = ppcg_max(0, 32 * b0 - 1024); c1 <= ppcg_min(1023, 32 * b0 + 31); c1 += 32) {
        shared_a[t0] = a[t0 + c1];
        for (int c2 = ppcg_max(t0, -32 * b0 + t0 + 32 * ((-t0 + c1 + 62) / 32)); c2 <= ppcg_min(62, -32 * b0 + c1 + 1054); c2 += 32)
          shared_b[c2] = b[32 * b0 - c1 + c2 - 31];
        __syncthreads();
        if (c1 == 0)
          private_result[0] = 0.0;
        for (int c3 = ppcg_max(0, 32 * b0 + t0 - c1 - 1023); c3 <= ppcg_min(31, 32 * b0 + t0 - c1); c3 += 1)
          private_result[0] += (shared_a[c3] * shared_b[t0 - c3 + 31]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2048)
        result[32 * b0 + t0] = private_result[0];
    }
}
