#include "nat_ppcg_kernel.hu"
__global__ void kernel0(float *A, float *B, float *C)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[311];

    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (b0 >= 10) {
      for (int c1 = -32 * ((3 * b0 + 3) / 7) + 192; c1 <= ppcg_min(73, -14 * b0 + (2 * b0 + 1) / 7 + 219); c1 += 32) {
        for (int c2 = t0; c2 <= -96 * b0 - 7 * c1 + 1534; c2 += 32)
          shared_A[c2] = A[96 * b0 + 7 * c1 + c2 - 511];
        __syncthreads();
        if (96 * b0 + 3 * t0 + 7 * c1 >= 1311 && 96 * b0 + 3 * t0 + 7 * c1 <= 1534) {
          if (32 * b0 + t0 >= 339) {
            C[32 * b0 + t0] = (shared_A[-98 * b0 + 3 * t0 - 7 * c1 - 7 * ppcg_fdiv_q(-2 * b0 + 3 * t0 - 2, 7) + 1526] * B[64 * b0 + 2 * t0]);
          } else {
            C[t0 + 320] = (shared_A[3 * t0 + 63] * B[2 * t0 + 640]);
          }
        }
        __syncthreads();
      }
    } else {
      for (int c2 = ppcg_max(t0, ((t0 + 1) % 32) - 96 * b0 + 63); c2 <= ppcg_min(310, -96 * b0 + 1086); c2 += 32)
        shared_A[c2] = A[96 * b0 + c2 - 63];
      __syncthreads();
      C[32 * b0 + t0] = (shared_A[3 * t0 + 63] * B[64 * b0 + 2 * t0]);
      __syncthreads();
    }
}
