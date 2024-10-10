#include <assert.h>
#include <stdio.h>
#include "seidel_kernel.hu"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <iostream>
#define Smax 1
#define Smin 0
#define alpha 0.6

void Init_Matrix_rand_PSxM(float matrix[1024][1024], int PS1, int PS2) {
    for (int i = 0; i < PS1; i++) {
        for (int j = 0; j < PS2; j++) {
            matrix[i][j] = (float)rand() / 32767 * (Smax - Smin) + Smin;
        }
    }
}

void Init_Matrix_zero_PSxM(float matrix[1024][1024], int PS1, int PS2) {
    for (int i = 0; i < PS1; i++) {
        for (int j = 0; j < PS2; j++) {
            matrix[i][j] = 0;
        }
    }
}

int main() {
    int i, j;
    float a[1024][1024],b[1024][1024];
    srand((unsigned)time(NULL));

    Init_Matrix_rand_PSxM(a, 1024, 1024);
    Init_Matrix_zero_PSxM(b,1024,1024);
    struct timespec start, finish;
    double elapsed;
    float milliseconds = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);

    {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

      float *dev_a;
      float *dev_b;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      
      cudaCheckReturn(cudaMalloc((void **) &dev_a, (1024) * (1024) * sizeof(float)));
      cudaCheckReturn(cudaMalloc((void **) &dev_b, (1023) * (1024) * sizeof(float)));
      
      cudaCheckReturn(cudaMemcpy(dev_a, a, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_b, b, (1023) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      {
        dim3 k0_dimBlock(16, 32);
        dim3 k0_dimGrid(32, 32);
        cudaEventRecord(start,0);
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_a, dev_b);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaCheckKernel();
        cudaEventElapsedTime(&milliseconds, start, stop);
      }
      
      cudaCheckReturn(cudaMemcpy(b, dev_b, (1023) * (1024) * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheckReturn(cudaFree(dev_a));
      cudaCheckReturn(cudaFree(dev_b));
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%.10f\n", elapsed);
    printf("%.10f\n", milliseconds/1000);
    return 0;
}

