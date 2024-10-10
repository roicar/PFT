#include <assert.h>
#include <stdio.h>
#include "2mm_kernel.hu"
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
    float A[1024][1024], B[1024][1024], C[1024][1024], D[1024][1024], E[1024][1024];
    double Total_time = 0.0;
    srand((unsigned)time(NULL));
    Init_Matrix_zero_PSxM(C, 1024,1024);
    Init_Matrix_rand_PSxM(A, 1024,1024);
    Init_Matrix_rand_PSxM(B, 1024,1024);
    Init_Matrix_rand_PSxM(D, 1024,1024);
    Init_Matrix_rand_PSxM(E, 1024,1024);
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

      float *dev_A;
      float *dev_B;
      float *dev_C;
      float *dev_D;
      float *dev_E;
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      
      cudaCheckReturn(cudaMalloc((void **) &dev_A, (1024) * (1024) * sizeof(float)));
      cudaCheckReturn(cudaMalloc((void **) &dev_B, (1024) * (1024) * sizeof(float)));
      cudaCheckReturn(cudaMalloc((void **) &dev_C, (1024) * (1024) * sizeof(float)));
      cudaCheckReturn(cudaMalloc((void **) &dev_D, (1024) * (1024) * sizeof(float)));
      cudaCheckReturn(cudaMalloc((void **) &dev_E, (1024) * (1024) * sizeof(float)));
      
      cudaCheckReturn(cudaMemcpy(dev_A, A, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_B, B, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_C, C, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_D, D, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_E, E, (1024) * (1024) * sizeof(float), cudaMemcpyHostToDevice));
      {
        dim3 k0_dimBlock(16, 32);
        dim3 k0_dimGrid(32, 32);
        cudaEventRecord(start,0);
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B, dev_C);
        cudaCheckKernel();
      }
      
      {
        dim3 k1_dimBlock(16, 32);
        dim3 k1_dimGrid(32, 32);
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_C, dev_D, dev_E);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaCheckKernel();
        cudaEventElapsedTime(&milliseconds, start, stop);
      }
      
      cudaCheckReturn(cudaMemcpy(C, dev_C, (1024) * (1024) * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheckReturn(cudaMemcpy(D, dev_D, (1024) * (1024) * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheckReturn(cudaFree(dev_A));
      cudaCheckReturn(cudaFree(dev_B));
      cudaCheckReturn(cudaFree(dev_C));
      cudaCheckReturn(cudaFree(dev_D));
      cudaCheckReturn(cudaFree(dev_E));
    }

    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%.10f\n", elapsed);
    printf("%.10f\n", milliseconds/1000);

    return 0;
}
