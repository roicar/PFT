#include <stdio.h>
#include "math.h"
#include <stdlib.h>
#include "time.h"
#define Smax 1
#define Smin 0


float **Init_Matrix_rand_PSxM(float **matrix, int PS, int M) {
    matrix = (float **)malloc(sizeof(float *) * PS);
    for (int i = 0; i < PS; ++i) {
        matrix[i] = (float *)malloc(sizeof(float) * M);
    }
    for (int i = 0; i < PS; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix[i][j] = (float)rand() / 32767 * (Smax - Smin) + Smin;
        }
    }
    return matrix;
}

float **Init_Matrix_zero_PSxM(float **matrix, int PS, int M) {
    matrix = (float **)malloc(sizeof(float *) * PS);
    for (int i = 0; i < PS; ++i) {
        matrix[i] = (float *)malloc(sizeof(float) * M);
    }
    for (int i = 0; i < PS; ++i) {
        for (int j = 0; j < M; ++j) {
            matrix[i][j] = 0;
        }
    }
    return matrix;
}

void free_Matrix(float **matrix, int PS) {
    for (int i = 0; i < PS; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}
int main(int argc,char *argv[]){
    int i,j;
    float **a;
    float **b;
    FILE *fp = NULL;
    float Total_time = 0.0;
    srand((unsigned)time(NULL));
    int PS3 =atoi(argv[1]);
    int PS1 =atoi(argv[2]);
    int PS2 =atoi(argv[3]);
    int I = atoi(argv[4]);
    int J = atoi(argv[5]);
    int K = atoi(argv[6]);
    int order = atoi(argv[7]);
    a = Init_Matrix_rand_PSxM(a,PS1,PS2);
    b = Init_Matrix_zero_PSxM(b,PS1,PS2);
    struct timespec start, finish;
    float elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
#pragma scop
        for (i=2; i<=PS1-3; i++)  {
            for (j=2; j<=PS2-3; j++)  {
                b[i][j] = (a[i-2][j-2] + a[i-2][j-1] + a[i-2][j] + a[i-2][j+1] + a[i-2][j+2]
          + a[i-1][j-2] + a[i-1][j-1] + a[i-1][j] + a[i-1][j+1] + a[i-1][j+2]
          + a[i][j-2] + a[i][j-1] + a[i][j] + a[i][j+1] + a[i][j+2]
          + a[i+1][j-2] + a[i+1][j-1] + a[i+1][j] + a[i+1][j+1] + a[i+1][j+2]
          + a[i+2][j-2] + a[i+2][j-1] + a[i+2][j] + a[i+2][j+1] + a[i+2][j+2]) / 25.0;

            }
        }
#pragma endscop

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, elapsed);
    fp = fopen("seidelOut.txt", "a+");
    fprintf(fp," %d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, elapsed);
    fclose(fp);
    free_Matrix(a,PS1);
    return 0;
}
