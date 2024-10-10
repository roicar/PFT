#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define Smax 1
#define Smin 0
#define alpha 0.6

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

int main(int argc, char *argv[]) {
    float **A, **B, **C, **D, **E;
    FILE *fp = NULL;
    float Total_time = 0.0;
    srand((unsigned)time(NULL));
    int PS1 = atoi(argv[1]);
    int PS2 = atoi(argv[2]);
    int PS3 = atoi(argv[3]);
    int I = atoi(argv[4]);
    int J = atoi(argv[5]);
    int K = atoi(argv[6]);
    int order = atoi(argv[7]);
    int schorder = atoi(argv[8]);
    C = Init_Matrix_zero_PSxM(C, PS1, PS3);
    A = Init_Matrix_rand_PSxM(A, PS1, PS2);
    B = Init_Matrix_rand_PSxM(B, PS2, PS3);
    D = Init_Matrix_rand_PSxM(D, PS2, PS3);
    E = Init_Matrix_rand_PSxM(E, PS2, PS1);
    struct timespec start, finish;
    float elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma scop
    for (int i = 0; i < PS1; i++) {
        for (int j = 0; j < PS3; j++) {
            for (int k = 0; k < PS2; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }


    for (int i = 0; i < PS2; i++) {
        for (int j = 0; j < PS3; j++) {
            for (int k = 0; k < PS1; k++) {
                D[i][j] = D[i][j] + E[i][k] * C[k][j];
            }
        }
    }
    #pragma endscop
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, schorder, elapsed);
    fp = fopen("time.txt", "a+");
    fprintf(fp, " %d %d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, schorder, elapsed);
    fclose(fp);
    free_Matrix(A, PS1);
    free_Matrix(B, PS2);
    free_Matrix(C, PS1);
    free_Matrix(D, PS2);
    free_Matrix(E, PS2);
    return 0;
}

