#include <stdio.h>
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
            matrix[i][j] = 0.0f;
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
    float **A, **B, **C, **D, **E, **F, **G;
    FILE *fp = NULL;
    double Total_time = 0.0;
    srand((unsigned)time(NULL));
    int PS1 = atoi(argv[1]);
    int PS2 = atoi(argv[2]);
    int PS3 = atoi(argv[3]);
    int I = atoi(argv[4]);
    int J = atoi(argv[5]);
    int K = atoi(argv[6]);
    int order = atoi(argv[7]);

    C = Init_Matrix_zero_PSxM(C, PS1, PS3);
    A = Init_Matrix_rand_PSxM(A, PS1, PS2);
    B = Init_Matrix_rand_PSxM(B, PS2, PS3);
    D = Init_Matrix_rand_PSxM(D, PS3, PS1);
    E = Init_Matrix_rand_PSxM(E, PS1, PS2);
    F = Init_Matrix_rand_PSxM(F, PS3, PS2);
    G = Init_Matrix_rand_PSxM(G, PS1, PS2);

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

#pragma scop
    for (int i = 0; i < PS1; ++i) {
        for (int j = 0; j < PS3; ++j) {
            for (int k = 0; k < PS2; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < PS3; ++i) {
        for (int j = 0; j < PS2; ++j) {
            for (int k = 0; k < PS1; ++k) {
                F[i][j] += D[i][k] * E[k][j];
            }
        }
    }

    for (int i = 0; i < PS1; ++i) {
        for (int j = 0; j < PS2; ++j) {
            for (int k = 0; k < PS3; ++k) {
                G[i][j] += C[i][k] * F[k][j];
            }
        }
    }
#pragma endscop

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, elapsed);
    fp = fopen("time.txt", "a+");
    fprintf(fp, " %d %d %d %d %d %d %d %.10f\n", PS1, PS2, PS3, I, J, K, order, elapsed);
    fclose(fp);

    free_Matrix(A, PS1);
    free_Matrix(B, PS2);
    free_Matrix(C, PS1);
    free_Matrix(D, PS3);
    free_Matrix(E, PS1);
    free_Matrix(F, PS3);
    free_Matrix(G, PS1);
    return 0;
}

