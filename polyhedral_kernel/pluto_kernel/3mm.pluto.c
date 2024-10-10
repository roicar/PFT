#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

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

/* Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
  int t1, t2, t3, t4;
 register int lbv, ubv;
/* Start of CLooG code */
if ((PS1 >= 1) && (PS2 >= 1) && (PS3 >= 1)) {
  for (t2=0;t2<=PS3-1;t2++) {
    for (t3=0;t3<=PS1-1;t3++) {
      lbv=0;
      ubv=PS2-1;
#pragma ivdep
#pragma vector always
      for (t4=lbv;t4<=ubv;t4++) {
        F[t2][t4] += D[t2][t3] * E[t3][t4];;
      }
    }
  }
  for (t2=0;t2<=PS1-1;t2++) {
    for (t3=0;t3<=PS2-1;t3++) {
      lbv=0;
      ubv=PS3-1;
#pragma ivdep
#pragma vector always
      for (t4=lbv;t4<=ubv;t4++) {
        C[t2][t4] += A[t2][t3] * B[t3][t4];;
      }
    }
  }
  for (t2=0;t2<=PS1-1;t2++) {
    for (t3=0;t3<=PS3-1;t3++) {
      lbv=0;
      ubv=PS2-1;
#pragma ivdep
#pragma vector always
      for (t4=lbv;t4<=ubv;t4++) {
        G[t2][t4] += C[t2][t3] * F[t3][t4];;
      }
    }
  }
}
/* End of CLooG code */

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

