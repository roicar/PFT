#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define Smax 1
#define Smin 0
#define alpha 0.6

int main(int argc, char *argv[]) {
    FILE *fp = NULL;
    float Total_time = 0.0;
    srand((unsigned)time(NULL));
    int PS1 = atoi(argv[1]);
    int PS2 = atoi(argv[2]);
    float *A = (float *)malloc(sizeof(float) * (PS1));
    float *B = (float *)malloc(sizeof(float) * (PS2));
    // Initialize polynomials A and B with random coefficients
    for (int i = 0; i <= PS1; i++) {
        A[i] = (float)rand() / 32767 * (Smax - Smin) + Smin;
    }

    for (int i = 0; i <= PS2; i++) {
        B[i] = (float)rand() / 32767 * (Smax - Smin) + Smin;
    }
    double *C = (double *)malloc(sizeof(double) * (PS1));
    struct timespec start, finish;
    float elapsed;
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
  int t1, t2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((PS1 >= 1) && (PS2 >= 1)) {
  for (t1=0;t1<=min(min(floord(PS2-1,2),floord(4*PS1+PS2-5,14)),PS1-1);t1++) {
    lbv=max(t1,7*t1-2*PS1+2);
    ubv=min(floord(PS2-1,2),7*t1);
#pragma ivdep
#pragma vector always
    for (t2=lbv;t2<=ubv;t2++) {
      if ((t1+t2)%2 == 0) {
        C[t1 + 2*((-t1+t2)/2)] += A[3*t1 + ((-t1+t2)/2)] * B[2*t1 + 4*((-t1+t2)/2)];;
      }
    }
  }
}
/* End of CLooG code */
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %.10f\n", PS1, PS2, elapsed);
    fp = fopen("time.txt", "a+");
    fprintf(fp, " %d %d %.10f\n", PS1, PS2, elapsed);
    fclose(fp);

    return 0;
}

