#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

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
    #pragma scop

    for(int i = 0; i < PS1; i++) { 
    	for(int j = 0; j < PS2; j++) {
    		if((2*i+j) < PS1 && 3*i<PS2){
     		C[2*i + j] += A[3*i] * B[j];
     		}
     	}
    }
    #pragma endscop
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %.10f\n", PS1, PS2, elapsed);
    fp = fopen("time.txt", "a+");
    fprintf(fp, " %d %d %.10f\n", PS1, PS2, elapsed);
    fclose(fp);

    return 0;
}

