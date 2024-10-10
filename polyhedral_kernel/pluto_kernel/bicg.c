#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

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
void Init_array_rand_PSxM(float matrix[1024], int PS1) {
    for (int i = 0; i < PS1; i++) {
        matrix[i] = (float)rand() / 32767 * (Smax - Smin) + Smin;
    }
}

void Init_Matrix_zero_PSxM(float matrix[1024][1024], int PS1, int PS2) {
    for (int i = 0; i < PS1; i++) {
        for (int j = 0; j < PS2; j++) {
            matrix[i][j] = 0;
        }
    }
}

void Init_array_zero_PSxM(float matrix[1024], int PS1) {
    for (int i = 0; i < PS1; i++) {
        matrix[i] = 0;
    }
}



int main(){
    float A[1024][1024], s[1024],q[1024],p[1024],r[1024];
    double Total_time = 0.0;
    srand((unsigned)time(NULL));
    
    Init_Matrix_rand_PSxM(A,1024,1024);
    Init_array_zero_PSxM(s, 1024);
    Init_array_rand_PSxM(p, 1024);
    Init_array_rand_PSxM(r, 1024);
    Init_array_zero_PSxM(q, 1024);
    
    struct timespec start, finish;
    double elapsed;
    float milliseconds = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);


#pragma scop
	
  for (int i = 0; i < 1024; i++)
    s[i] = 0;
  for (int i = 0; i < 1024; i++)
    {
      q[i] = 0;
      for (int j = 0; j < 1024; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }
#pragma endscop


    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%.10f\n",elapsed);
    return 0;
}

