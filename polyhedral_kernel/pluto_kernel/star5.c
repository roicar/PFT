#include <stdio.h>
#include "math.h"
#include <stdlib.h>
#include "time.h"
#include <sys/time.h>
#define Smax 1
#define Smin 0
#include <omp.h>
#define alpha 0.6

double **Init_Matrix_rand_PSxM(double **matrix,int PS,int M){
    matrix=(double**)malloc(sizeof(double*)*PS);
    for(int i=0;i<PS;++i){
        matrix[i]=(double*)malloc(sizeof(double)*M);
    }
    for(int i=0;i<PS;++i){
        for(int j=0;j<M;++j){
            //32767 (2^16-1)
            matrix[i][j]=(double)rand()/32767*(Smax-Smin)+Smin;
        }
    }
    return matrix;
}

double **Init_Matrix_zero_PSxM(double **matrix,int PS,int M){
    matrix=(double**)malloc(sizeof(double*)*PS);
    for(int i=0;i<PS;++i){
        matrix[i]=(double*)malloc(sizeof(double)*M);
    }
    for(int i=0;i<PS;++i){
        for(int j=0;j<M;++j){
            //32767 (2^16-1)
            matrix[i][j]=0;
        }
    }
    return matrix;
}

void free_Matrix(double **matrix,int PS){
     for(int i=0;i<PS;++i){
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc,char *argv[]){
    int t,i,j;
    double **a;
    FILE *fp = NULL;
    double Total_time = 0.0;
    srand((unsigned)time(NULL));
    int PS3 =atoi(argv[1]);
    int PS1 =atoi(argv[2]);
    int PS2 =atoi(argv[3]);
    int I = atoi(argv[4]);
    int J = atoi(argv[5]);
    int K = atoi(argv[6]);
    int order = atoi(argv[7]);
    int schorder = atoi(argv[8]);
    a = Init_Matrix_rand_PSxM(a,PS1,PS2);
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
#pragma scop
        for (i=1; i<=PS1-2; i++)  {
            for (j=1; j<=PS2-2; j++)  {
                a[i][j] = (a[i - 1][j] + a[i][j + 1] + a[i][j] + a[i + 1][j] + a[i][j - 1]) / 5.0;
            }
        }
#pragma endscop

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("%d %d %d %d %d %d %d %d %.10f\n",PS1,PS2,PS3,I,J,K,order,schorder,elapsed);
    fp = fopen("seidelOut.txt", "a+");
    fprintf(fp," %d %d %d %d %d %d %d %d %.10f\n",PS1,PS2,PS3,I,J,K,order,schorder,elapsed);
    fclose(fp);
    free_Matrix(a,PS1);
    return 0;
}
