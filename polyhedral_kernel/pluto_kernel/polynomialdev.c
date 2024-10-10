#include <stdio.h>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#define Smax 1
#define Smin 0

void printPolynomial(double *poly, int degree) {
    for (int i = 0; i <= degree; i++) {
        printf("%.2f*x^%d ", poly[i], i);
        if (i < degree) {
            printf("+ ");
        }
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));

    // Parse degree of polynomials from command line arguments
    int degreeA = atoi(argv[1]);
    int degreeB = atoi(argv[2]);
    FILE *fp = NULL;
    // Allocate memory for coefficients of polynomials
    double *a = (double *)malloc(sizeof(double) * (degreeA + 1));
    double *b = (double *)malloc(sizeof(double) * (degreeB + 1));

    // Initialize polynomials A and B with random coefficients
    for (int i = 0; i <= degreeA; i++) {
        a[i] = (double)rand() / 32767 * (Smax - Smin) + Smin;
    }

    for (int i = 0; i <= degreeB; i++) {
        b[i] = (double)rand() / 32767 * (Smax - Smin) + Smin;
    }

    // Print polynomials A and B
    printf("Polynomial A:\n");
    printPolynomial(a, degreeA);

    printf("Polynomial B:\n");
    printPolynomial(b, degreeB);

    // Compute the degree of the result polynomial
    int resultDegree = degreeA + degreeB;

    // Allocate memory for coefficients of the result polynomial
    double *result = (double *)malloc(sizeof(double) * (resultDegree + 1));
    // Measure execution time
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
#pragma scop
    // Initialize result polynomial with zeros
    for (int i = 0; i <= resultDegree; i++) {
        result[i] = 0.0;
    }
    // Perform polynomial multiplication
    for (int i = 0; i <= degreeA; i++) {
        for (int j = 0; j <= degreeB; j++) {
            result[i + j] += a[i] * b[j];
        }
    }
#pragma endscop
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // Print the result polynomial
    printf("Result of A * B:\n");
    printPolynomial(result, resultDegree);
    printf(" %d %d %.10f\n",degreeA,degreeB,elapsed);
    fp = fopen("polynomial.txt", "a+");
    fprintf(fp," %d %d %.10f\n",degreeA,degreeB,elapsed);
    fclose(fp);

    // Free allocated memory
    free(a);
    free(b);
    free(result);

    return 0;
}

