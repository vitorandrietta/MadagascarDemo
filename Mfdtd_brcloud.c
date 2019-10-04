/*simple fdtd*/

#ifndef FDTD_2D_H
# define FDTD_2D_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NX) && ! defined(NY) && !defined(TMAX)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define TMAX 2
#   define NX 32
#   define NY 32
#  endif

#  ifdef SMALL_DATASET
#   define TMAX 10
#   define NX 500
#   define NY 500
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define TMAX 65
#   define NX 1000
#   define NY 1000
#  endif

#  ifdef LARGE_DATASET
#   define TMAX 50
#   define NX 2000
#   define NY 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TMAX 100
#   define NX 4000
#   define NY 4000
#  endif
# endif /* !N */


# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2f "
# endif


#endif /* !FDTD_2D */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <rsf.h>

/*simple fdtd*/


#include <omp.h>
/* Include benchmark-specific header. */
/* Default data type is float, default size is 50x1000x1000. */


#define NUM_THREADS 30

/* Array initialization. */
static
void init_array(int nx,
                int ny,
                float **ex,
                float **ey,
                float **hz,
                float *_fict_) {
    int i, j;
    for (i = 0; i < ny; i++)
        _fict_[i] = (float) i;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            ex[i][j] = ((float) i * (j + 1)) / nx;
            ey[i][j] = ((float) i * (j + 2)) / ny;
            hz[i][j] = ((float) i * (j + 3)) / nx;
        }
}



//}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
                 int ny,
                 float **ex,
                 float **ey,
                 float **hz) {
    int i, j;

    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            printf(DATA_PRINTF_MODIFIER, ex[i][j]);
            printf(DATA_PRINTF_MODIFIER, ey[i][j]);
            printf(DATA_PRINTF_MODIFIER, hz[i][j]);
            if ((i * nx + j) % 20 == 0)
                printf("\n");
        }
    printf("\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
                    int nx,
                    int ny,
                    float **ex,
                    float **ey,
                    float **hz,
                    float *_fict_) {
    int t, i, j;

#pragma scop
//you need to  parallelize  the  following  code:
    for (t = 0; t < tmax; t++) {

        for (j = 0; j < ny; j++)
            ey[0][j] = _fict_[t];

        for (i = 1; i < nx; i++)
            for (j = 0; j < ny; j++)
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i - 1][j]);


        for (i = 0; i < nx; i++)
            for (j = 1; j < ny; j++)
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j - 1]);


        for (i = 0; i < nx - 1; i++)
            for (j = 0; j < ny - 1; j++)
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
    };


// -----------------------------------------------------------------
#pragma endscop
}

float **allocateMatrix() {
    float **mat = malloc(sizeof(float *) * NX);

    for (int i = 0; i < NX; ++i) {
        mat[i] = malloc(sizeof(float) * NY);
    }

    return mat;
}

void freeMatrix(float **mat) {
    for (int i = 0; i < NX; ++i) {
        free(mat[i]);

    }

    free(mat);
}

int main(int argc, char **argv) {
    /* Retrieve problem size. */
    sf_file in = NULL, out = NULL;
    sf_init(argc, argv);
    /* standard input */

    /* standard output */
    out = sf_output("out");
    int tmax = TMAX;
    int nx = NX;
    int ny = NY;


    /* Variable declaration/allocation. */
    float **ex = allocateMatrix();
    float **ey = allocateMatrix();
    float **hz = allocateMatrix();
    float *_fict_ = malloc(sizeof(float) * NY);


    /* Initialize array(s). */
    init_array(nx, ny, ex, ey, hz, _fict_);

    float e, s;
    //use this to measure the obtained speedup
//    s = omp_get_wtime();
//    /* Run kernel. */
    kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_);
//    e = omp_get_wtime();

    /*print resulting matrices */


    for (int i = 0; i < NX; ++i) {
        sf_floatwrite(ex[i], NX, out);
        sf_floatwrite(ey[i], NY, out);
        sf_floatwrite(hz[i], NY, out);
    }

    sf_close();
    freeMatrix(ex);
    freeMatrix(ey);
    freeMatrix(hz);
    free(_fict_);
    exit(0);

}
