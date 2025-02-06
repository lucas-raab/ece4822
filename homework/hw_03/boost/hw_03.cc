#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <time.h>

namespace ublas = boost::numeric::ublas;

bool genmat(float*** mat, long nrows, long ncols) {
    if (mat == NULL) return false;
    *mat = (float**)malloc(nrows * sizeof(float*));
    for (int i = 0; i < nrows; i++) {
        (*mat)[i] = (float*)malloc(ncols * sizeof(float));
        for (int j = 0; j < ncols; j++) {
            (*mat)[i][j] = (float)rand() / (float)(RAND_MAX / 5);
        }
    }
    return true;
}

// Convert dynamically allocated 2D array to Boost matrix
ublas::matrix<float> convert_to_boost_matrix(float** mat, long nrows, long ncols) {
    ublas::matrix<float> boost_mat(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            boost_mat(i, j) = mat[i][j];
        }
    }
    return boost_mat;
}

int main(int argc, char** argv) {
    srand(time(NULL));
   if (argc != 4){
     fprintf(stderr,"ERROR, p01 requires 3 input arguments\n");
     return -1;
   }
   char *out;

   long nrows=strtol(argv[1],&out,10);
   if(argv[1] == out){
     fprintf(stderr, "ERROR, nrows not a number\n");
     return -1;
       }

   long ncols=strtol(argv[2],&out,10);
     if(argv[2] == out){
     fprintf(stderr, "ERROR, ncols not a number\n");
     return -1;
     }
   long niter=strtol(argv[3],&out,10);
     if(argv[3] == out){
     fprintf(stderr, "ERROR, niter not a number\n");
     return -1;
     }


     float **mat1, **mat2, **mat3;
    genmat(&mat1, nrows, ncols);
    genmat(&mat2, ncols, nrows);
    genmat(&mat3, nrows, nrows);

    // Convert to Boost matrices
    ublas::matrix<float> boost_mat1 = convert_to_boost_matrix(mat1, nrows, ncols);
    ublas::matrix<float> boost_mat2 = convert_to_boost_matrix(mat2, ncols, nrows);
    ublas::matrix<float> boost_mat3 = convert_to_boost_matrix(mat3, ncols, nrows);

    // Perform matrix multiplication

    time_t start,end;
    int index=niter;
    start=clock();
    while(0<index--){
      boost::numeric::ublas::axpy_prod(boost_mat1, boost_mat2, boost_mat3, true);  // C = A * B
    }
    end=clock();

    double finTime=(double)(end-start)/CLOCKS_PER_SEC;
    printf("%f;%ld\n",(finTime/niter),nrows);
   

    // Free dynamically allocated memory
    for (int i = 0; i < nrows; i++) free(mat1[i]);
    free(mat1);
    for (int i = 0; i < ncols; i++) free(mat2[i]);
    free(mat2);

    return 0;
}

