#include <stdbool.h>
#include <stdio.h>


bool mmult(float mat3[], float mat1[], float mat2[], long nrows, long ncols);
bool genmat(float mat[], long nrows, long ncols); //genearte matrix with random values
bool printmat(float mat[],long nrows, long ncols);

#include <stdio.h>


bool printmat(float mat[], long nrows, long ncols) {
    fprintf(stdout, "[\n");
    for (long i = 0; i < nrows; i++) {
        fprintf(stdout, "  [ ");
        for (long j = 0; j < ncols; j++) {
            fprintf(stdout, "%.2f", mat[i * ncols + j]);
            if (j < ncols - 1) {
                fprintf(stdout, ", ");
            }
        }
        fprintf(stdout, " ]");
        if (i < nrows - 1) {
            fprintf(stdout, ",\n");
        }
    }
    fprintf(stdout, "\n]\n");
    return true;
}



bool mmult(float* mat3, float* mat1, float* mat2, long nrows, long ncols) {

    for (long i = 0; i < nrows; i++) {
        for (long j = 0; j < nrows; j++) {

            mat3[i * nrows + j]  = 0;

            for (long k = 0; k < ncols; k++) {
	            mat3[i * nrows + j] += mat1[i * ncols + k] * mat2[k * nrows + j];



            }

        }
    }

    return true;
}



bool genmat(float mat[], long nrows, long ncols){

  for(int i=0; i<nrows;i++){
    for(int j=0; j<ncols;j++){
      mat[i * ncols +j]=(float)(rand());


    }

  }

  return 1;
}

