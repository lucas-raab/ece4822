#include <iostream>
#include <stdbool.h>
#include <time.h>
#include <cmath>

  bool genmat(float mat[], long nrows, long ncols){
    
    for (long i = 0; i < nrows; ++i) {
        for (long j = 0; j < ncols; ++j) {
            mat[i * ncols + j] = (float)(rand());
        }
    }

    return true;
}

bool print_matrix(float mat[], long nrows, long ncols) {
  fprintf(stdout, "[\n");
  for (long i = 0; i < nrows; ++i) {
    fprintf(stdout, "  [ ");
    for (long j = 0; j < ncols; ++j) {
      fprintf(stdout, "%.2f", mat[i * ncols + j]);
      if (j < ncols - 1){
	fprintf(stdout, ", ");
      }
    }
    fprintf(stdout, " ]");
    printf("\n");
    if (i < nrows - 1) {
      fprintf(stdout, ",\n");
    }
  }
  fprintf(stdout, "\n]\n");
  return true;
}


long twoPower(long n) {
    return std::pow(2, std::ceil(std::log2(n)));
}


void padmat(float mat[], float mat1[], long size, long size1) {
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            mat1[i * size1 + j] = mat[i * size + j];
        }
    }
    // Fill extra rows and columns with zeros.
    for (long i = size ; i < size1; ++i) {
        for (long j = 0; j < size1; ++j) {
            mat1[i * size1 + j] = 0.0;
        }
    }
    for (long i = 0; i < size1; ++i) {
        for (long j = size; j < size1; ++j) {
	  mat1[i * size1 + j] = 0.0;
        }
    }
}
//function for naive mmult
void mmult(long size,long xdist,float mat1[],long ydist,float mat2[],long depth, float mat3[]) {
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (long k = 0; k < size; ++k) {
                sum += mat1[i * xdist + k] * mat2[k * ydist + j];
            }
            mat3[i * depth + j] = sum;
        }
    }
}


// function for matrix addition
void madd(long size, long xdist_A, float mat1[], long xdist_B, float mat2[], long xdist_result, float mat3[]) {
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            mat3[i * xdist_result + j] = mat1[i * xdist_A + j] + mat2[i * xdist_B + j];
        }
    }
}

// function for matrix subtraction
void msub(long size, long xdist_A, float mat1[], long xdist_B, float mat2[], long xdist_result, float mat3[]) {
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            mat3[i * xdist_result + j] = mat1[i * xdist_A + j] - mat2[i * xdist_B + j];
        }
    }
}



void strassen_mmult(long size, long xdist_X, float mat1[], long ydist_Y, float mat2[], long depth_Z, float mat3[]) {

  if (size <= 32) {
        mmult(size, xdist_X, mat1, ydist_Y, mat2, depth_Z, mat3);
        return;
    }

    long half_size = size / 2;
    float *A = mat1;
    float *B = mat1 + half_size;
    float *C = mat1 + half_size * xdist_X;
    float *D = C + half_size;

    float *E = mat2;
    float *F = mat2 + half_size;
    float *G = mat2 + half_size * ydist_Y;
    float *H = G + half_size;

    float *P[7];
    long matrix_size = half_size * half_size * sizeof(float);

    for (long i = 0; i < 7; i++) {
        P[i] = (float *)malloc(matrix_size);
    }

    float *temp1 = (float *)malloc(matrix_size);
    float *temp2 = (float *)malloc(matrix_size);

    msub(half_size, ydist_Y, F, ydist_Y, H, half_size, temp1);
    strassen_mmult(half_size, xdist_X, A, half_size, temp1, half_size, P[0]);

    madd(half_size, xdist_X, A, xdist_X, B, half_size, temp1);
    strassen_mmult(half_size, half_size, temp1, ydist_Y, H, half_size, P[1]);

    madd(half_size, xdist_X, C, xdist_X, D, half_size, temp1);
    strassen_mmult(half_size, half_size, temp1, ydist_Y, E, half_size, P[2]);

    msub(half_size, ydist_Y, G, ydist_Y, E, half_size, temp1);
    strassen_mmult(half_size, xdist_X, D, half_size, temp1, half_size, P[3]);

    madd(half_size, xdist_X, A, xdist_X, D, half_size, temp1);
    madd(half_size, ydist_Y, E, ydist_Y, H, half_size, temp2);
    strassen_mmult(half_size, half_size, temp1, half_size, temp2, half_size, P[4]);

    msub(half_size, xdist_X, B, xdist_X, D, half_size, temp1);
    madd(half_size, ydist_Y, G, ydist_Y, H, half_size, temp2);
    strassen_mmult(half_size, half_size, temp1, half_size, temp2, half_size, P[5]);

    msub(half_size, xdist_X, A, xdist_X, C, half_size, temp1);
    madd(half_size, ydist_Y, E, ydist_Y, F, half_size, temp2);
    strassen_mmult(half_size, half_size, temp1, half_size, temp2, half_size, P[6]);

    madd(half_size, half_size, P[4], half_size, P[3], half_size, temp1);
    msub(half_size, half_size, P[5], half_size, P[1], half_size, temp2);
    madd(half_size, half_size, temp1, half_size, temp2, depth_Z, mat3);

    madd(half_size, half_size, P[2], half_size, P[3], depth_Z, mat3 + half_size * depth_Z);

    madd(half_size, half_size, P[0], half_size, P[1], depth_Z, mat3 + half_size);

    madd(half_size, half_size, P[0], half_size, P[4], half_size, temp1);
    madd(half_size, half_size, P[2], half_size, P[6], half_size, temp2);
    msub(half_size, half_size, temp1, half_size, temp2, depth_Z, mat3 + half_size * (depth_Z + 1));


}



int main(const int argc, const char **argv) {
  
  if (argc != 4){
    fprintf(stderr,"ERROR, p01 requires 3 input arguments\n");
    return -1;
  }
  char *out;
  long matrix_size=strtol(argv[1],&out,10);
  if(argv[1] == out){
    fprintf(stderr, "ERROR, matrixsize not a number\n");
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
    
    
    long padded_size = twoPower(matrix_size);
    
    float *matrix_A = (float *)malloc(padded_size * padded_size * sizeof(float));
    float *matrix_B = (float *)malloc(padded_size * padded_size * sizeof(float));
    float *matrix_C = (float *)malloc(padded_size * padded_size * sizeof(float));

    float *original_A = (float *)malloc(matrix_size * matrix_size * sizeof(float));
    float *original_B = (float *)malloc(matrix_size * matrix_size * sizeof(float));


    genmat(original_A, matrix_size, matrix_size);
    genmat(original_B, matrix_size, matrix_size);

    padmat(original_A, matrix_A, matrix_size, padded_size);
    padmat(original_B, matrix_B, matrix_size, padded_size);


    time_t start,end;

     start=clock();

     int i=niter;
     while(i-->0){
        strassen_multiply(padded_size, padded_size, matrix_A, padded_size, matrix_B, padded_size, matrix_C);
    }
     end=clock();
     
     float elapsed_time = double((end - start)) / CLOCKS_PER_SEC;
    // Print the elapsed time
    
    printf("%f;%d\n", elapsed_time/niter,matrix_size);

    return 0;
}
