#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>



// CUDA kernel
__global__ void mmult_kernel(float* mat3, float* mat1, float* mat2, long nrows, long ncols) {
      for (long i = 0; i < nrows; i++) {
          for (long j = 0; j < nrows; j++) {
 
              mat3[i * nrows + j]  = 0;
 
              for (long k = 0; k < ncols; k++) {
                      mat3[i * nrows + j] += mat1[i * ncols + k] * mat2[k * nrows + j];
 
 
 
              }
 
          }
      }
 
}

bool genmat(float mat[], long nrows, long ncols){
   for(int i=0; i<nrows;i++){
     for(int j=0; j<ncols;j++){
       mat[i * ncols +j]=(float)(rand());
     }
   }
   return 1;
 }



int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "ERROR, p01 requires 3 input arguments\n");
        return -1;
    }

    char *out;
    long nrows = strtol(argv[1], &out, 10);
    if (argv[1] == out) {
        fprintf(stderr, "ERROR, nrows not a number\n");
        return -1;
    }
  
    long ncols = strtol(argv[2], &out, 10);
    if (argv[2] == out) {
        fprintf(stderr, "ERROR, ncols not a number\n");
        return -1;
    }
    
    long niter = strtol(argv[3], &out, 10);
    if (argv[3] == out) {
        fprintf(stderr, "ERROR, niter not a number\n");
        return -1;
    }

    float *mat1, *mat2, *mat3;
    float *cmat1, *cmat2, *cmat3;

    mat1 = (float*)malloc(sizeof(float) * nrows * ncols);
    mat2 = (float*)malloc(sizeof(float) * nrows * ncols);
    mat3 = (float*)malloc(sizeof(float) * nrows * nrows);

    // Initialize host arrays
    genmat(mat1, nrows, ncols);
    genmat(mat2, ncols, nrows);
      
    cudaMalloc((void**)&cmat1, sizeof(float) * nrows * ncols);
    cudaMalloc((void**)&cmat2, sizeof(float) * nrows * ncols);
    cudaMalloc((void**)&cmat3, sizeof(float) * nrows * nrows);
      
    cudaMemcpy(cmat1, mat1, sizeof(float) * nrows * ncols, cudaMemcpyHostToDevice);
    cudaMemcpy(cmat2, mat2, sizeof(float) * nrows * ncols, cudaMemcpyHostToDevice);

 

    clock_t start, end;
    double elapsed_time = 0;

    int i = niter;
    start = clock();
    
    cudaDeviceSynchronize();
    while (0 < i--) {
      mmult_kernel<<<1,1>>>(cmat3, cmat1, cmat2, nrows, ncols);

      cudaDeviceSynchronize();
     
    }
    end = clock();

#ifdef DEBUG
    cudaMemcpy(mat3, cmat3, sizeof(float) * nrows * nrows, cudaMemcpyDeviceToHost);
    printmat(mat3, nrows, nrows);
#endif
    
    
    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Print the elapsed time
    printf("%f;%d\n", elapsed_time / niter, nrows);

    // Clean up
    free(mat1);
    free(mat2);
    free(mat3);
    cudaFree(cmat1);
    cudaFree(cmat2);
    cudaFree(cmat3);

    return 0;
}
