#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "p01.h"
#include <time.h>
//#define DEBUG true

#include <omp.h>


int main(int argc, char** argv){

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
 
#ifdef DEBUG
  printf("nrows= %d\n", nrows);
#endif
  
  long ncols=strtol(argv[2],&out,10);
    if(argv[2] == out){
    fprintf(stderr, "ERROR, ncols not a number\n");
    return -1;
    }
    
#ifdef DEBUG
    printf("ncols= %d\n", ncols);
#endif
    
  long niter=strtol(argv[3],&out,10);
    if(argv[3] == out){
    fprintf(stderr, "ERROR, niter not a number\n");
    return -1;
      }

#ifdef DEFINE  
      printf("niter= %d\n", niter);
#endif
     
      float* mat1=new float[nrows*ncols];
    float* mat2=new float[ncols*nrows];

    float* mat3=new float[nrows*nrows];
    double start,end;
    float elasped=0;


    genmat(mat1,nrows,ncols);
    genmat(mat2,ncols,nrows);
    
    int i= niter;
    start = omp_get_wtime();
    while(0<i--){
#pragma omp parallel 


      #pragma omp for schedule(static,1)
      for (long i = 0; i < nrows; i++) {
	for (long j = 0; j < nrows; j++) {
	  
	  mat3[i * nrows + j]  = 0;
	  
	  for (long k = 0; k < ncols; k++) {
	    mat3[i * nrows + j] += mat1[i * ncols + k] * mat2[k * nrows + j];
	    
	    
	    
	  }

	}
      }
      


      #ifdef DEBUG 
     printmat(matrix3,nrows,nrows);
      #endif
     


    }
     end = omp_get_wtime();
   
    
    float elapsed_time = double((end - start));
    // Print the elapsed time
    
    printf("%f;%d\n", elapsed_time/niter,nrows, getenv("OMP_NUM_THREADS"));

}
