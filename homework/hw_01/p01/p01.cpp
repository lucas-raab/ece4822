#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "p01.h"
#include <time.h>
//#define DEBUG true

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
     
    float* matrix1=new float[nrows*ncols];
    float* matrix2=new float[ncols*nrows];

    float* matrix3=new float[nrows*nrows];
    time_t start,end;
    float elasped=0;


    genmat(matrix1,nrows,ncols);
    genmat(matrix2,ncols,nrows);
    
    int i= niter;
    start=clock();
    while(0<i--){



      if (!(mmult(matrix3,matrix1,matrix2,nrows,ncols)))
	    return -1;

      #ifdef DEBUG 
     printmat(matrix3,nrows,nrows);
      #endif
     


    }
    end=clock();
   
    
    float elapsed_time = double((end - start)) / CLOCKS_PER_SEC;
    // Print the elapsed time
    
    printf("%f;%d\n", elapsed_time/niter,nrows);

}
