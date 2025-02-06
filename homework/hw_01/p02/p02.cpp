#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "p02.h"
#include <time.h>
#include <math.h>

int main(int argc, char** argv){

  if (argc != 4){
    fprintf(stderr,"ERROR, p01 requires 3 input arguments\n");
    return -1;
  }

  char *out;
  long N=strtol(argv[1],&out,10);
  if(argv[1] == out){
    fprintf(stderr, "ERROR, N not a number\n");
    return -1;
      } 

  
  long K=strtol(argv[2],&out,10);

  
    if(argv[2] == out){
    fprintf(stderr, "ERROR, K not a number\n");
    return -1;
    } 
         
  long niter=strtol(argv[3],&out,10);
    if(argv[3] == out){
    fprintf(stderr, "ERROR, niter not a number\n");
    return -1;
      }
    float X[N];
    float R[K];
    printf("%d %d\n",N,K);

    time_t start,end;    
    start=clock();
    gensig(X,N,false);
    while(0<niter--){
      

      autocor(R,X,N,K);
 

    }
    end=clock();

    float elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("Time taken: %f seconds\n", elapsed_time);






}
