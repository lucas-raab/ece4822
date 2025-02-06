#include <stdio.h>
#include <stdlib.h>
#include <iostream>




// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RAND_MAX 100
int main(int argc, char** argv){

  if (argc != 3){
    fprintf(stderr,"ERROR, p01 requires 2 input arguments\n");
    return -1;
  }

  char *out;
  long firLen=strtol(argv[1],&out,10);
  if(argv[1] == out){
    fprintf(stderr, "ERROR, firlen not a number\n");
    return -1;
  }

   
  long niter=strtol(argv[2],&out,10);
  if(argv[2] == out){
    fprintf(stderr, "ERROR, iter not a number\n");
    return -1;
  }
  
  long bufferLen= 15000*10;
  float* buffer = (float*) malloc(sizeof(float) * bufferLen);

  float* weight = (float*) malloc(sizeof(float) * firLen);
  for( int i=0;i< firLen;i++){
    weight[i] = (rand() % (RAND_MAX + 1)) % 101;
  }
  for(int i=0; i < bufferLen; i++){
    buffer[i]= (rand() % (RAND_MAX + 1)) % 101;
  }

  double start, end;
  int local_sum,sum = 0;
  // Start wall-clock time measurement
  start = omp_get_wtime();
  
  for (int cur = 0; cur < niter; cur++) {  // rerun experiment for n iterations
#pragma omp parallel private(local_sum) shared(sum)
    {
      local_sum=0;
       // Print the number of threads being used
      for(int j=0;j<bufferLen;j++){

	// Parallel for loop
#pragma omp for schedule(static,1)
	for (int i = 0; i < firLen; i++) {
	  if(i<j){
	  local_sum += weight[i] * buffer[(j-i) % bufferLen];
	  }
	 
      }


				 
      
      
#pragma omp atomic
sum += local_sum;


      }
  }
  }
  // End wall-clock time measurement
  end = omp_get_wtime();
  
  // Calculate and print the elapsed time
  double elapsed_time = (end - start) / niter;  // Average time per iteration
  printf("%f;%d;%s\n", elapsed_time,firLen, getenv("OMP_NUM_THREADS"));
  
  return 0;
}
