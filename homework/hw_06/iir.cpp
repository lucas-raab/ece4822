#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <cmath>


//#define DEBUG true
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "ERROR, p01 requires 3 input arguments\n");
        return -1;
    }

    char *out;
    long feedforwardOrder = strtol(argv[1], &out, 10);
    if (argv[1] == out) {
        fprintf(stderr, "ERROR, feedforwardOrder not a number\n");
        return -1;
    }

    long feedbackOrder = strtol(argv[2], &out, 10);
    if (argv[2] == out) {
        fprintf(stderr, "ERROR, feedbackOrder not a number\n");
        return -1;
    }
    
    long niter = strtol(argv[3], &out, 10);
    if (argv[3] == out) {
        fprintf(stderr, "ERROR, niter not a number\n");
        return -1;
    }

    long bufferLen = 15000*10;
    float* buffer = (float*)malloc(sizeof(float) * bufferLen);
    float* forwardweight = (float*)malloc(sizeof(float) * feedforwardOrder);
    float* behindweight = (float*)malloc(sizeof(float) * feedbackOrder);
    float* output = (float*)malloc(sizeof(float) * bufferLen); // need for feed behind



    // Initialize weights and input signal
    for (int i = 0; i < feedforwardOrder; i++) {
        forwardweight[i] = (float)rand() / RAND_MAX; // Normalize weights
    }
    
    for (int i = 0; i < feedbackOrder; i++) {
        behindweight[i] = (float)rand() / RAND_MAX; // Normalize weights
    }

    for (int i = 0; i < bufferLen; i++) {
        buffer[i] = (float)rand() / RAND_MAX; // Example input signal
    }
    double start, end;
    int local_sum, sum = 0;
    start = omp_get_wtime();

    for (int cur = 0; cur < niter; cur++) {  // rerun experiment for n iterations
#pragma omp parallel private(local_sum) shared(sum)
        {
            local_sum = 0; // Initialize local sum for each thread

            for (int i = 0; i < bufferLen; i++) { //loop over our buffer 
	      
#pragma omp for schedule(static,1)
                for (int j = 0; j < feedforwardOrder; j++) {
		  if (i - j >= 0) {
                        local_sum += forwardweight[j] * buffer[i - j];
                    }
                }

#pragma omp for schedule(static,1)
                for (int j = 1; j < feedbackOrder; j++) { //find our feedbacks
                    if (i - j >= 0) {
                        local_sum -= behindweight[j - 1] * output[i - j]; // Feedback
                    }
                }
		
                output[i] = local_sum; // Store the output
            }
            #pragma omp atomic
            sum += local_sum;
        }
    }
    
    end = omp_get_wtime();
    double elapsed_time = (end - start) / niter;  // Average time per iteration
    printf("%f;%d;%s\n", elapsed_time,feedforwardOrder, getenv("OMP_NUM_THREADS"));
    int error;
    #ifdef DEBUG

    float* expected_output = (float*)malloc(sizeof(float) * bufferLen);



	}
	
    }
    printf("Total error: %f\n", error);
    free(expected_output);
    
    #endif

    


    // Free allocated memory
    free(buffer);
    free(forwardweight);
    free(behindweight);
    free(output);


    return 0;
}
