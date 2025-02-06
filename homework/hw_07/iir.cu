#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <cmath>

__global__ void iir_kernel(
    float* bweight,
    int border,
    float* fweight,
    int forder,
    float* buffer,
    float* result,
    int bufferLen
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < bufferLen) {
        float sum = fweight[0] * buffer[idx];

        // Apply feed-forward coefficients
        for (int j = 1; j <= forder && idx - j >= 0; ++j) {
            sum += fweight[j] * buffer[idx - j];
        }

        // Apply feedback coefficients
        for (int j = 1; j <= border && idx - j >= 0; ++j) {
            sum -= bweight[j] * result[idx - j];
        }

        result[idx] = sum / bweight[0];
    }

    __syncthreads();  // Synchronization within the block if necessary (although not required here)
}



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
    float* result = (float*)malloc(sizeof(float) * bufferLen); // need for feed behind



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

    float* d_buffer;
    float *d_fweight;
    float *d_bweight;
    float *d_result;

    cudaMalloc(&d_buffer,bufferLen *sizeof(float));
    cudaMalloc(&d_fweight,feedforwardOrder*sizeof(float));
    cudaMalloc(&d_bweight,feedbackOrder*sizeof(float));
    cudaMalloc(&d_result,bufferLen*sizeof(float));

    cudaMemcpy(d_fweight,forwardweight,feedforwardOrder*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_bweight,behindweight,feedbackOrder*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer,buffer,bufferLen*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_result,result,bufferLen*sizeof(float),cudaMemcpyHostToDevice);

// Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int warpSize = deviceProp.warpSize;

    // Calculate optimal thread and block dimensions
    long minOrder = std::min(feedforwardOrder, feedbackOrder);
    int threadsPerBlock = std::min((long)maxThreadsPerBlock,
                          (((long)minOrder + (long)warpSize - 1) / (long)warpSize) * (long)warpSize);
    int blocksPerGrid = (bufferLen + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    

    // Warm-up run
    iir_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_bweight, feedbackOrder, d_fweight, 
                                                   feedforwardOrder, d_buffer, d_result, bufferLen);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Synchronize before starting timer
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    
    for (int cur = 0; cur < niter; cur++) {  // rerun experiment for n iterations
      iir_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock>>>(d_bweight, feedbackOrder, d_fweight, feedforwardOrder, d_buffer, d_result, bufferLen);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
	printf("CUDA error: %s\n", cudaGetErrorString(err));
      }

    }
    
     cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time=0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("%f;%ld\n",(elapsed_time / (float)niter) / 1000.0f,feedbackOrder);
    
    

    return 0;
}
