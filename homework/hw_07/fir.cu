#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void fir_kernel(float* weight, float* buffer, float* result, int firLen, int bufferLen) {
    
  for(int i=0;i<bufferLen;i++){
    int tid = threadIdx.x;  // Thread index within the block
    int globalId = blockIdx.x * blockDim.x + tid;  // Global index
    float sum=0;

    // Each thread computes a part of the sum
    if (globalId < firLen) {
        sum = weight[globalId] * buffer[(globalId) % bufferLen];
    }

    __syncthreads();  // Ensure all threads have written to shared memory

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum += sum;
        }
        __syncthreads();  // Ensure all threads have updated shared memory
    }

    // Write the result from the first thread of the block
    if (tid == 0) {
        atomicAdd(result, sum);  // Use atomic add to avoid race conditions
    }
  }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "ERROR: Program requires 2 input arguments\n");
        return -1;
    }

    char *out;
    long firLen = strtol(argv[1], &out, 10);
    if (argv[1] == out) {
        fprintf(stderr, "ERROR: firLen is not a number\n");
        return -1;
    }

    long niter = strtol(argv[2], &out, 10);
    if (argv[2] == out) {
        fprintf(stderr, "ERROR: niter is not a number\n");
        return -1;
    }

    long bufferLen = 15000 * 10;
    float* buffer = (float*) malloc(sizeof(float) * bufferLen);
    float* weight = (float*) malloc(sizeof(float) * firLen);
    float result = 0.0f;

    for (int i = 0; i < firLen; i++) {
        weight[i] = rand() % 101;  // Custom range from 0 to 100
    }
    for (int i = 0; i < bufferLen; i++) {
        buffer[i] = rand() % 101;  // Custom range from 0 to 100
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);  // Get properties for the first GPU
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int warpSize = deviceProp.warpSize;
    
    // Calculate threads per block, rounding up to a multiple of warpSize and capping at 1024
    int threadsPerBlock = ((firLen + warpSize - 1) / warpSize) * warpSize;
    threadsPerBlock = min(threadsPerBlock, 1024);  // Limit to 1024 threads
    
    // Calculate number of blocks
    int blocksPerGrid = (firLen + threadsPerBlock - 1) / threadsPerBlock;


    // Allocate memory on the GPU
    float* d_weight;
    float* d_buffer;
    float* d_result;
    cudaMalloc(&d_weight, firLen * sizeof(float));
    cudaMalloc(&d_buffer, bufferLen * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(d_weight, weight, firLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer, buffer, bufferLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);

    // Start wall-clock time measurement
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < niter; i++) {
        // Reset result to 0 for each iteration
        cudaMemset(d_result, 0, sizeof(float));

        // Launch the kernel
        fir_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_weight, d_buffer, d_result, firLen, bufferLen);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){

	  printf("CUDA ERROR: %s\n",cudaGetErrorString(err));
	}
	
        cudaDeviceSynchronize();  // Ensure the kernel completes
    }

    cudaEventRecord(stop);

    // Copy the result back to the host
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_weight);
    cudaFree(d_buffer);
    cudaFree(d_result);

    // Calculate and print the elapsed time
    float elapsed_time=0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    
    printf("%f;%ld\n",(elapsed_time / (float)niter) / 1000.0f,firLen);
    // Free CPU memory
    free(buffer);
    free(weight);

    return 0;
}
