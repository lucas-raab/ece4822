#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "p01.h"
#include <cuda_runtime.h>

// CUDA kernel
__global__ void mmult_kernel(float* mat3, float* mat1, float* mat2, long nrows, long ncols, long start_row, long end_row) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < end_row && col < nrows) {
        float sum = 0.0f;
        for (int k = 0; k < ncols; k++) {
            sum += mat1[(row - start_row) * ncols + k] * mat2[k * nrows + col];
        }
        mat3[(row - start_row) * nrows + col] = sum;
    }
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

    mat1 = (float*)malloc(sizeof(float) * nrows * ncols);
    mat2 = (float*)malloc(sizeof(float) * nrows * ncols);
    mat3 = (float*)malloc(sizeof(float) * nrows * nrows);

    // Initialize host arrays
    genmat(mat1, nrows, ncols);
    genmat(mat2, ncols, nrows);

    // Get the number of available GPUs
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("Number of GPUs: %d\n", num_gpus);

    // Allocate memory for device pointers and CUDA streams
    float **d_mat1 = (float**)malloc(num_gpus * sizeof(float*));
    float **d_mat2 = (float**)malloc(num_gpus * sizeof(float*));
    float **d_mat3 = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    if (d_mat1 == NULL || d_mat2 == NULL || d_mat3 == NULL || streams == NULL) {
        fprintf(stderr, "Failed to allocate host memory for device pointers or streams\n");
        return -1;
    }

    // Calculate the number of rows per GPU
    long rows_per_gpu = nrows / num_gpus;
    long remainder = nrows % num_gpus;

    for (int i = 0; i < num_gpus; i++) {
        printf("gpu : %d\n", i);
        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed for GPU %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        long start_row = i * rows_per_gpu + (i < remainder ? i : remainder);
        long end_row = start_row + rows_per_gpu + (i < remainder ? 1 : 0);
        long gpu_nrows = end_row - start_row;

        err = cudaMalloc((void**)&d_mat1[i], sizeof(float) * gpu_nrows * ncols);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_mat1[%d]: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        err = cudaMalloc((void**)&d_mat2[i], sizeof(float) * nrows * ncols);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_mat2[%d]: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        err = cudaMalloc((void**)&d_mat3[i], sizeof(float) * gpu_nrows * nrows);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_mat3[%d]: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaStreamCreate failed for GPU %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        err = cudaMemcpyAsync(d_mat1[i], mat1 + start_row * ncols, sizeof(float) * gpu_nrows * ncols, cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for d_mat1[%d]: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        err = cudaMemcpyAsync(d_mat2[i], mat2, sizeof(float) * nrows * ncols, cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync failed for d_mat2[%d]: %s\n", i, cudaGetErrorString(err));
            continue;
        }

        // Get the maximum number of threads per block for this GPU
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for GPU %d: %s\n", i, cudaGetErrorString(err));
            continue;
        }
        int max_threads_per_block = prop.maxThreadsPerBlock;

        // Calculate optimal block and grid sizes
        int block_size = 16;
        dim3 blockSize(block_size, block_size);
        dim3 gridSize((nrows + blockSize.x - 1) / blockSize.x,
                      (gpu_nrows + blockSize.y - 1) / blockSize.y);

        printf("GPU %d: Start Row: %ld, End Row: %ld, Block Size: %d\n", i, start_row, end_row, block_size);
    }


    clock_t start, end;
    double elapsed_time = 0;

    int iter = niter;
    start = clock();

    printf("starting\n");
    for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            
            long start_row = i * rows_per_gpu + (i < remainder ? i : remainder);
            long end_row = start_row + rows_per_gpu + (i < remainder ? 1 : 0);
            long gpu_nrows = end_row - start_row;

            // Get the maximum number of threads per block for this GPU
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            int max_threads_per_block = prop.maxThreadsPerBlock;

            // Calculate optimal block and grid sizes
            int block_size = (int)sqrt(max_threads_per_block);
            dim3 blockSize(block_size, block_size);
            dim3 gridSize((nrows + blockSize.x - 1) / blockSize.x,
                          (gpu_nrows + blockSize.y - 1) / blockSize.y);


    
    while (0 < iter--) {


            mmult_kernel<<<gridSize, blockSize, 0, streams[i]>>>(d_mat3[i], d_mat1[i], d_mat2[i], nrows, ncols, start_row, end_row);
        }

        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        #ifdef DEBUG
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            long start_row = i * rows_per_gpu + (i < remainder ? i : remainder);
            long end_row = start_row + rows_per_gpu + (i < remainder ? 1 : 0);
            long gpu_nrows = end_row - start_row;
            cudaMemcpyAsync(mat3 + start_row * nrows, d_mat3[i], sizeof(float) * gpu_nrows * nrows, cudaMemcpyDeviceToHost, streams[i]);
        }
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }
        printmat(mat3, nrows, nrows);
        #endif
    }
    end = clock();

    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("%f;%d\n", elapsed_time / niter, nrows);



    return 0;
}
