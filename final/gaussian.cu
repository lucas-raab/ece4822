__global__ void gaussianFilterKernel(
    float* input,     // Input image array
    float* output,    // Output filtered image array
    float* kernel,    // Gaussian kernel
    int width,        // Image width
    int height,       // Image height
    int kernelSize    // Size of the Gaussian kernel
) {
    // Calculate 2D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within image bounds
    if (x < width && y < height) {
        float sum = 0.0f;
        int kernelRadius = kernelSize / 2;

        // Convolution operation
        for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
            for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                // Calculate source pixel coordinates
                int sourceX = x + kx;
                int sourceY = y + ky;

                // Handle boundary conditions (zero padding)
                if (sourceX >= 0 && sourceX < width && 
                    sourceY >= 0 && sourceY < height) {
                    int kernelIndex = (ky + kernelRadius) * kernelSize + 
                                      (kx + kernelRadius);
                    int sourceIndex = sourceY * width + sourceX;
                    
                    sum += input[sourceIndex] * kernel[kernelIndex];
                }
            }
        }

        // Store result
        output[y * width + x] = sum;
    }
}

// Host function to set up and launch the kernel
void applyGaussianFilter(
    float* d_input,   // Device input image
    float* d_output,  // Device output image
    float* d_kernel,  // Device Gaussian kernel
    int width,        // Image width
    int height,       // Image height
    int kernelSize,   // Size of the Gaussian kernel
    int channels      // Number of color channels
) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x, 
        (height + blockDim.y - 1) / blockDim.y
    );

    // Process each channel separately
    for (int c = 0; c < channels; c++) {
        // Calculate offsets for current channel
        float* channelInput = d_input + (c * width * height);
        float* channelOutput = d_output + (c * width * height);

        // Launch kernel for this channel
        gaussianFilterKernel<<<gridDim, blockDim>>>(
            channelInput,
            channelOutput,
            d_kernel,
            width,
            height,
            kernelSize
        );
    }


}
