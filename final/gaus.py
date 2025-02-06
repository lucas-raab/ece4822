import cupy as cp
from cupyx.scipy.signal import convolve2d
import numpy as np
import math




# Function to generate a 7x7 Gaussian kernel
def gaussian_kernel(size=7, sigma=1.0):
    kernel = [[0] * size for _ in range(size)]
    sum_val = 0
    for x in range(size):
        for y in range(size):
            # 2D Gaussian function
            exponent = -(x - size//2)**2 - (y - size//2)**2
            kernel[x][y] = math.exp(exponent / (2 * sigma**2))
            sum_val += kernel[x][y]

    # Normalize the kernel so the sum is 1
    kernel = [[val / sum_val for val in row] for row in kernel]
    return np.array(kernel, dtype=np.float32)
