import warnings

# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
import cupy as cp
import os
import ctypes
import time
import sys
import math
import gc 
# import image and annotation tools

import nedc_image_tools
import nedc_dpath_ann_tools
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"




import gaus
from cupyx.scipy.signal import convolve2d
from tqdm import tqdm
import nvtx




memDebug=False
global_min = 0
global_max = 5*10**6
props = cp.cuda.runtime.getDeviceProperties(0)
total_mem=props['totalGlobalMem'] 
gpu_id=0
frame_size = 128
window_size = 256

gaussian_filter_kernel = cp.RawKernel(r'''
extern "C" __global__
void gaussian_filter(const unsigned char* input, // 8 bit Int input  
                     float* output, // Output
                     const float* kernel, // Input kernel
                     int width,     // Width of a single image
                     int height,    // Height of a single image
                     int kernel_size, // Size of the kernel
                     int batch_size) // Number of batches
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;  // Batch index

    if (x >= width || y >= height || batch_idx >= batch_size) return;

    float sum = 0.0f;
    int kernel_radius = kernel_size / 2;

    // Calculate starting offset for the batch
    int batch_offset = batch_idx * width * height;

    // Apply the kernel on the input image
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
            int ix = min(max(x + kx, 0), width - 1);  // Clamping the boundary
            int iy = min(max(y + ky, 0), height - 1); // Clamping the boundary

            float pixel_value = (float)input[batch_offset + iy * width + ix];
            float kernel_value = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
            sum += pixel_value * kernel_value;
        }
    }

    // Write the output pixel value to the output image
    output[batch_offset + y * width + x] = sum;
}
''', 'gaussian_filter')
import cupy as cp

histogram_kernel = cp.RawKernel(r'''
extern "C" __global__ 
void histogram(
    float* fft_result_gpu,  // Input: Flattened FFT result array
    int* histogram_gpu,     // Output: Histogram array (bins)
    int bin_count,          // Number of bins for the histogram
    float global_min,       // Global minimum value for scaling
    float global_max,       // Global maximum value for scaling
    int batch_size          // Number of elements in the batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float value = fft_result_gpu[idx];
        float normalized_value = (value ) / (5000000 );


        int bin_index = int(normalized_value * 16);
        bin_index = min(max(bin_index, 0), 15);

        atomicAdd(&histogram_gpu[bin_index], 1);
    }
}
''', 'histogram')



def acquire_lock(lock_file):

    initPrint=True;

    while True:
        try:
            
            # Attempt to create the lock file exclusively
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as lock:
                lock.write(str(os.getpid()))  
            #print("Starting on gpu lock: ",gpu_id," Process id :", os.getpid())
            return  # Lock acquired
        except FileExistsError:
            if initPrint:
                #print("Waiting on gpu lock: ",gpu_id," Process id :", os.getpid())
                initPrint=False

            # Lock file exists, wait and retry
            #print("Waiting on lock :D", os.getpid())
            
            time.sleep(.05)

def release_lock(lock_file):
    try:
        os.remove(lock_file)
    except FileNotFoundError:
        pass  # Lock file already removed

def generateTopLeftFrameCoordinates(height:int, width:int, frame_size:tuple, quadrant:int=0)->list:
    # Calculate quadrant bounds
    mid_width = width // 2
    mid_height = height // 2
    
    # Set x and y ranges based on quadrant
    if quadrant in [0, 1]:  # top quadrants
        y_start, y_end = 0, mid_height
    else:  # bottom quadrants
        y_start, y_end = mid_height, height
        
    if quadrant in [0, 2]:  # left quadrants
        x_start, x_end = 0, mid_width
    else:  # right quadrants
        x_start, x_end = mid_width, width
    
    # instantiate a return list
    return_list = []

    # for every column in the quadrant
    for x in range(x_start, x_end, frame_size[0]):
        # for every row in the quadrant
        for y in range(y_start, y_end, frame_size[1]):
            # append (column, row) index 
            return_list.append((x,y))

    return return_list
def windowRGBValues(image_file:str, frame_size, window_size:tuple, quadrant:int=0):
    if quadrant not in [0, 1, 2, 3]:
        raise ValueError("Quadrant must be 0, 1, 2, or 3")
    
    # instantiate the image reader class
    image_reader = nedc_image_tools.Nil()
    
    # initialize the class using the image file
    image_reader.open(image_file)

    # get the width and height of the image
    width, height = image_reader.get_dimension()
    
    # get the top left coordinates of each frame for the specified quadrant
    frame_top_left_coordinates = generateTopLeftFrameCoordinates(height, width,
                                                               frame_size,
                                                               quadrant)
    
    # read all of the windows into memory
    windows = image_reader.read_data_multithread(frame_top_left_coordinates,
                                               npixy = window_size[1],
                                               npixx = window_size[0],
                                               color_mode="RGB")
    
    return windows

def generate_synthetic_data(list_length):
    """
    Generate a list of 3D numpy arrays of size [255][255][3].

    Parameters:
        list_length (int): Number of arrays in the list.

    Returns:
        list: A list of 3D numpy arrays.
    """
    return [np.random.randint(0, 256, size=(255, 255, 3), dtype=np.uint8) for _ in range(list_length)]

def cuda_init():
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"CuPy version: {cp.__version__}")
        print("\nDevice properties:")
    
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Device count: {device_count}")
    
        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\nDevice {i}:")
            print(f"  Name: {props['name'].decode()}")
            total_mem=props['totalGlobalMem'] / (1024**3)
            print(f"  Total Memory: {props['totalGlobalMem'] / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Error getting device properties: {e}. Some sort of problem with cuda, so we just aborting lol get fucked ")
        return False
    return True

def process_on_gpu(data, kernel, device_id, global_min=0, global_max=1):
  
    # Ensure kernel is a numpy array
    kernel = np.array(kernel)
    kernel_size = len(kernel)
    batch_size = len(data) 
    height, width, channels = data[0].shape

    # Histogram configuration
    bin_count = 16
    bin_edges = cp.linspace(0, global_max, 17)  # 17 edges for 16 bins
    bin_size = ( 5*10**6 - 0) / bin_count  # Compute bin size

    try:
        with cp.cuda.Device(device_id):
            # Move data to GPU
            kernel_gpu = cp.asarray(kernel, dtype=cp.float32)
            bin_edges_gpu = cp.asarray(bin_edges, dtype=cp.float32)
            data_gpu = cp.asarray(data, dtype=cp.float32)  # Changed to float32
           
           
            histogram_result_gpu = cp.zeros(bin_count, dtype=cp.uint32)
            total_histogram_result_gpu = cp.zeros(bin_count, dtype=cp.int64)
            # Configure grid and block dimensions for Gaussian filter
            block_dim = (16, 16, 1)
            grid_dim = (
                (width + block_dim[0] - 1) // block_dim[0], 
                (height + block_dim[1] - 1) // block_dim[1],
                batch_size
            )
            # Process each channel
            for channel in range(channels):
                data_channel_gpu = data_gpu[..., channel]
                gaus_result_gpu = cp.empty_like(data_channel_gpu, dtype=cp.float32)
                histogram_gpu = cp.zeros(bin_count, dtype=cp.int32)
              
                gaussian_filter_kernel(
                    grid_dim, 
                    block_dim, 
                    (
                        data_channel_gpu,
                        gaus_result_gpu,
                        kernel_gpu,
                        width,
                        height,
                        kernel_size,
                        batch_size
                    )
                )
                
                fft_result_gpu = cp.abs(cp.fft.fft2(gaus_result_gpu, axes=(1, 2))) 

                histogram_kernel(
                grid_dim, block_dim, 
                    (
                        fft_result_gpu.ravel(),    
                        histogram_gpu,              
                        bin_count,                  
                        global_min,                   
                        global_max ,                                 
                        256*256*batch_size
                    )
                )
                
                total_histogram_result_gpu += histogram_gpu

                
            # Convert the final histogram back to CPU if needed
            total_histogram_result = cp.asnumpy(total_histogram_result_gpu)
            
            return total_histogram_result
        
    except Exception as e:
        print(f"Error in GPU processing: {e}")
        raise
                
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Runtime Error: {e}")
        # Clear memory in case of an error
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        raise

    finally:
        # Ensure memory is cleared even if no exception occurred
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    

# Test and profile the function
def process_chunk(data,kernel,device_id,output_file,lock_file,total_bins=16):
    try:
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        total_window_len=len(data)
        
        window_cnt=len(data)
        window_size=len(data[0])
        channel_size=len(data[0][0][0])

        chunksize = int(.50*total_mem/(window_size*window_size*channel_size*3*8))
       
        # Initialize variables
        combined_histogram = None
       
        acquire_lock(lock_file)

        for i in range(0,total_window_len,chunksize):
            
            histograms = process_on_gpu(data[i:min(total_window_len,i+chunksize)],kernel,device_id)

     
            with cp.cuda.Device(device_id):
                if combined_histogram is None:
                    combined_histogram=histograms

                else:
                    combined_histogram+=histograms

            
           
        with open(output_file, "a") as f:
            f.write(f"{global_min} {global_max} {combined_histogram.tolist()}")
            f.write("\n")

   
        
    except Exception as e:
        print(f"Error occurred on device {device_id}: {e}")

    finally:
        release_lock(lock_file)     


def validate_arguments():
    """
    Validate and process command-line arguments with comprehensive checks.
    
    Returns:
    tuple: Validated (device_id, image_file, output_file, lock_file)
    """
    # Check if sufficient arguments are provided
    if len(sys.argv) < 5:
        print("Error: Insufficient arguments.")
        print("Usage: script.py <device_id> <image_file> <output_file> <lock_file>")
        sys.exit(1)

    # Validate device_id
    try:
        device_id = int(sys.argv[1])
        if device_id < 0:
            raise ValueError("Device ID must be a non-negative integer")
    except ValueError as e:
        print(f"Error with device ID: {e}")
        print("Device ID must be a valid non-negative integer.")
        sys.exit(1)

    # Validate image file
    image_file = sys.argv[2]
    if not os.path.exists(image_file):
        print(f"Error: Image file '{image_file}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(image_file):
        print(f"Error: '{image_file}' is not a regular file.")
        sys.exit(1)
    if not os.access(image_file, os.R_OK):
        print(f"Error: No read permissions for image file '{image_file}'.")
        sys.exit(1)

    # Validate and process output file
    output_file = sys.argv[3]
    if os.path.isdir(output_file):
        print(f"Warning: {output_file} is a directory, not a file. Setting to 'out.data'.")
        output_file = 'out.data'
    
    # Ensure output file can be created/written
    output_dir = os.path.dirname(output_file) or '.'
    if not os.access(output_dir, os.W_OK):
        print(f"Error: No write permissions for output directory '{output_dir}'.")
        sys.exit(1)

    # Validate lock file
    lock_file = sys.argv[4]
    lock_dir = os.path.dirname(lock_file) or '.'
    if os.path.exists(lock_file):
        print(f"Warning: Lock file '{lock_file}' already exists.")
    
    if not os.access(lock_dir, os.W_OK):
        print(f"Error: No write permissions for lock file directory '{lock_dir}'.")
        sys.exit(1)

    return device_id, image_file, output_file, lock_file

# Run the profiler
if __name__ == "__main__":
    kernel = gaus.gaussian_kernel(size=7, sigma=1.0)
     
    gpu_id, image_file, output_file, lock_file = validate_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    #loop over all 4 quads in our file
    for i in range(4):
        if(sys.argv[2].isdigit()):
            ダタ =generate_synthetic_data(int(sys.argv[2]))
        else:
            ダタ = windowRGBValues(image_file, (frame_size,frame_size), (window_size,window_size),i)


        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        process_chunk(ダタ,kernel,gpu_id,output_file,lock_file)

