#!/bin/bash

# Start timing
start_time=$(date +%s.%N)

#Activate the env
source /home/tuo73004/.bashrc
conda activate /home/tuo73004/cpp-openslide

# Global cleanup function
cleanup() {
    echo "Interrupt received. Cleaning up resources..."
    
    # Kill all child processes
    pkill -P $$
    
    # Free up GPU resources
    for gpu_num in $(seq 0 $((max_GPU - 1))); do
        for proc_count in $(seq 0 $((max_jobs_per_gpu))); do
            if [ -n "${gpu_pid[$gpu_num,$proc_count]}" ]; then
                kill ${gpu_pid[$gpu_num,$proc_count]} 2>/dev/null
            fi

            
        done
    rm -f "/tmp/gpu_${gpu_num}_lock"

    done

    echo "Cleanup complete. Exiting..."
    exit 1
}

trap cleanup SIGINT SIGTERM





output_file='out.data'
rm -f $output_file

# Default values
DEFAULT_GPUS=4
DEFAULT_IMAGE_COUNT=9999999999
DEFAULT_MAX_JOBS_PER_GPU=2  # Maximum concurrent jobs per GPU

# Assign inputs to variables
max_GPU=$2
image_list=$1
image_cnt=$3
max_jobs_per_gpu=${4:-$DEFAULT_MAX_JOBS_PER_GPU}

# Check if the user provided at least one argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <image_list_file> [<number_of_GPUs>] [<image_count>] [<max_jobs_per_gpu>]"
    exit 1
fi

# Validate $2 (number_of_GPUs)
if [[ -z "$2" ]]; then
    number_of_GPUs=$DEFAULT_GPUS
elif [[ "$2" =~ ^[0-9]+$ && "$2" -ge 1 && "$2" -le 4 ]]; then
    number_of_GPUs=$2
else
    echo "Error: <number_of_GPUs> must be a number between 1 and 4."
    exit 1
fi

# Validate $3 (image_count)
if [[ -z "$3" ]]; then
    image_count=$DEFAULT_IMAGE_COUNT
elif [[ "$3" =~ ^[0-9]+$ && "$3" -gt 0 ]]; then
    image_count=$3
else
    echo "Error: <image_count> must be a number greater than 1."
    exit 1
fi

# Validate $4 (max_jobs_per_gpu)
if [[ ! "$max_jobs_per_gpu" =~ ^[0-9]+$ || "$max_jobs_per_gpu" -lt 1 ]]; then
    echo "Error: <max_jobs_per_gpu> must be a positive integer."
    exit 1
fi

# Print the values for confirmation
echo "Image list file: $1"
echo "Number of GPUs: $number_of_GPUs"
echo "Image count: $image_count"
echo "Max jobs per GPU: $max_jobs_per_gpu"

# Read the image paths into an array
mapfile -t images < "$image_list"
total_images=${#images[@]}

if [ $total_images -eq 0 ]; then
    echo "Error: No images found in the list."
    exit 1
fi

# Export the Python script name for use in subshells
script="histogramTest.py"



# Counter for total images processed
processed=0

echo "Starting!"
echo ""
echo ""


# Initialize an array to track GPU usage status
declare -a gpu_in_use
declare -A gpu_pid
declare -a lock_files  # Array to store lock file paths
for gpu_num in $(seq 0 $((max_GPU - 1))); do
    gpu_in_use[gpu_num]=0
    lock_files[gpu_num]="/tmp/gpu_${gpu_num}_lock"
done

mem_alloc=0

# Loop through the images until the target count is reached
while [ $processed -lt $image_cnt ]; do
    # Try to allocate jobs to available GPUs
    for gpu_num in $(seq 0 $((max_GPU - 1))); do
        # Skip if this GPU is at max capacity
        if [ ${gpu_in_use[gpu_num]} -ge $max_jobs_per_gpu ]; then
            continue
        fi

        # Skip if memory allocation is too high
        if [ $mem_alloc -ge 240 ]; then
            echo "lamo too much mem chill out dawg "
        fi

        # Check if we have more images to process
        if [ $processed -ge $image_cnt ]; then
            break
        fi
        
        # Assign an image to the process
        image_index=$((processed % total_images))
        image_path=${images[$image_index]}
        
        # Mark the GPU as in use
        ((gpu_in_use[gpu_num]+=1))
        ((mem_alloc+=10))
        
        # Start the Python script for the current process
        python "$script" "$gpu_num" "$image_path" "$output_file" "${lock_files[$gpu_num]}" &
        
        # Store the PID of the background process
        gpu_pid[$gpu_num,$((gpu_in_use[gpu_num]-1))]=$!
        
        # Increment the processed count
        ((processed++))
    done

    # Check for completed processes and free up GPUs
    for gpu_num in $(seq 0 $((max_GPU - 1))); do
        if [ ${gpu_in_use[gpu_num]} -ge 1 ]; then
            for ((proc_count=0; proc_count<gpu_in_use[gpu_num]; proc_count++)); do 
                # Check if the process is still running
                if [ -n "${gpu_pid[$gpu_num,$proc_count]}" ] && ! kill -0 "${gpu_pid[$gpu_num,$proc_count]}" 2>/dev/null; then
                    # Process has completed, free up the GPU

                    ((gpu_in_use[gpu_num]--))
                    ((mem_alloc-=10))
                    
                    # Remove the completed PID
                    unset "gpu_pid[$gpu_num,$proc_count]"
                fi
            done
        fi
    done

    # Small sleep to prevent tight looping
    sleep 0.1
done

# Wait for any remaining background processes to complete
wait

python process.py $output_file


# End timing
end_time=$(date +%s.%N)

# Calculate elapsed time with precision
elapsed_time=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")
echo "All tasks completed in $elapsed_time seconds."

# Calculate average time
average_time=$(awk "BEGIN {printf \"%.3f\", $elapsed_time / $image_count}")
echo "Average task completed in $average_time seconds."