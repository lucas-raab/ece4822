# ece8527  
Repository showcasing my work for the course ECE 4822: Engineering Computation IV. [Course web page](https://isip.piconepress.com/courses/temple/ece_4822/)

This repo is divided into two sections:

# Homework
This section contains the code for each individual homework assignment for the course. Most of these assignments involved simple programs optimized to run efficiently using various techniques such as sophisticated algorithms, parallel programming, and, by the end of the course, GPUs.

A major focus was on matrix multiplication, with my benchmarks available [here](https://isip.piconepress.com/courses/temple/ece_4822/resources/benchmarks/matrix_multiplication/2024_01_fall/).

# Final Project
The final project involved a five-stage program applying signal processing techniques to pathology slides. The class competed for the fastest processing time.

The leaderboard is available here. I placed first with a time of 22.97 seconds.

### Design Process
The system featured four NVIDIA A40 GPUs, 180 GB of VRAM, 256 GB of RAM, and two AMD EPYC 7413 24-core processors. To optimize performance, I focused on efficient hardware utilization.

Each image was large enough to fit only one per GPU. Network bottlenecks required continuous loading, so I split the images into quadrants, loading 1/4 at a time.

The key optimization was pipelining. While the GPU processed data sequentially, the CPU loaded images in parallel.

I used a startup script to spawn an instance per GPU. Each instance managed a file, loaded 1/4 at a time, locked an available GPU, processed data, and generated a histogram.

This method ensured continuous operation—one process worked on the GPU while another loaded data or waited for a lock.

To improve this, I would optimize the startup process. Initially, two processes per GPU loaded files, causing inefficiencies.

As the cluster's sysadmin, I definitely need to work on improving some of the I/O bottlenecks I observed. Given the small dataset, SSD caching could have improved performance. Moving the data locally to the compute node might have also reduced a lot of the variability.

# Conclusion
Overall, this has been one of my favorite courses. I enjoyed the competition and the challenge of competing for the fastest times. It also involved a type of programming I hadn’t dealt with before, and it was great to learn.

Thanks, Dr. Picone!

![image](https://github.com/user-attachments/assets/5f2d01b6-2738-4b36-a808-b037ec2d156c)
