# High Performance Computing - OpenCL

# Abstract

The architecture of high-performance computing systems is becoming more and more heterogeneous, as Heavily Parallelised Compute has the ability to run on both CPU and GPU devices. In this paper, I use the Lattice Boltzmann Method (D2Q9) code implemented in OpenCL and analyze its performance on each speed improved stage. I also compare with results obtained with codes optimized and implemented in OpenMP. My work shows that specific arithmetic with large data sets using GPU hardware will be faster than the CPU.

# Introduction

Both CPU and GPU have multiple cores in general, however, GPU devices typically have a significantly larger number of cores than CPU. Therefore, GPU ordinarily has insanely high memory bandwidths, and it is ideally suited for computations that can be run in parallel. This generally involves arithmetic on large data sets, where the same operation can be performed across thousands of elements at the same time. Additionally, an efficient computation will need both to run properly.

The Lattice Boltzmann Method (LBM) is a technique for simulating the movement of complex fluid systems. Fluid systems are used in many industries to transmit signals and power using a network of tanks, pipes, valves, pumps and other flow devices. This paper describes how to accelerate the LBM technique by using OpenCL – the open standard for parallel programming.

Open Computing Language(OpenCL) is a framework for running code on both sets of computer hardware: a host system that relies on one or more CPUs and a device system with one or more OpenCL-enabled GPUs. The part running on the host usually performs initialization and launches and controls kernels running on devices, and devices run those parts of the algorithm with thousands of lightweight threads in parallel. To use OpenCL effectively, it’s important to understand the differences of the design between the device and the host system, and how to determine the performance of OpenCL applications. The performance of OpenCL applications depends on several factors, such as the instruction set architecture, the precision of computations and number of cells. For this reason, rather than discussing hardware components, it's important to measure the performance of the implemented codes in order to best structure the OpenCL code for faster execution.

# Code implementation

In order to call a kernel that can be executed on an OpenCL device, host side needs to create a context and queue, create and build the program, setup memory objects, define the kernel, pass the data and enqueue commands. The host can read the data from the device after all. In the device side, OpenCL programs generate many work-items perform the kernel functions parallelly, usually each work-item processes one item of the data-set.

The OpenCL compiles the data and tasks in the kernel functions. Kernels are executed by a set of work-item, and the kernel code specifies the instructions executed by each work-item. Work-items are collected into work-groups and each work-group executes on a computer unit. Each work-item has a global ID and a local ID. The global ID is a unique identifier among all work-items of a kernel. The local ID identifies a work-item within a work-group. Work-items in different work-groups may have the same local ID, but they do not have the same global ID. Moreover, each work-group has a unique group ID.

The OpenCL memory model contains several spaces with different size and access time. The data in global/constant memory is mapped on the entire device, the data in local memory is shared by work-items only within a work-group, and the data in private memory is used only by individual work-item. When the host transfers data to and from the device, data is stored in and read from global memory. Global memory is the largest memory region on an OpenCL device, but it is the slowest for work-items to access. Work-items access local memory much faster than global memory. Local memory is not as large as global memory, but because of its access speed, it is a good place for work-items to store their intermediate results. Finally, each work-item has exclusive access to its private memory, this is the fastest storage place, but it is commonly much smaller than all other address spaces, so it is important not to use private memory too much. Furthermore, since each work-item and work-group cannot communicate with other work-items and work-groups, it is also important to consider the parallelism instructions.

## Kernel functions

Firstly, I moved rebound and collision functions into the kernel by following the structures of OpenCL programming.

Compare the running time of each function in the different processor: CPU implementation, CPU parallel implementation OpenMP and GPU implementation OpenCL, the results show that for small problem-scale and the simple calculation (e.g. accelerate_flow function), CUP incredibly performs the best. It can be considered as in a one-for-one comparison CPU cores are typically faster than GPU cores. However, with if statement, OpenMP does not perform as well as expected. In contrast, when it comes to the heterogeneous computing (e.g. collision), OpenCL and OpenMP perform almost 10 times better than CPU implementation. In addition, propagate stage needs to implement some out of order memory accesses to swap data between adjacent lattice points, and rebound stage also needs to read from and write to memory, all those memory access ultimately limits the performance of the code.

accelerate_flow 0.6s / 0.006s / 1.0s+-0.3s
propagate 1.4s / 4.3s / 1.0s+-0.3s
rebound 0.8s / 0.6s / 1.0s+-0.3s
collision 1.9s / 14s / 2.0s+-0.3s

Overall OpenCL stably performs the best. In OpenCL, data values must be transferred from the host to the device. However, the bandwidth between the device memory and the GPU is much higher than the bandwidth between host memory and device memory. Therefore, these transfers are costly in terms of performance and so they should be minimized. Programs that run multiple kernels on the same data should favour leaving the data on the device between kernel calls, rather than transferring intermediate results to the host and then sending them back to the device for subsequent calculations.

In conclusion, for the best overall application performance, all lattice Boltzmann codes should be handled in the kernel by OpenCL, in order to keep data on the device as long as possible. Therefore, I moved all functions into the kernel, even for some stages running kernels on the GPU does not demonstrate any speed-up compared with running them on the host CPU.

## Optimization

Secondly, after moving all functions into the kernel, I implement some optimizations which are efficient for CPU codes: merged all stages into only one function; unloop the small loop to accelerate the calculation of local density; use a ternary operator to select store value instead of using if statement. All of these optimizations aim to reduce memory access and achieve better performance.

Coalesced memory access or memory coalescing refers to combining multiple memory accesses into a single transaction. On the K20 GPUs, every successive memory can be accessed by a warp in a single transaction. However, when memory is not sequential, memory access is sparse or misaligned memory access, these may result in uncoalesced load. How the matrix is laid out will affect the memory access.

Thirdly, in order to keep matrix to reside linearly in memory as much as possible, I change the data type of _cells_ and _tmp_cells_ , split _t_speed_ into several independent arrays. In another word, use arrays instead of the structure of arrays.

The results show that performance is significantly improved after moving all functions into the kernel, there is no more cells and tmp_cells transfer between the host and the device. And the performance becomes even better with optimizations, merging all stages together reduces a lot of global memory accesses. Changing the data type slightly speed up the computation, every thread still requires data from the neighbouring lattice points.

128 \* 128 8.7 / 3.5 / 2.9
265 \* 265 68.7 / 19.4 / 16.3

## Reduction

At the end of the av_velocity stage, the host still needs to read the data from the device and performs the sum which cost 1.5s in total approximately. In order to improve its performance, I tried some different methods to apply the reduction for calculating the av_vels.

Firstly, use OpenMP on host performs partial in each thread to do the reduction of an array of elements to a single value. It's the simplest way to improve the performance on the CPU, only needs one line to apply the OpenMP. However, the performance gets worse to some extent. Considering opening parallel regions is expensive, it's not worthy for simple loop and small data set, and the data transfer from device to host still cost high memory bandwidth after all.

128 \* 128 3.8
265 \* 265 42.0
1024 \* 1024 42.0

Secondly, use one work-item to perform a sum within a work-group and store the sum in global memory, then return the array of the partial sum to host. The work-items read from global memory serially which is very inefficient, and only one work-item forms the partial sum is not the parallel process literally.

figure 1

Thirdly, work-items perform the partial sum and copy it to the local memory in parallel, wait for work-items within the same work-group finishing the task, use one work-item performs the partial sum. Using local memory instead of global memory speeds up the memory access, and the size of data transfer is reduced which leads to better performance.

figure 2

In this case, the size of the local memory is the critical element for performance. However, tuning the kernel work-group size for GPUs is a challenging problem. There are many factors may affect the choice of a good work-group size. Depends on the data re-use ratio and coalesced ness of access pattern. Random access to local memory is much faster than random access to global memory. While using too much local memory/private file, it can be even slower because more local memory consumption leads to less occupation and less memory latency hiding and more register spilling to global memory.

Therefore, in order to find the best fit work-group size, I simply compared their running time by applying different size of local memory: 8\*8, 16\*16 and 32\*32. According to the results, obviously using 16\*16 as the local memory size achieves the best performance. However, it is still not an efficient way of using only one work-item to perform the partial sum.

128 \* 128 2.6 | 2.4 | 5.9
265 \* 265 11.5 | 12.0 | 27.0
1024 \* 1024 32.6 | 24.3 | 94.0

Fourthly, use binary addition to improve on the linear sum. Same as the last solution, work-items perform the partial sum and copy it to the local memory in parallel, but adding one-half values to the other repeatedly, until getting the global sum.

figure 3

By forming the sum parallelly in local memory, the performance gets plainly better compared to the former solution. However, reading the data from the device and performing the sum on the host still cost 0.5s roughly for 128\*128 data set.

128 \* 128 1.9
265 \* 265 8.2
1024 \* 1024 19.1

Finally, set global work size to be half of the input array length, slightly re-order to remove a couple of loop iterations. Then perform the sum parallelly same as the former solution. In order to change the global work size without affecting other stages, I sperate the av_velocity kernel. Use 16, 16*16, 32*32 respectively as local work size to implement the reduction.

However, the running time is about from 2.5s to 2.8s, it does not improve the performance. There are some factors affecting can be considered, in this case, each work-item needs to do double calculation task including the arithmetic of local density and so on, so this can cost more time than expected when it only does add up two of the input values.

figure 4

# Conclusions and future works

Combine all stages in OpenCL kernel to implement the lattice Boltzmann code gets the better performance than CPU and OpenMP implementations. In addition, doing reduction in GPU parallely improves the performance even more. The best running time I achieved shows below.

but ultimately limited by the global memory bandwidth available. The Altera OpenCL compiler offers an alternative OpenCL programming model that creates one or more pipelined kernels, where parallelism comes from the complexity of the pipeline. The more complex the pipeline, the more floating logic is performed in parallel.

tiles hola exchange?

however the propagate stage must implement some out of order memory accesses to swap data between adjacent lattice points.
For a GPU implementation, it is the global memory access that ultimately limits the performance of the lattice Boltzmann code.

Typical OpenCL Lattice Boltzmann implementations work by creating hundreds of threads, all working in parallel, but ultimately limited by the global memory bandwidth available. The Altera OpenCL compiler offers an alternative OpenCL programming model that creates one or more pipelined kernels, where parallelism comes from the complexity of the pipeline. The more complex the pipeline, the more floating logic is performed in parallel.

However, the streaming stage requires data from the neighboring lattice points. The delivery of the data can be optimized by using a cached copy of the output in what is referred to as a Sliding Window. A Sliding Window approach allows data to be read linearly and buffered in local memory, from which data can be read as often as required.

figure 5

128 \* 128 1.9
265 \* 265 8.2
1024 \* 1024 19.1

# Reference

(13) (PDF) A Portable OpenCL Lattice Boltzmann Code for Multi- and Many-core Processor Architectures. Available from: https://www.researchgate.net/publication/262932546_A_Portable_OpenCL_Lattice_Boltzmann_Code_for_Multi-_and_Many-core_Processor_Architectures [accessed Apr 01 2019].

https://wiki.tuflow.com/index.php?title=Hardware_Benchmarking_Topic_HPC_on_CPU_vs_GPU

https://www.nallatech.com/fpga-acceleration-of-lattice-boltzmann-using-opencl/

https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Best_Practices_Guide.pdf

https://ukopenmpusers.co.uk/wp-content/uploads/uk-openmp-users-2018-OpenMP45Tutorial_new.pdf

https://www.cl.cam.ac.uk/teaching/1617/AdvGraph/07_OpenCL.pdf

https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/opencl/opencl-05-reduction.pdf?__blob=publicationFile

http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.2pp.pdf

https://dournac.org/info/gpu_sum_reduction

https://fgiesen.wordpress.com/2017/04/11/memory-bandwidth/

http://www.admin-magazine.com/HPC/Articles/Finding-Memory-Bottlenecks-with-Stream

https://cvw.cac.cornell.edu/gpu/coalesced
