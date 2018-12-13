# MPI Explorations

## Abstract



## Introduction
MPI - Message Passing Interface is a specification of message passing libraries, it provides a lot of functions which developers can use to specify multiple computers or multiple processor cores within the same computer to identify and implement a single parallel program across distributed memory.

A communicator defines a group of MPI processes. When an MPI application starts, the group is initially given a predefined name called MPI_COMM_WORLD, for simple programs we do not have to worry about names. Additionally, in this group, each process is assigned a unique rank.

Each process starts its own part of the program and explicitly communicates with other processes by exchanging data. For instance, one process sends a copy of the data to another process, and the other process receives it. Communications such as this which involve one sender and receiver are known as point-to-point communications.

Basic send/receive is blocking communication, there is also non-blocking communication in MPI. MPI_Send() and MPI_Recv() are blocking functions. The process sending data will be blocked until data has been delivered and the buffer is emptied. The process receiving data will be blocked until a matching message is received from the system and the buffer is filled. Blocking communication is simple to use but can be prone to deadlocks. MPI_ISEND() and MPI_IRECV() are non-blocking functions. These functions return immediately even if the communication is not finished yet. Non-blocking communication leads to improved performance but needs to use MPI_Wait() or MPI_Test() to see whether the communication has finished.

One commonly used communication pattern for domain decomposition problems is halo exchange. Each process stores its boundary data in halo regions. At every timestep, these
regions are exchanged with neighbouring processes so that every process has access to the correct information. The processes then use their newly acquired information to update the halo
regions for the next timestep. For a big calculation of a grid of cells, we can consider decomposing original grid into multiple smaller grids, each owned by a different rank, so that each rank only calculates its own data. The gird can be decomposed by columns or rows, or as tiles.

## Halo Exchange - Decompose by rows
To achieve better performance, instead of starting with decomposing by columns, I challenged decompose the grid by rows firstly, since I know row-major order leads to improved performance in computing from last assignment.
Split a grid of cells by rows equally to every rank, each rank sends the data of its first row of core regions to the rank-1(if it's the first rank, it will send to the last rank), and sends the data of its last row of core regions to the rank+1(if it's the last rank, it will send data to the first rank). Each rank receives the data from rank-1 and rank+1, then updates the data of its halo regions. (Figure_1)

### MPI_Sendrecv
Since non-blocking functions start data exchange immediately which gets better performance, I use MPI_IRECV instead of MPI_RECV. Each rank needs to send and receive data, so instead of MPI_Irecv, MPI_Send and MPI_Wait, using MPI_Sendrecv may not be faster, but more convenient, the send-receive operations combine in one call the sending of a message to one destination and the receiving of another message from another process.

## Halo Exchange - Decompose as tiles
Secondly, I tried to decompose the grid as tiles. Split a grid of cells as tiles to every rank, each rank sends the boundary data of kernel to each neighbour rank and receives the data from all neighbour ranks, then updates the data of its halo regions. (Figure_2)

### MPI_DIMS_CREATE
Not like decompose by rows, the neighbour ranks of rank x can be specified as rank x-1 and rank x+1, I choose to use cartesian topologies to easily locate all neighbour ranks in a mesh. In MPI, the function MPI_DIMS_CREATE helps developers select a balanced distribution of processes per coordinate direction, depending on the number of processes in the group to be balanced. The entries in the array dim are set to describe a cartesian grid with ndims dimensions and a total of nnodes nodes. The dimensions are set to be as close to each other as possible, using an appropriate divisibility algorithm. I use it to partition all the processes into a 2-dimensional topology, and set dims[i] = 0 so that the routines are modified.

### MPI_CART_CREATE
Paired with MPI_DIMS_CREATE, MPI_CART_CREATE returns a handle to a new communicator to which the cartesian topology information is attached. If reorder = true, the rank of each process will be reordered onto the physical machine, which will cause the errors in this stencil code for specifical calculation of boundary. Therefore, I set reorder = false(0), so the rank of each process in the new group is identical to its rank in the old group. Additionally, as the specifical calculation of boundary in the stencil, I set periods[i] = false(0), specifying the grid is not periodic in each dimension. In other words, coordinate 0 in dimension n is not a neighbour of coordinate n_max (Figure_3), so that the data of boundary will not be updated.

### MPI_Neighbor_alltoall
Consider each rank passes the message to all 4 of neighbour ranks, rather than using MPI_Send/MPI_IRecv to optimise each message individually, I use MPI_Neighbor_alltoall for message exchange where MPI can optimise the whole pattern of messages. So that I only need to package all messages, then MPI will send the specified size of messages to each neighbour rank at once, and each rank receives all messages from all neighbour ranks at once, then unpackages the message and updates the data of halo region. By using MPI_Neighbor_alltoall, the code is much simpler and easier to read.

## Performance Analysis
I completed the code with 2 different patterns of halo exchange, decompose by rows and decompose as tiles. I compared their runtime by running code on 2~16 cores (Figure_4). The results shows that decompose as tiles performs much better than decompose by rows. And in decompose by rows, the runtime does not change much with more than 4 cores. However, in decompose as tiles, the performance is getting better with 16 cores, but the performance is not as good as expected like a linear.
With using single precision, the operational intensity of stencil is 0.375. Refer to roofline model of Intel Xeon CPU E5-2670 (Sandy Bridge) on 16 cores, when OI = 0.375, the rate limiting is memory bandwidth bound, so performance is all about moving bytes. With 16 cores, the peak DRAM bandwidth is about 66 GB/s, tiles halo exchange code is using  

mpirun -n 16 -l amplxe-cl -quiet -collect hotspots -trace-mpi -result-dir vtune.txt my_app ./



## Conclusion
There is room for improvement，as the best result I could get from halo exchange(decompose as tiles) shows below. (Figure_3)




## Reference
Wikipedia. 2018. Message Passing Interface. [ONLINE] Available at: https://en.wikipedia.org/wiki/Message_Passing_Interface. 

Wikipedia. 2018. SPMD. [ONLINE] Available at: https://en.wikipedia.org/wiki/SPMD. 
Definition from WhatIs.com. 2018. message passing interface (MPI). [ONLINE] Available at: https://searchenterprisedesktop.techtarget.com/definition/message-passing-interface-MPI.

Partnership for Advanced Computing in Europe, 2013, “An Interface for Halo Exchange Pattern.” Bianco, Mauro. [ONLINE] Available at http://www.prace-project.eu/IMG/pdf/wp86.pdf.

Argonne MPI Tutorials. 2013. Introduction to MPI. [ONLINE] Available at: http://www.mcs.anl.gov/~balaji/permalinks/2014-06-06-argonne-mpi-basic.pptx

Open MPI Softwarem Documentation. 2018. MPI_Sendrecv(3) man page (version 4.0.0). [ONLINE] Available at: https://www.open-mpi.org/doc/v2.0/man3/MPI_Sendrecv.3.php

Cartesian Convenience Function. 2000. 6.5.2. Cartesian Convenience Function: MPI_DIMS_CREATE. [ONLINE] Available at: https://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.1/node134.htm
	
 Open MPI Software. 2018. MPI_Cart_create(3) man page (version 1.6.5). [ONLINE] Available at: https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_create.3.php