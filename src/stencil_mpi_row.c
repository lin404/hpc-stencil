#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(int rank, int size, const int local_ncols, const int local_nrows, float **  image, float **  tmp_image);
void init_image(int rank, int size, const int nx, const int ny, float **  image);
double wtime(void);

int calc_nrows_from_rank(int NCOWS, int rank, int size);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  int i,j;                /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  int tag1 = 1;           
  MPI_Status status;      /* struct used by MPI_Recv */
  MPI_Request request;    /* request for action used by async functions */ 
  int local_nrows;        /* number of rows apportioned to this rank */
  int local_ncols;        /* number of columns apportioned to this rank */
  int master_nrows;       /* number of columns apportioned to a remote rank */
  int remote_nrows;       /* number of columns apportioned to a remote rank */
  double boundary_mean;   /* mean of boundary values used to initialise inner cells */
  float **tmp_image;      /* local temperature grid at time t - 1 */
  float **image;          /* local temperature grid at time t     */
  float *printbuf;        /* buffer to hold values for printing */

  int t;                  /* index for timestep iterations */
  double maximum = 0.0;
  double local_maximum = 0.0;

  /* MPI_Init returns once it has started up processes */
  /* get size and rank */ 
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = calc_nrows_from_rank(ny, rank, size);
  local_ncols = nx;

  // Allocate the image
  tmp_image = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(i=0;i<local_nrows + 2;i++) {
    tmp_image[i] = (float*)malloc(sizeof(float) * local_ncols);
  }
  image = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(i=0;i<local_nrows + 2;i++) {
    image[i] = (float*)malloc(sizeof(float) * local_ncols);
  }

  for(i=0;i<local_nrows + 2;i++) {
    for(j=0;j<local_ncols;j++) {
      tmp_image[i][j] == 0.0f;
      image[i][j]==0.0f;
    }
  } 

  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_nrows = calc_nrows_from_rank(nx, size-1, size); 
  printbuf = (float*)malloc(sizeof(float) * ((local_ncols)));

  // Set the input image
  init_image(rank, size, local_ncols, local_nrows, image);

  // Call the stencil kernel
  double tic = wtime();
  for (t = 0; t < niters; ++t) {
    stencil(rank, size, local_ncols, local_nrows, image, tmp_image);
    stencil(rank, size, local_ncols, local_nrows, tmp_image, image);
  }
  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  // Output the image
  if(rank == MASTER){
    // Open output file
    FILE *fp = fopen(OUTPUT_FILE, "w");
    if (!fp) {
      fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
      exit(EXIT_FAILURE);
    }

    // Ouptut image header
    fprintf(fp, "P5 %d %d 255\n", nx, ny);

    // find the maximum
    for (j = 1; j < local_nrows+1; ++j) {
      for (i = 0; i < local_ncols; ++i) {
        if (image[j][i] > maximum)
          maximum = image[j][i];
      }
    }

    for(kk=1;kk<size;kk++) {
      MPI_Recv(&local_maximum,1,MPI_DOUBLE,kk,tag1,MPI_COMM_WORLD,&status);
      if (local_maximum > maximum)
          maximum = local_maximum;
    }

    // Printout rank master
    for(j=1;j<=local_nrows;++j) {
      for (i = 0; i < local_ncols; ++i) {
        fputc((char)(255.0*image[j][i]/maximum), fp);
      }
    }
    
    // Printout other ranks
    for(kk=1;kk<size;kk++) {
      for(j = 1; j < local_nrows+1; ++j) {
        MPI_Recv(printbuf,local_ncols,MPI_FLOAT,kk,j,MPI_COMM_WORLD,&status);
        for (i = 0; i < local_ncols; ++i) {
          fputc((char)(255.0*printbuf[i]/maximum), fp);
        }
      }
    }

    // Close the file
    fclose(fp);

  }else {
    for (j = 1; j < local_nrows+1; ++j) {
      for (i = 0; i < local_ncols; ++i) {
        if (image[j][i] > local_maximum)
          local_maximum = image[j][i];
      }
    }
    MPI_Send(&local_maximum,1,MPI_DOUBLE,MASTER,tag1,MPI_COMM_WORLD);
    for(j = 1; j < local_nrows+1; ++j) {
      MPI_Send(image[j],local_ncols,MPI_FLOAT,MASTER,j,MPI_COMM_WORLD);
    }
  } 

  MPI_Finalize();

  /* free up allocated memory */
  // for(i=0;i<local_nrows+2;i++) {
  //   free(tmp_image[i]);
  //   free(image[i]);
  // }
  // for(i=0;i<remote_nrows+2;i++) {
  //   free(printbuf[i]);
  // }

  // free(tmp_image);
  // free(image);
  // free(printbuf);

  /* and exit the program */
  return EXIT_SUCCESS;
}

void stencil(int rank, int size, const int local_ncols, const int local_nrows, float ** image, float ** tmp_image) {
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int i,j;             /* row and column indices for the grid */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */

  sendbuf = (float*)malloc(sizeof(float) * local_ncols);
  recvbuf = (float*)malloc(sizeof(float) * local_ncols);
  
  /* 
  ** determine process ranks to the left and right of rank
  ** respecting periodic boundary conditions
  */
  left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  right = (rank + 1) % size;

  /* send to the left, receive from right */
  for(i=0;i<local_ncols;i++){
    sendbuf[i] = image[1][i];
    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, left, tag,
		recvbuf, local_ncols, MPI_FLOAT, right, tag,
		MPI_COMM_WORLD, &status);
  }
  
  if(rank!=size-1){
    for(i=0;i<local_ncols;i++){
      image[local_nrows+1][i] = recvbuf[i];
    }
  }

  /* send to the right, receive from left */
  for(i=0;i<local_ncols;i++){
    sendbuf[i] = image[local_nrows][i];
    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, right, tag,
		recvbuf, local_ncols, MPI_FLOAT, left, tag,
		MPI_COMM_WORLD, &status);
  }
  if(rank!=MASTER){
    for(i=0;i<local_ncols;i++){
      image[0][i] = recvbuf[i];
    }
  }

  for (j = 1; j < local_nrows+1; ++j) {
    for (i = 1; i < local_ncols-1; ++i) {
      tmp_image[j][i] = image[j][i] * 3.0f/5.0f
        + image[j-1][i] * 0.5f/5.0f // TOP
        + image[j+1][i] * 0.5f/5.0f // BELOW
        + image[j][i-1] * 0.5f/5.0f // LEFT
        + image[j][i+1] * 0.5f/5.0f; // RIGHT
      }
  }

  for (i = 1; i < local_nrows+1; ++i) {
    // FIRST column
    tmp_image[i][0] = image[i][0] * 3.0f/5.0f
      + image[i-1][0] * 0.5f/5.0f // TOP
      + image[i+1][0] * 0.5f/5.0f // BELOW
      + image[i][1] * 0.5f/5.0f; // RIGHT
    
    // LAST column
    tmp_image[i][local_ncols-1] = image[i][local_ncols-1] * 3.0f/5.0f
      + image[i-1][local_ncols-1] * 0.5f/5.0f // TOP
      + image[i+1][local_ncols-1] * 0.5f/5.0f // BELOW
      + image[i][local_ncols-2] * 0.5f/5.0f; // LEFT
  }

  free(sendbuf);
  free(recvbuf);
}

// Create the input image
void init_image(int rank, int size, const int nx, const int ny, float ** image) {
  int j, i, global_j;
  // Checkerboard
  for (j = 0; j < ny; ++j) {
    global_j = rank*ny+j;
    for (i = 0; i < nx; ++i) {
      if (((i/(nx/8))%2) == ((global_j/(ny*size/8))%2)) {
        image[j+1][i] = 0.0f;
      } else {
        image[j+1][i] = 100.0f;
      }   
    }
  }
}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}

int calc_nrows_from_rank(int NCOWS, int rank, int size)
{
  int nrows;

  nrows = NCOWS / size;       /* integer division */
  if ((NCOWS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += NCOWS % size;  /* add remainder to last rank */
  }
  
  return nrows;
}