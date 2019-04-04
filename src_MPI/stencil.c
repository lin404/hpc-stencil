#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0
#define NDIMS 2  /* setting the number of dimensions in the grid with a macro */

void stencil(int rank, int size, const int nx, const int ny, float **  image, float **  tmp_image, MPI_Comm comm_cart);
void init_image(int rank, int size, const int nx, const int ny, float **  image);
int calc_nrows_from_rank(int NCOWS, int rank, int size);
double wtime(void);

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

  int i,j;             /* row and column indices for the grid */
  int ii,jj;             /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int rank;              /* the rank of this process */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  int tag1 = 1;           /* scope for adding extra information to a message */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_ncols;      /* number of columns apportioned to a remote rank */
  double boundary_mean;  /* mean of boundary values used to initialise inner cells */
  float **tmp_image;            /* local temperature grid at time t - 1 */
  float **image;            /* local temperature grid at time t     */
  float *printbuf;      /* buffer to hold values for printing */

  int t;              /* index for timestep iterations */
  double maximum = 100.0;
  double local_maximum = 0.0;

  int reorder = 0;       /* an argument to MPI_Cart_create() */
  int dims[NDIMS];       /* array to hold dimensions of an NDIMS grid of processes */
  int periods[NDIMS];    /* array to specificy periodic boundary conditions on each dimension */
  
  MPI_Comm comm_cart;    /* a cartesian topology aware communicator */
  MPI_Status status;     /* struct used by MPI_Recv */
  MPI_Request request;

  /* MPI_Init returns once it has started up processes */
  /* get size and rank */ 
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  if (size < (NDIMS * NDIMS)) {
    fprintf(stderr,"Error: size assumed to be at least NDIMS * NDIMS, i.e. 4.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if ((size % 2) > 0) {
    fprintf(stderr,"Error: size assumed to be even.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /* Initialise the dims and periods arrays */
  for (ii=0; ii<NDIMS; ii++) {
    dims[ii] = 0;
    periods[ii] = 0; /* set periodic boundary conditions to True for all dimensions */
  }

  MPI_Dims_create(size, NDIMS, dims);
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &comm_cart);

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = calc_nrows_from_rank(ny, rank, size);
  local_ncols = calc_nclos_from_rank(nx, rank, size);

  tmp_image = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(i=0;i<local_nrows + 2;i++) {
    tmp_image[i] = (float*)malloc(sizeof(float) * local_ncols + 2);
  }
  image = (float**)malloc(sizeof(float*) * (local_nrows + 2));
  for(i=0;i<local_nrows + 2;i++) {
    image[i] = (float*)malloc(sizeof(float) * local_ncols + 2);
  }
  for(j=0; j<local_nrows + 2; j++){
    for (i=0;i<local_ncols + 2;i++){
      image[j][i]=0.0f;
      tmp_image[j][i]=0.0f;
    }
  }

  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  remote_ncols = calc_nclos_from_rank(nx, size-1, size); 
  printbuf = (float*)malloc(sizeof(float) * ((remote_ncols+2)));

  // Set the input image
  init_image(rank, size, local_ncols, local_nrows, image);

  // Call the stencil kernel
  double tic = wtime();
  for (t = 0; t < niters; ++t) {
    stencil(rank, size, local_ncols, local_nrows, image, tmp_image, comm_cart);
    stencil(rank, size, local_ncols, local_nrows, tmp_image, image, comm_cart);
  }
  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  // Printout the image
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

    for(jj= 0; jj < 4; ++jj) {
      for(j=1;j< local_nrows+1;++j) {
        for(kk=jj;kk<jj+13;kk+=4) {
          if(kk==0){
            for (i = 1; i < local_ncols+1; ++i) {
              fputc((char)(255.0*image[j][i]/maximum), fp);
            }
          }else{
            MPI_Recv(printbuf,local_ncols+2,MPI_FLOAT,kk,j,MPI_COMM_WORLD,&status);
            for (i = 1; i < local_ncols+1; ++i) {
              fputc((char)(255.0*printbuf[i]/maximum), fp);
            }
          }
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
      MPI_Send(image[j],local_ncols+2,MPI_FLOAT,MASTER,j,MPI_COMM_WORLD);
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
  // free(&local_maximum);

  /* and exit the program */
  return EXIT_SUCCESS;
}

void stencil(int rank, int size, const int nx, const int ny, float ** image, float ** tmp_image, MPI_Comm comm_cart) {
  int i,j;             /* row and column indices for the grid */
  float *sendbuf;       /* buffer to hold values to send */
  float *recvbuf;       /* buffer to hold received values */

  sendbuf = (float*)malloc(sizeof(float) * (nx*2+ny*2));
  recvbuf = (float*)malloc(sizeof(float) * (nx*2+ny*2));
  
  for(i=0; i<ny; ++i) sendbuf[i] = image[i+1][1]; // pack west
  for(i=0; i<ny; ++i) sendbuf[ny+i] = image[i+1][nx]; // pack east
  for(i=0; i<nx; ++i) sendbuf[2*ny+i] = image[1][i+1]; // pack north
  for(i=0; i<nx; ++i) sendbuf[2*ny+nx+i] = image[ny][i+1]; // pack south

  MPI_Neighbor_alltoall(sendbuf, nx, MPI_FLOAT, recvbuf, nx, MPI_FLOAT, comm_cart);

  for(i=0; i<ny; ++i) image[i+1][0] = recvbuf[i]; // unpack west
  for(i=0; i<ny; ++i) image[i+1][nx+1] = recvbuf[ny+i]; // unpack east
  for(i=0; i<nx; ++i) image[0][i+1] = recvbuf[2*ny+i]; // unpack north
  for(i=0; i<nx; ++i) image[ny+1][i+1] = recvbuf[2*ny+nx+i]; // unpack south

  for (j = 1; j < ny+1; ++j) {
    for (i = 1; i < nx+1; ++i) {
      tmp_image[j][i] = image[j][i] * 3.0f/5.0f
      + image[j][i-1] * 0.5f/5.0f
      + image[j][i+1] * 0.5f/5.0f
      + image[j-1][i] * 0.5f/5.0f
      + image[j+1][i] * 0.5f/5.0f;
    }
  }

  free(sendbuf);
  free(recvbuf);
}

// Create the input image
void init_image(int rank, int size, const int nx, const int ny, float ** image) {
  int j, i, global_j, global_i;
  // Checkerboard
  for (j = 0; j < ny; ++j) {
    for (i = 0; i < nx; ++i) {
      if (((i/(nx/2))%2) == ((j/(ny/2))%2)) {
        image[j+1][i+1] = 0.0f;
      } else {
        image[j+1][i+1] = 100.0f;
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

int calc_nrows_from_rank(int ny, int rank, int size)
{
  int nrows;

  nrows = ny / (size/4);       /* integer division */
  if ((ny % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nrows += ny % size;  /* add remainder to last rank */
  }
  
  return nrows;
}

int calc_nclos_from_rank(int nx, int rank, int size)
{
  int nclos;

  nclos = nx / (size/4);       /* integer division */
  if ((nx % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      nclos += nx % size;  /* add remainder to last rank */
  }
  
  return nclos;
}