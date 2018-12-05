#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MIN(a,b) (((a)<(b))?(a):(b))

void stencil(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
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

  // Allocate the image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    stencil(nx, ny, image, tmp_image);
    stencil(nx, ny, tmp_image, image);
  }
  double toc = wtime();

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, image);
  free(image);
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  #pragma ivdep
  for (int y = 1; y < ny-1; y+=1024) {
  for (int x = 1; x < nx-1; x+=1024) {
  for (int j = y; j < MIN(y+1024,ny-1); ++j) {
    for (int i = x; i < MIN(x+1024,nx-1); ++i) {
      tmp_image[i+j*nx] = image[i+j*nx] * 3.0f/5.0f
        + image[i  +(j-1)*nx] * 0.5f/5.0f // TOP
        + image[i  +(j+1)*nx] * 0.5f/5.0f // BELOW
        + image[i-1+j*nx] * 0.5f/5.0f // LEFT
        + image[i+1+j*nx] * 0.5f/5.0f; // RIGHT
    }
  }
  }}

  #pragma ivdep
  for (int i = 1; i < nx-1; ++i) {
    // FIRST row
    tmp_image[i] = image[i] * 3.0f/5.0f
      + image[i+nx] * 0.5f/5.0f // BELOW
      + image[i-1] * 0.5f/5.0f // LEFT
      + image[i+1] * 0.5f/5.0f; // RIGHT

      // LAST row
    tmp_image[i+(ny-1)*nx] = image[i+(ny-1)*nx] * 3.0f/5.0f
      + image[i+(ny-2)*nx] * 0.5f/5.0f // TOP
      + image[i+(ny-1)*nx-1] * 0.5f/5.0f // LEFT
      + image[i+(ny-1)*nx+1] * 0.5f/5.0f; // RIGHT
  }

  #pragma ivdep
  for (int i = 1; i < ny-1; ++i) {
    // FIRST column
    tmp_image[i*nx] = image[i*nx] * 3.0f/5.0f
      + image[(i-1)*nx] * 0.5f/5.0f // TOP
      + image[(i+1)*nx] * 0.5f/5.0f // BELOW
      + image[i*nx+1] * 0.5f/5.0f; // RIGHT
    
     // LAST column
    tmp_image[(i+1)*nx-1] = image[(i+1)*nx-1] * 3.0f/5.0f
      + image[i*nx-1] * 0.5f/5.0f // TOP
      + image[(i+2)*nx-1] * 0.5f/5.0f // BELOW
      + image[(i+1)*nx-2] * 0.5f/5.0f; // LEFT
  }

  // TOP-LEFT corner
  tmp_image[0] = image[0] * 3.0f/5.0f
    + image[nx] * 0.5f/5.0f // BELOW
    + image[1] * 0.5f/5.0f; // RIGHT
  
  // TOP-RIGHT corner
  tmp_image[nx-1] = image[nx-1] * 3.0f/5.0f
    + image[2*nx - 1] * 0.5f/5.0f // BELOW
    + image[nx-2] * 0.5f/5.0f; // LEFT
  
  // BELOW-LEFT corner
  tmp_image[nx*(ny-1)] = image[nx*(ny-1)] * 3.0f/5.0f
    + image[nx*(ny-2)] * 0.5f/5.0f // TOP
    + image[nx*(ny-1) + 1] * 0.5f/5.0f; // RIGHT

  // BELOW-RIGHT corner
  tmp_image[nx*ny-1] = image[nx*ny-1] * 3.0f/5.0f
    + image[nx*(ny-1)-1] * 0.5f/5.0f // TOP
    + image[nx*ny-2] * 0.5f/5.0f; // LEFT
}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
 double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}