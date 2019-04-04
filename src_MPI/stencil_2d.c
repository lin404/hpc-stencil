
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MIN(a,b) (((a)<(b))?(a):(b))

void stencil(const int nx, const int ny, float **  image, float ** tmp_image);
void init_image(const int nx, const int ny, float**  image);
void output_image(const char * file_name, const int nx, const int ny, float **image);
double wtime(void);
float** matrixAllocate(int x, int y);
void matrixFree(float** matrix, int size);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int ny = atoi(argv[1]);
  int nx = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float **image = matrixAllocate(nx, ny);
  float **tmp_image = matrixAllocate(nx, ny);

  // Set the input image
  init_image(nx, ny, image);

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

  matrixFree(image, ny);
  matrixFree(tmp_image, ny);
}

void stencil(const int nx, const int ny, float ** restrict image, float ** restrict tmp_image) {

#pragma ivdep
for (int y = 1; y < ny-1; y+=1024) {
#pragma ivdep
for (int x = 1; x < nx-1; x+=1024) {
  #pragma ivdep
  for (int j = y; j < MIN(y+1024,ny-1); ++j) {
    for (int i = x; i < MIN(x+1024,nx-1); ++i) {
      tmp_image[j][i] = image[j][i] * 3.0f/5.0f
        + image[j-1][i] * 0.5f/5.0f // TOP
        + image[j+1][i] * 0.5f/5.0f // BELOW
        + image[j][i-1] * 0.5f/5.0f // LEFT
        + image[j][i+1] * 0.5f/5.0f; // RIGHT
    }
  }
    }}

#pragma ivdep
  for (int i = 1; i < nx-1; ++i) {
    // FIRST row
    tmp_image[0][i] = image[0][i] * 3.0f/5.0f
      + image[1][i] * 0.5f/5.0f // BELOW
      + image[0][i-1] * 0.5f/5.0f // LEFT
      + image[0][i+1] * 0.5f/5.0f; // RIGHT
    // LAST row
    tmp_image[ny-1][i] = image[ny-1][i] * 3.0f/5.0f
      + image[ny-2][i] * 0.5f/5.0f // TOP
      + image[ny-1][i-1] * 0.5f/5.0f // LEFT
      + image[ny-1][i+1] * 0.5f/5.0f; // RIGHT
  }

#pragma ivdep
  for (int i = 1; i < ny-1; ++i) {
    // FIRST column
    tmp_image[i][0] = image[i][0] * 3.0f/5.0f
      + image[i-1][0] * 0.5f/5.0f // TOP
      + image[i+1][0] * 0.5f/5.0f // BELOW
      + image[i][1] * 0.5f/5.0f; // RIGHT
    // LAST column
    tmp_image[i][nx-1] = image[i][nx-1] * 3.0f/5.0f
      + image[i-1][nx-1] * 0.5f/5.0f // TOP
      + image[i+1][nx-1] * 0.5f/5.0f // BELOW
      + image[i][nx-2] * 0.5f/5.0f; // LEFT
  }
  // TOP-LEFT corner
  tmp_image[0][0] = image[0][0] * 3.0f/5.0f
    + image[1][0] * 0.5f/5.0f // BELOW
    + image[0][1] * 0.5f/5.0f; // RIGHT
  
  // TOP-RIGHT corner
  tmp_image[0][nx-1] = image[0][nx-1] * 3.0f/5.0f
    + image[1][nx-1] * 0.5f/5.0f // BELOW
    + image[0][nx-2] * 0.5f/5.0f; // LEFT
  
  // BELOW-LEFT corner
  tmp_image[ny-1][0] = image[ny-1][0] * 3.0f/5.0f
    + image[ny-2][0] * 0.5f/5.0f // TOP
    + image[ny-1][1] * 0.5f/5.0f; // RIGHT
  
  // BELOW-RIGHT corner
  tmp_image[ny-1][nx-1] = image[ny-1][nx-1] * 3.0f/5.0f
    + image[ny-2][nx-1] * 0.5f/5.0f // TOP
    + image[ny-1][nx-2] * 0.5f/5.0f; // LEFT
}

// Create the input image
void init_image(const int nx, const int ny, float ** image) {
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (((i/(nx/8))%2) == ((j/(nx/8))%2)) {
        image[j][i] = 0.0;
      } else {
        image[j][i] = 100.0;
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float **image) {

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
      if (image[j][i] > maximum)
        maximum = image[j][i];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j][i]/maximum), fp);
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

float** matrixAllocate(int x, int y) {
 float** matrix = malloc(y*sizeof(float*));

 for (int j=0; j<y; j++) {
  matrix[j] = malloc(x*sizeof(float));
 }

 return matrix;
}
void matrixFree(float** matrix, int size) {
  for(int i =0; i<size; i++) {
    free(matrix[i]);
  }

  free(matrix);
}