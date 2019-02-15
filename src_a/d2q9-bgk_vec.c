/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define squared(x) ((x)*(x))
#define u_u(a,b,c,d,e,f,g,h) ((a+b+c-d-e-f)*g/h)
#define c_c(a,b,d,e,f) ((a) * b +  (d) * (squared(e) + f))

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
float propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
void collision(float* av_vels, int tt, const t_param params, t_speed* cells, int* obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);
void calculation(const float omg, float* transf, float *tot_u);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  #pragma ivdep
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = (tt % 2 == 0) ? (timestep(params, cells, tmp_cells, obstacles)) : (timestep(params, tmp_cells, cells, obstacles));
    
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  if( params.maxIters % 2 == 0 ){
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  } else {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, tmp_cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, tmp_cells, obstacles, av_vels);
    finalise(&params, &tmp_cells, &cells, &obstacles, &av_vels);
  }
  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles)
{
  accelerate_flow(params, cells, obstacles); 
  return propagate(params, cells, tmp_cells, obstacles);
}

void accelerate_flow(const t_param params, t_speed* restrict cells, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */

  #pragma omp parallel for simd
  #pragma ivdep
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + (params.ny - 2)*params.nx]
        && (cells[ii + (params.ny - 2)*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + (params.ny - 2)*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + (params.ny - 2)*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + (params.ny - 2)*params.nx].speeds[1] += w1;
      cells[ii + (params.ny - 2)*params.nx].speeds[5] += w2;
      cells[ii + (params.ny - 2)*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + (params.ny - 2)*params.nx].speeds[3] -= w1;
      cells[ii + (params.ny - 2)*params.nx].speeds[6] -= w2;
      cells[ii + (params.ny - 2)*params.nx].speeds[7] -= w2;
    }
  }
  // return EXIT_SUCCESS;
}

float propagate(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* obstacles)
{

  int tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  #pragma omp parallel for reduction(+:tot_u, tot_cells)
  #pragma ivdep
  for (int jj = 0; jj < params.ny; jj++)
  {
    #pragma ivdep
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      float transf[NSPEEDS];
      transf[0] = cells[ii + jj*params.nx].speeds[0]; /* central cell, no movement */
      transf[1] = cells[x_w + jj*params.nx].speeds[1]; /* east */
      transf[2] = cells[ii + y_s*params.nx].speeds[2]; /* north */
      transf[3] = cells[x_e + jj*params.nx].speeds[3]; /* west */
      transf[4] = cells[ii + y_n*params.nx].speeds[4]; /* south */
      transf[5] = cells[x_w + y_s*params.nx].speeds[5]; /* north-east */
      transf[6] = cells[x_e + y_s*params.nx].speeds[6]; /* north-west */
      transf[7] = cells[x_e + y_n*params.nx].speeds[7]; /* south-west */
      transf[8] = cells[x_w + y_n*params.nx].speeds[8]; /* south-east */

      (obstacles[ii + jj*params.nx]) ? (transf): calculation(params.omega, &transf[0], &tot_u);
      tot_cells += (obstacles[ii + jj*params.nx]) ? (0) : (1);

      tmp_cells[ii + jj*params.nx].speeds[0] = (obstacles[ii + jj*params.nx]) ? (transf[0]) : (transf[0]);
      tmp_cells[ii + jj*params.nx].speeds[1] = (obstacles[ii + jj*params.nx]) ? (transf[3]) : (transf[1]);
      tmp_cells[ii + jj*params.nx].speeds[2] = (obstacles[ii + jj*params.nx]) ? (transf[4]) : (transf[2]);
      tmp_cells[ii + jj*params.nx].speeds[3] = (obstacles[ii + jj*params.nx]) ? (transf[1]) : (transf[3]);
      tmp_cells[ii + jj*params.nx].speeds[4] = (obstacles[ii + jj*params.nx]) ? (transf[2]) : (transf[4]);
      tmp_cells[ii + jj*params.nx].speeds[5] = (obstacles[ii + jj*params.nx]) ? (transf[7]) : (transf[5]);
      tmp_cells[ii + jj*params.nx].speeds[6] = (obstacles[ii + jj*params.nx]) ? (transf[8]) : (transf[6]);
      tmp_cells[ii + jj*params.nx].speeds[7] = (obstacles[ii + jj*params.nx]) ? (transf[5]) : (transf[7]);
      tmp_cells[ii + jj*params.nx].speeds[8] = (obstacles[ii + jj*params.nx]) ? (transf[6]) : (transf[8]);
      
    }
  }
  return tot_u / (float)tot_cells;
}

#pragma omp declare simd inbranch
void calculation(const float omega, float* transf, float *tot_u)
{
  const float w0 = omega * 2.f / 9.f ;  /* weighting factor */
  const float w1 = omega * 1.f / 18.f ;  /* weighting factor */
  const float w2 = omega * 1.f / 72.f ; /* weighting factor */
  const float omg = 1 - omega;
  float u_x;
  float u_y;
  float u_sq;
  
  float local_density = 0.f;

  for (int kk = 0; kk < NSPEEDS; kk++)
  {
    local_density += transf[kk];
  }

  u_x = u_u(transf[1], transf[5], transf[8],
    transf[3], transf[6], transf[7], 3.f, local_density);
  u_y = u_u(transf[2], transf[5], transf[6],
    transf[4], transf[7], transf[8], 3.f, local_density);

  u_sq = 1.f - (squared(u_x) + squared(u_y)) / 3.f;

  transf[0] = c_c(omg, transf[0], w0 * local_density, 1.f, u_sq);
  transf[1] = c_c(omg, transf[1], w1 * local_density, 1.f + u_x, u_sq);
  transf[2] = c_c(omg, transf[2], w1 * local_density, 1.f + u_y, u_sq);
  transf[3] = c_c(omg, transf[3], w1 * local_density, 1.f - u_x, u_sq);
  transf[4] = c_c(omg, transf[4], w1 * local_density, 1.f - u_y, u_sq);
  transf[5] = c_c(omg, transf[5], w2 * local_density, 1.f + u_x + u_y, u_sq);
  transf[6] = c_c(omg, transf[6], w2 * local_density, 1.f - u_x + u_y, u_sq);
  transf[7] = c_c(omg, transf[7], w2 * local_density, 1.f - u_x - u_y, u_sq);
  transf[8] = c_c(omg, transf[8], w2 * local_density, 1.f + u_x - u_y, u_sq);

  local_density = 0.f;

  for (int kk = 0; kk < NSPEEDS; kk++)
  {
    local_density += transf[kk];
  }

  u_x = u_u(transf[1], transf[5], transf[8],
    transf[3], transf[6], transf[7], 1.f, local_density);
  u_y = u_u(transf[2], transf[5], transf[6],
    transf[4], transf[7], transf[8], 1.f, local_density);

  *tot_u += sqrtf(squared(u_x) + squared(u_y));
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        float u_x = u_u(cells[ii + jj*params.nx].speeds[1], cells[ii + jj*params.nx].speeds[5], cells[ii + jj*params.nx].speeds[8],
        cells[ii + jj*params.nx].speeds[3], cells[ii + jj*params.nx].speeds[6], cells[ii + jj*params.nx].speeds[7], 1.f, local_density);
        float u_y = u_u(cells[ii + jj*params.nx].speeds[2], cells[ii + jj*params.nx].speeds[5], cells[ii + jj*params.nx].speeds[6],
        cells[ii + jj*params.nx].speeds[4], cells[ii + jj*params.nx].speeds[7], cells[ii + jj*params.nx].speeds[8], 1.f, local_density);

        tot_u += sqrtf(squared(u_x) + squared(u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
