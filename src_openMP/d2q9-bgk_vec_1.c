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

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define squared(x) ((x) * (x))
#define u_u(a, b, c, d, e, f, g, h) ((a + b + c - d - e - f) * g / h)
#define c_c(a, b, d, e, f) ((a)*b + (d) * (squared(e) + f))
#define boundary(a, b) ((a == 0) ? (a + b) : (a - 1))

/* struct to hold the parameter values */
typedef struct
{
  int nx;           /* no. of cells in x-direction */
  int ny;           /* no. of cells in y-direction */
  int maxIters;     /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float *speeds_0;
  float *speeds_1;
  float *speeds_2;
  float *speeds_3;
  float *speeds_4;
  float *speeds_5;
  float *speeds_6;
  float *speeds_7;
  float *speeds_8;
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed *cells_ptr, t_speed *tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed cells, t_speed tmp_cells, int *obstacles);
void accelerate_flow(const t_param params, t_speed cells, int *obstacles);
float propagate(const t_param params, t_speed cells, t_speed tmp_cells, int *obstacles);
int write_values(const t_param params, t_speed cells, int *obstacles, float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed *cells_ptr, t_speed *tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed cells, int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed cells, int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
  char *paramfile = NULL;    /* name of the input parameter file */
  char *obstaclefile = NULL; /* name of a the input obstacle file */
  t_param params;            /* struct to hold parameter values */
  t_speed cells;             /* grid containing fluid densities */
  t_speed tmp_cells;         /* scratch space */
  int *obstacles = NULL;     /* grid indicating which cells are blocked */
  float *av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;     /* structure to hold elapsed time */
  struct rusage ru;          /* structure to hold CPU time--system and user */
  double tic, toc;           /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;             /* floating point number to record elapsed user CPU time */
  double systim;             /* floating point number to record elapsed system CPU time */

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

// #pragma omp for schedule(static)
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

  if (params.maxIters % 2 == 0)
  {
    /* write final values and free memory */
    printf("==done==\n");
    // printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    // printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    // printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    // printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    // write_values(params, cells, obstacles, av_vels);
    // finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  }
  else
  {
    printf("==done==\n");
    // printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, tmp_cells, obstacles));
    // printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    // printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    // printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    // write_values(params, tmp_cells, obstacles, av_vels);
    // finalise(&params, &tmp_cells, &cells, &obstacles, &av_vels);
  }
  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed cells, t_speed tmp_cells, int *obstacles)
{
  accelerate_flow(params, cells, obstacles);
  // return propagate(params, cells, tmp_cells, obstacles);
  return 0.f;
}

void accelerate_flow(const t_param params, t_speed cells, int *obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  for (int ii = 0; ii < params.nx; ii++)
  {
    printf("==done==\n %f", cells.speeds_3[0]);
    /* if the cell is not occupied and
    ** we don't send a negative density */
    // if (!obstacles[ii + (params.ny - 2) * params.nx] && (cells.speeds_3[ii + (params.ny - 2) * params.nx] - w1) > 0.f && (cells.speeds_6[ii + (params.ny - 2) * params.nx] - w2) > 0.f && (cells.speeds_7[ii + (params.ny - 2) * params.nx] - w2) > 0.f)
    // {
    //   /* increase 'east-side' densities */
    //   cells.speeds_1[ii + (params.ny - 2) * params.nx] += w1;
    //   cells.speeds_5[ii + (params.ny - 2) * params.nx] += w2;
    //   cells.speeds_8[ii + (params.ny - 2) * params.nx] += w2;
    //   /* decrease 'west-side' densities */
    //   cells.speeds_3[ii + (params.ny - 2) * params.nx] -= w1;
    //   cells.speeds_6[ii + (params.ny - 2) * params.nx] -= w2;
    //   cells.speeds_7[ii + (params.ny - 2) * params.nx] -= w2;
    // }
  }
}

float propagate(const t_param params, t_speed cells, t_speed tmp_cells, int *obstacles)
{

  const float w0 = params.omega * 2.f / 9.f;  /* weighting factor */
  const float w1 = params.omega * 1.f / 18.f; /* weighting factor */
  const float w2 = params.omega * 1.f / 72.f; /* weighting factor */
  const float omg = 1 - params.omega;

  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */
  int tot_cells = 0; /* no. of cells used in calculation */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = (jj + 1) & (params.ny - 1);
      int x_e = (ii + 1) & (params.nx - 1);
      int y_s = boundary(jj, params.ny - 1);
      int x_w = boundary(ii, params.nx - 1);
      ;

      if (obstacles[ii + jj * params.nx])
      {

        tmp_cells.speeds_0[ii + jj * params.nx] = cells.speeds_0[ii + jj * params.nx];   /* central cell, no movement */
        tmp_cells.speeds_1[ii + jj * params.nx] = cells.speeds_3[x_e + jj * params.nx];  /* east */
        tmp_cells.speeds_2[ii + jj * params.nx] = cells.speeds_4[ii + y_n * params.nx];  /* north */
        tmp_cells.speeds_3[ii + jj * params.nx] = cells.speeds_1[x_w + jj * params.nx];  /* west */
        tmp_cells.speeds_4[ii + jj * params.nx] = cells.speeds_2[ii + y_s * params.nx];  /* south */
        tmp_cells.speeds_5[ii + jj * params.nx] = cells.speeds_7[x_e + y_n * params.nx]; /* north-east */
        tmp_cells.speeds_6[ii + jj * params.nx] = cells.speeds_8[x_w + y_n * params.nx]; /* north-west */
        tmp_cells.speeds_7[ii + jj * params.nx] = cells.speeds_5[x_w + y_s * params.nx]; /* south-west */
        tmp_cells.speeds_8[ii + jj * params.nx] = cells.speeds_6[x_e + y_s * params.nx]; /* south-east */
      }
      else
      {

        float local_density = cells.speeds_0[ii + jj * params.nx] + cells.speeds_1[x_w + jj * params.nx] + cells.speeds_2[ii + y_s * params.nx] + cells.speeds_3[x_e + jj * params.nx] + cells.speeds_4[ii + y_n * params.nx] + cells.speeds_5[x_w + y_s * params.nx] + cells.speeds_6[x_e + y_s * params.nx] + cells.speeds_7[x_e + y_n * params.nx] + cells.speeds_8[x_w + y_n * params.nx];

        float u_x = u_u(cells.speeds_1[x_w + jj * params.nx], cells.speeds_5[x_w + y_s * params.nx],
                        cells.speeds_8[x_w + y_n * params.nx], cells.speeds_3[x_e + jj * params.nx],
                        cells.speeds_6[x_e + y_s * params.nx], cells.speeds_7[x_e + y_n * params.nx], 3.f, local_density);
        float u_y = u_u(cells.speeds_2[ii + y_s * params.nx], cells.speeds_5[x_w + y_s * params.nx],
                        cells.speeds_6[x_e + y_s * params.nx], cells.speeds_4[ii + y_n * params.nx],
                        cells.speeds_7[x_e + y_n * params.nx], cells.speeds_8[x_w + y_n * params.nx], 3.f, local_density);

        float u_sq = 1.f - (squared(u_x) + squared(u_y)) / 3.f;

        tmp_cells.speeds_0[ii + jj * params.nx] = c_c(omg, cells.speeds_0[ii + jj * params.nx], w0 * local_density, 1.f, u_sq);
        tmp_cells.speeds_1[ii + jj * params.nx] = c_c(omg, cells.speeds_1[x_w + jj * params.nx], w1 * local_density, 1.f + u_x, u_sq);
        tmp_cells.speeds_2[ii + jj * params.nx] = c_c(omg, cells.speeds_2[ii + y_s * params.nx], w1 * local_density, 1.f + u_y, u_sq);
        tmp_cells.speeds_3[ii + jj * params.nx] = c_c(omg, cells.speeds_3[x_e + jj * params.nx], w1 * local_density, 1.f - u_x, u_sq);
        tmp_cells.speeds_4[ii + jj * params.nx] = c_c(omg, cells.speeds_4[ii + y_n * params.nx], w1 * local_density, 1.f - u_y, u_sq);
        tmp_cells.speeds_5[ii + jj * params.nx] = c_c(omg, cells.speeds_5[x_w + y_s * params.nx], w2 * local_density, 1.f + u_x + u_y, u_sq);
        tmp_cells.speeds_6[ii + jj * params.nx] = c_c(omg, cells.speeds_6[x_e + y_s * params.nx], w2 * local_density, 1.f - u_x + u_y, u_sq);
        tmp_cells.speeds_7[ii + jj * params.nx] = c_c(omg, cells.speeds_7[x_e + y_n * params.nx], w2 * local_density, 1.f - u_x - u_y, u_sq);
        tmp_cells.speeds_8[ii + jj * params.nx] = c_c(omg, cells.speeds_8[x_w + y_n * params.nx], w2 * local_density, 1.f + u_x - u_y, u_sq);

        float local_density_1 = tmp_cells.speeds_0[ii + jj * params.nx] + tmp_cells.speeds_1[ii + jj * params.nx] + tmp_cells.speeds_2[ii + jj * params.nx] + tmp_cells.speeds_3[ii + jj * params.nx] + tmp_cells.speeds_4[ii + jj * params.nx] + tmp_cells.speeds_5[ii + jj * params.nx] + tmp_cells.speeds_6[ii + jj * params.nx] + tmp_cells.speeds_7[ii + jj * params.nx] + tmp_cells.speeds_8[ii + jj * params.nx];

        float u_x_1 = u_u(tmp_cells.speeds_1[ii + jj * params.nx], tmp_cells.speeds_5[ii + jj * params.nx],
                          tmp_cells.speeds_8[ii + jj * params.nx], tmp_cells.speeds_3[ii + jj * params.nx],
                          tmp_cells.speeds_6[ii + jj * params.nx], tmp_cells.speeds_7[ii + jj * params.nx], 1.f, local_density_1);
        float u_y_1 = u_u(tmp_cells.speeds_2[ii + jj * params.nx], tmp_cells.speeds_5[ii + jj * params.nx],
                          tmp_cells.speeds_6[ii + jj * params.nx], tmp_cells.speeds_4[ii + jj * params.nx],
                          tmp_cells.speeds_7[ii + jj * params.nx], tmp_cells.speeds_8[ii + jj * params.nx], 1.f, local_density_1);

        tot_u += sqrtf(squared(u_x_1) + squared(u_y_1));
        tot_cells++;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_speed cells, int *obstacles)
{
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u;       /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj * params.nx])
      {
        /* local density total */
        float local_density = 0.f;
        local_density += cells.speeds_0[ii + jj * params.nx];
        local_density += cells.speeds_1[ii + jj * params.nx];
        local_density += cells.speeds_2[ii + jj * params.nx];
        local_density += cells.speeds_3[ii + jj * params.nx];
        local_density += cells.speeds_4[ii + jj * params.nx];
        local_density += cells.speeds_5[ii + jj * params.nx];
        local_density += cells.speeds_6[ii + jj * params.nx];
        local_density += cells.speeds_7[ii + jj * params.nx];
        local_density += cells.speeds_8[ii + jj * params.nx];

        float u_x = u_u(cells.speeds_1[ii + jj * params.nx], cells.speeds_5[ii + jj * params.nx], cells.speeds_8[ii + jj * params.nx],
                        cells.speeds_3[ii + jj * params.nx], cells.speeds_6[ii + jj * params.nx], cells.speeds_7[ii + jj * params.nx], 1.f, local_density);
        float u_y = u_u(cells.speeds_2[ii + jj * params.nx], cells.speeds_5[ii + jj * params.nx], cells.speeds_6[ii + jj * params.nx],
                        cells.speeds_4[ii + jj * params.nx], cells.speeds_7[ii + jj * params.nx], cells.speeds_8[ii + jj * params.nx], 1.f, local_density);

        tot_u += sqrtf(squared(u_x) + squared(u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed *cells_ptr, t_speed *tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr)
{
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1)
    die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1)
    die("could not read param file: omega", __LINE__, __FILE__);

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
  cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  cells_ptr->speeds_0 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_1 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_2 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_3 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_4 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_5 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_6 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_7 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  cells_ptr->speeds_8 = (float *)malloc(sizeof(float) * (params->ny * params->nx));

  // if (cells_ptr == NULL)
  //   die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  tmp_cells_ptr->speeds_0 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_1 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_2 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_3 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_4 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_5 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_6 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_7 = (float *)malloc(sizeof(float) * (params->ny * params->nx));
  tmp_cells_ptr->speeds_8 = (float *)malloc(sizeof(float) * (params->ny * params->nx));

  // if (*tmp_cells_ptr == NULL)
  //   die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      // (*cells_ptr).speeds_0[ii + jj * params->nx] = w0;
      // /* axis directions */
      // (*cells_ptr).speeds_1[ii + jj * params->nx] = w1;
      // (*cells_ptr).speeds_2[ii + jj * params->nx] = w1;
      // (*cells_ptr).speeds_3[ii + jj * params->nx] = w1;
      // (*cells_ptr).speeds_4[ii + jj * params->nx] = w1;
      // /* diagonals */
      // (*cells_ptr).speeds_5[ii + jj * params->nx] = w2;
      // (*cells_ptr).speeds_6[ii + jj * params->nx] = w2;
      // (*cells_ptr).speeds_7[ii + jj * params->nx] = w2;
      // (*cells_ptr).speeds_8[ii + jj * params->nx] = w2;
      /* centre */
      cells_ptr->speeds_0[ii + jj * params->nx] = w0;
      /* axis directions */
      cells_ptr->speeds_1[ii + jj * params->nx] = w1;
      cells_ptr->speeds_2[ii + jj * params->nx] = w1;
      cells_ptr->speeds_3[ii + jj * params->nx] = w1;
      cells_ptr->speeds_4[ii + jj * params->nx] = w1;
      /* diagonals */
      cells_ptr->speeds_5[ii + jj * params->nx] = w2;
      cells_ptr->speeds_6[ii + jj * params->nx] = w2;
      cells_ptr->speeds_7[ii + jj * params->nx] = w2;
      cells_ptr->speeds_8[ii + jj * params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
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
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float *)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_speed *cells_ptr, t_speed *tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  // free(*cells_ptr);
  // *cells_ptr = NULL;

  // free(*tmp_cells_ptr);
  // *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed cells, int *obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed cells)
{
  float total = 0.f; /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells.speeds_0[ii + jj * params.nx];
      total += cells.speeds_1[ii + jj * params.nx];
      total += cells.speeds_2[ii + jj * params.nx];
      total += cells.speeds_3[ii + jj * params.nx];
      total += cells.speeds_4[ii + jj * params.nx];
      total += cells.speeds_5[ii + jj * params.nx];
      total += cells.speeds_6[ii + jj * params.nx];
      total += cells.speeds_7[ii + jj * params.nx];
      total += cells.speeds_8[ii + jj * params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed cells, int *obstacles, float *av_vels)
{
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;          /* per grid cell sum of densities */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u;                      /* norm--root of summed squares--of u_x and u_y */

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
      if (obstacles[ii + jj * params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;
        local_density += cells.speeds_0[ii + jj * params.nx];
        local_density += cells.speeds_1[ii + jj * params.nx];
        local_density += cells.speeds_2[ii + jj * params.nx];
        local_density += cells.speeds_3[ii + jj * params.nx];
        local_density += cells.speeds_4[ii + jj * params.nx];
        local_density += cells.speeds_5[ii + jj * params.nx];
        local_density += cells.speeds_6[ii + jj * params.nx];
        local_density += cells.speeds_7[ii + jj * params.nx];
        local_density += cells.speeds_8[ii + jj * params.nx];

        /* compute x velocity component */
        u_x = (cells.speeds_1[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx] - (cells.speeds_3[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx])) / local_density;
        /* compute y velocity component */
        u_y = (cells.speeds_2[ii + jj * params.nx] + cells.speeds_5[ii + jj * params.nx] + cells.speeds_6[ii + jj * params.nx] - (cells.speeds_4[ii + jj * params.nx] + cells.speeds_7[ii + jj * params.nx] + cells.speeds_8[ii + jj * params.nx])) / local_density;
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

void die(const char *message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}