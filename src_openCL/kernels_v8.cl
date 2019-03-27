#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define blksz 16

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}

kernel void rebound(global t_speed* cells, 
                    global t_speed* tmp_cells, 
                    global int* obstacles,
                    local t_speed* Ablk,
                    int nx, int ny, float omega,
                    float density, float accel)
{  

  const float c_sq = 1.f / 3.f;
  const float w0 = 4.f / 9.f;
  const float w1 = 1.f / 9.f;
  const float w2 = 1.f / 36.f;
  float w3 = density * accel / 9.0;
  float w4 = density * accel / 36.0;

  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int gx = get_group_id(0);
  int gy = get_group_id(1);

  int xloc = get_local_size(0);
  int yloc = get_local_size(1);

  int nblock_nx = nx/blksz;
  int nblock_ny = ny/blksz;

  for (int k=lx; k<N; k+=xloc)
  {
    for (int k=lx; k<N; k+=xloc)
    {


    }
  }
      barrier(CLK_LOCAL_MEM_FENCE);
      int y_n = ((ly + b*blksz) + 1) % ny;
      int x_e = ((lx + b*blksz) + 1) % nx;
      int y_s = ((ly + b*blksz) == 0) ? ((ly + b*blksz) + ny - 1) : ((ly + b*blksz) - 1);
      int x_w = ((lx + b*blksz) == 0) ? ((lx + b*blksz) + nx - 1) : ((lx + b*blksz) - 1);

      Ablk[lx + ly*blksz].speeds[0] = cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[0];
      Ablk[lx + ly*blksz].speeds[1] = cells[x_w + (ly + b*blksz)*nx].speeds[1];
      Ablk[lx + ly*blksz].speeds[2] = cells[(lx + b*blksz) + y_s*nx].speeds[2];
      Ablk[lx + ly*blksz].speeds[3] = cells[x_e + (ly + b*blksz)*nx].speeds[3];
      Ablk[lx + ly*blksz].speeds[4] = cells[(lx + b*blksz) + y_n*nx].speeds[4];
      Ablk[lx + ly*blksz].speeds[5] = cells[x_w + y_s*nx].speeds[5];
      Ablk[lx + ly*blksz].speeds[6] = cells[x_e + y_s*nx].speeds[6];
      Ablk[lx + ly*blksz].speeds[7] = cells[x_e + y_n*nx].speeds[7];
      Ablk[lx + ly*blksz].speeds[8] = cells[x_w + y_n*nx].speeds[8];
      barrier(CLK_LOCAL_MEM_FENCE);

      if (obstacles[jj*nx + ii])
      {
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[1] = Ablk[lx + ly*blksz].speeds[3];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[2] = Ablk[lx + ly*blksz].speeds[4];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[3] = Ablk[lx + ly*blksz].speeds[1];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[4] = Ablk[lx + ly*blksz].speeds[2];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[5] = Ablk[lx + ly*blksz].speeds[7];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[6] = Ablk[lx + ly*blksz].speeds[8];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[7] = Ablk[lx + ly*blksz].speeds[5];
        cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[8] = Ablk[lx + ly*blksz].speeds[6];

      } else {
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += Ablk[lx + ly*blksz].[kk];
        }

        float u_x = (Ablk[lx + ly*blksz].[1] + Ablk[lx + ly*blksz].[5] + Ablk[lx + ly*blksz].[8] - (Ablk[lx + ly*blksz].[3] + Ablk[lx + ly*blksz].[6] + Ablk[lx + ly*blksz].[7])) / local_density;
        float u_y = (Ablk[lx + ly*blksz].[2] + Ablk[lx + ly*blksz].[5] + Ablk[lx + ly*blksz].[6] - (Ablk[lx + ly*blksz].[4] + Ablk[lx + ly*blksz].[7] + Ablk[lx + ly*blksz].[8])) / local_density;
        float u_sq = u_x * u_x + u_y * u_y;

        float u[NSPEEDS];
        u[1] = u_x;        /* east */
        u[2] = u_y;        /* north */
        u[3] = -u_x;       /* west */
        u[4] = -u_y;       /* south */
        u[5] = u_x + u_y;  /* north-east */
        u[6] = -u_x + u_y; /* north-west */
        u[7] = -u_x - u_y; /* south-west */
        u[8] = u_x - u_y;  /* south-east */

        float d_equ[NSPEEDS];
        d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[(lx + b*blksz) + (ly + b*blksz)*nx].speeds[kk] = Ablk[lx + ly*blksz].[kk] + omega * (d_equ[kk] - Ablk[lx + ly*blksz].[kk]);
        }
      }

}

kernel void av_velocity_kernel(global t_speed *cells,
                              global int *obstacles,
                              global float *d_partial_sums,
                              int nx, int ny)
{

  int ii = get_global_id(0);
  int jj = get_global_id(1);

      if (!obstacles[ii + jj * nx])
      {
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj * nx].speeds[kk];
        }

        float u_x = (cells[ii + jj * nx].speeds[1] + cells[ii + jj * nx].speeds[5] + cells[ii + jj * nx].speeds[8] - (cells[ii + jj * nx].speeds[3] + cells[ii + jj * nx].speeds[6] + cells[ii + jj * nx].speeds[7])) / local_density;
        float u_y = (cells[ii + jj * nx].speeds[2] + cells[ii + jj * nx].speeds[5] + cells[ii + jj * nx].speeds[6] - (cells[ii + jj * nx].speeds[4] + cells[ii + jj * nx].speeds[7] + cells[ii + jj * nx].speeds[8])) / local_density;
        
        d_partial_sums[ii + jj * nx] = sqrt((u_x * u_x) + (u_y * u_y));
      }else {
        d_partial_sums[ii + jj * nx] = 0.f;
      }
}