#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define blksize 16

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void rebound(global t_speed* cells, 
                    global t_speed* tmp_cells, 
                    global int* obstacles,
                    global float *d_partial_sums,
                    global float *av_vels,
                    int nt, int nx, int ny, int tot_cells,
                    float omega, float density, float accel)
{  

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  float w3 = density * accel / 9.0;
  float w4 = density * accel / 36.0;

  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int tt;
  
  
  for(tt=0; tt<nt; tt++){

    barrier(CLK_GLOBAL_MEM_FENCE);

if(jj == ny-2){
      if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w3) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w4) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w4) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w3;
    cells[ii + jj* nx].speeds[5] += w4;
    cells[ii + jj* nx].speeds[8] += w4;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w3;
    cells[ii + jj* nx].speeds[6] -= w4;
    cells[ii + jj* nx].speeds[7] -= w4;
  }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (obstacles[jj*nx + ii])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[ii + jj*nx].speeds[1] = tmp_cells[ii + jj*nx].speeds[3];
    cells[ii + jj*nx].speeds[2] = tmp_cells[ii + jj*nx].speeds[4];
    cells[ii + jj*nx].speeds[3] = tmp_cells[ii + jj*nx].speeds[1];
    cells[ii + jj*nx].speeds[4] = tmp_cells[ii + jj*nx].speeds[2];
    cells[ii + jj*nx].speeds[5] = tmp_cells[ii + jj*nx].speeds[7];
    cells[ii + jj*nx].speeds[6] = tmp_cells[ii + jj*nx].speeds[8];
    cells[ii + jj*nx].speeds[7] = tmp_cells[ii + jj*nx].speeds[5];
    cells[ii + jj*nx].speeds[8] = tmp_cells[ii + jj*nx].speeds[6];

    d_partial_sums[ii + jj * nx] = 0.f;

  } else {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[ii + jj * nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (tmp_cells[ii + jj * nx].speeds[1] + tmp_cells[ii + jj * nx].speeds[5] + tmp_cells[ii + jj * nx].speeds[8] - (tmp_cells[ii + jj * nx].speeds[3] + tmp_cells[ii + jj * nx].speeds[6] + tmp_cells[ii + jj * nx].speeds[7])) / local_density;
        /* compute y velocity component */
        float u_y = (tmp_cells[ii + jj * nx].speeds[2] + tmp_cells[ii + jj * nx].speeds[5] + tmp_cells[ii + jj * nx].speeds[6] - (tmp_cells[ii + jj * nx].speeds[4] + tmp_cells[ii + jj * nx].speeds[7] + tmp_cells[ii + jj * nx].speeds[8])) / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] = u_x;        /* east */
        u[2] = u_y;        /* north */
        u[3] = -u_x;       /* west */
        u[4] = -u_y;       /* south */
        u[5] = u_x + u_y;  /* north-east */
        u[6] = -u_x + u_y; /* north-west */
        u[7] = -u_x - u_y; /* south-west */
        u[8] = u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj * nx].speeds[kk] = tmp_cells[ii + jj * nx].speeds[kk] + omega * (d_equ[kk] - tmp_cells[ii + jj * nx].speeds[kk]);
        }

        local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj * nx].speeds[kk];
        }

        u_x = (cells[ii + jj * nx].speeds[1] + cells[ii + jj * nx].speeds[5] + cells[ii + jj * nx].speeds[8] - (cells[ii + jj * nx].speeds[3] + cells[ii + jj * nx].speeds[6] + cells[ii + jj * nx].speeds[7])) / local_density;
        u_y = (cells[ii + jj * nx].speeds[2] + cells[ii + jj * nx].speeds[5] + cells[ii + jj * nx].speeds[6] - (cells[ii + jj * nx].speeds[4] + cells[ii + jj * nx].speeds[7] + cells[ii + jj * nx].speeds[8])) / local_density;
        
        d_partial_sums[ii + jj * nx] = sqrt((u_x * u_x) + (u_y * u_y));
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  
  int i;

  if(ii==0 && jj==0){
    float sum = 0.f;
    for(i=0; i<nx*ny; i++){
      sum += d_partial_sums[i];
    }
    av_vels[tt] = sum / (float) tot_cells;
  }
  }
}