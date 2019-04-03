#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void rebound(global t_speed* cells, 
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    global float *global_sums,
                    int nx, int ny,
                    float omega,
                    float density, float accel)
{  

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float w3 = density * accel / 9.0;
  float w4 = density * accel / 36.0;

  int ii = get_global_id(0);
  int jj = get_global_id(1);

  int local_ii = get_local_id(0);
  int local_jj = get_local_id(1);   

  int group_ii = get_group_id(0);
  int group_jj = get_group_id(1);

  int work_items_x  = get_local_size(0);
  int work_items_y  = get_local_size(1);

  if(jj==ny-2){
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

  float speeds[NSPEEDS];
  speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */

        float local_density = 0.f;
        for(int kk=0; kk<NSPEEDS; kk++){
          local_density += speeds[kk];
        }

        float u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
        float u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;

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

        tmp_cells[ii + jj*nx].speeds[0] = (obstacles[jj*nx + ii]) ? (speeds[0]) : (speeds[0] + omega * (d_equ[0] - speeds[0]));
        tmp_cells[ii + jj*nx].speeds[1] = (obstacles[jj*nx + ii]) ? (speeds[3]) : (speeds[1] + omega * (d_equ[1] - speeds[1]));
        tmp_cells[ii + jj*nx].speeds[2] = (obstacles[jj*nx + ii]) ? (speeds[4]) : (speeds[2] + omega * (d_equ[2] - speeds[2]));
        tmp_cells[ii + jj*nx].speeds[3] = (obstacles[jj*nx + ii]) ? (speeds[1]) : (speeds[3] + omega * (d_equ[3] - speeds[3]));
        tmp_cells[ii + jj*nx].speeds[4] = (obstacles[jj*nx + ii]) ? (speeds[2]) : (speeds[4] + omega * (d_equ[4] - speeds[4]));
        tmp_cells[ii + jj*nx].speeds[5] = (obstacles[jj*nx + ii]) ? (speeds[7]) : (speeds[5] + omega * (d_equ[5] - speeds[5]));
        tmp_cells[ii + jj*nx].speeds[6] = (obstacles[jj*nx + ii]) ? (speeds[8]) : (speeds[6] + omega * (d_equ[6] - speeds[6]));
        tmp_cells[ii + jj*nx].speeds[7] = (obstacles[jj*nx + ii]) ? (speeds[5]) : (speeds[7] + omega * (d_equ[7] - speeds[7]));
        tmp_cells[ii + jj*nx].speeds[8] = (obstacles[jj*nx + ii]) ? (speeds[6]) : (speeds[8] + omega * (d_equ[8] - speeds[8]));
      
      
       float local_density_tmp = 0.f;

        local_density_tmp += tmp_cells[ii + jj*nx].speeds[0]
                      + tmp_cells[ii + jj*nx].speeds[1]
                      + tmp_cells[ii + jj*nx].speeds[2]
                      + tmp_cells[ii + jj*nx].speeds[3]
                      + tmp_cells[ii + jj*nx].speeds[4]
                      + tmp_cells[ii + jj*nx].speeds[5]
                      + tmp_cells[ii + jj*nx].speeds[6]
                      + tmp_cells[ii + jj*nx].speeds[7]
                      + tmp_cells[ii + jj*nx].speeds[8];

        float u_x_tmp = (tmp_cells[ii + jj*nx].speeds[1] + tmp_cells[ii + jj*nx].speeds[5] + tmp_cells[ii + jj*nx].speeds[8] - (tmp_cells[ii + jj*nx].speeds[3] + tmp_cells[ii + jj*nx].speeds[6] + tmp_cells[ii + jj*nx].speeds[7])) / local_density_tmp;
        float u_y_tmp = (tmp_cells[ii + jj*nx].speeds[2] + tmp_cells[ii + jj*nx].speeds[5] + tmp_cells[ii + jj*nx].speeds[6] - (tmp_cells[ii + jj*nx].speeds[4] + tmp_cells[ii + jj*nx].speeds[7] + tmp_cells[ii + jj*nx].speeds[8])) / local_density_tmp;
        
        global_sums[ii + jj*nx] = (obstacles[jj*nx + ii]) ? (0.f) : (sqrt((u_x_tmp * u_x_tmp) + (u_y_tmp * u_y_tmp)));

}

