#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define blksize 16

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
                    int nx, int ny, float omega,
                    float density, float accel,
                    local t_speed* block)
{  

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  
  const unsigned int idx = get_global_id(0);
  const unsigned int idy = get_global_id(1);
  
  const unsigned int id = idy * nx + idx;
  const unsigned int id_b = (get_local_id(1) + 1) * (get_local_size(0) + 2) + get_local_id(0) + 1;
  
  int kk;
  
  for(kk=0; kk < NSPEEDS; kk++){
    block[id_b].speeds[kk] = cells[id].speeds[kk];
  }

  const unsigned int block_r = (get_group_id(0) + 1) % get_num_groups(0);
  const unsigned int block_l = (get_group_id(0) == 0) ? get_num_groups(0) - 1 : get_group_id(0) - 1;
  const unsigned int block_u = (get_group_id(1) + 1) % get_num_groups(1);
  const unsigned int block_d = (get_group_id(1) == 0) ? get_num_groups(1) - 1 : get_group_id(1) - 1;

  if (get_local_id(1) == 0)
  { 
    for(kk=0; kk < NSPEEDS; kk++)
    {
      block[get_local_id(0) + 1].speeds[kk] = cells[(get_local_size(1) * block_d + get_local_size(1) - 1) * nx + idx].speeds[kk];
    }
  }

  if (get_local_id(1) == get_local_size(1) - 1)
  {
    for(kk=0; kk < NSPEEDS; kk++)
    {
      block[id_b + get_local_size(0) + 2].speeds[kk] = cells[(get_local_size(1) * block_u) * nx + idx].speeds[kk];
    }
  }

  if (get_local_id(0) == get_local_size(0) - 1)
  {
    for(kk=0; kk < NSPEEDS; kk++)
    {
      block[id_b + 1].speeds[kk] = cells[nx * idy + (get_local_size(0) * block_r)].speeds[kk];
    }
  }

  if (get_local_id(0) == 0)
  {
    for(kk=0; kk < NSPEEDS; kk++)
    {
      block[id_b - 1].speeds[kk] = cells[nx * idy + (get_local_size(0) * block_l + get_local_size(0) - 1)].speeds[kk];
    }
  }

  for(kk=0; kk < NSPEEDS; kk++)
  {
    block[0].speeds[kk] = cells[nx * (get_local_size(1) * block_d + get_local_size(1) - 1) + (get_local_size(0) * block_l) + get_local_size(0) - 1].speeds[kk];
    block[get_local_size(0) + 1].speeds[kk] = cells[nx * (get_local_size(1) * block_d + get_local_size(1) - 1) + (get_local_size(0) * block_r)].speeds[kk];
    block[(get_local_size(0) + 2) * (get_local_size(1) + 1)].speeds[kk] = cells[nx * (get_local_size(1) * block_u) + (get_local_size(0) * block_l) + get_local_size(0) - 1].speeds[kk];
    block[(get_local_size(0) + 2) * (get_local_size(1) + 2) - 1].speeds[kk] = cells[nx * (get_local_size(1) * block_u) + (get_local_size(0) * block_r)].speeds[kk];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int x_w, x_e, y_n, y_s;
  
  x_e = get_local_id(0) + 2;
  x_w = get_local_id(0);
  y_n = get_local_id(1) + 2;
  y_s = get_local_id(1);

  if (obstacles[id])
  {
    cells[id].speeds[1] = block[x_e + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[3];
    cells[id].speeds[2] = block[get_local_id(0) + 1 + y_n * (get_local_size(0) + 2)].speeds[4];
    cells[id].speeds[3] = block[x_w + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[1];
    cells[id].speeds[4] = block[get_local_id(0) + 1 + y_s * (get_local_size(0) + 2)].speeds[2];
    cells[id].speeds[5] = block[x_e + y_n * (get_local_size(0) + 2)].speeds[7];
    cells[id].speeds[6] = block[x_w + y_n * (get_local_size(0) + 2)].speeds[8];
    cells[id].speeds[7] = block[x_w + y_s * (get_local_size(0) + 2)].speeds[5];
    cells[id].speeds[8] = block[x_e + y_s * (get_local_size(0) + 2)].speeds[6];

  } else {
        float local_density = 0.f;

        local_density += block[get_local_id(0) + 1 + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[0]
          + block[x_w + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[1] 
          + block[get_local_id(0) + 1 + y_s * (get_local_size(0) + 2)].speeds[2]
          + block[x_e + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[3]
          + block[get_local_id(0) + 1 + y_n * (get_local_size(0) + 2)].speeds[4]
          + block[x_w + y_s * (get_local_size(0) + 2)].speeds[5]
          + block[x_e + y_s * (get_local_size(0) + 2)].speeds[6]
          + block[x_e + y_n * (get_local_size(0) + 2)].speeds[7]
          + block[x_w + y_n * (get_local_size(0) + 2)].speeds[8];


        float u_x = (block[x_w + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[1]
         + block[x_w + y_s * (get_local_size(0) + 2)].speeds[5] 
         + block[x_w + y_n * (get_local_size(0) + 2)].speeds[8] 
         - block[x_e + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[3] 
         - block[x_e + y_s * (get_local_size(0) + 2)].speeds[6] 
         - block[x_e + y_n * (get_local_size(0) + 2)].speeds[7]) / local_density;
        
        float u_y = (block[get_local_id(0) + 1 + y_s * (get_local_size(0) + 2)].speeds[2] 
          + block[x_w + y_s * (get_local_size(0) + 2)].speeds[5] 
          + block[x_e + y_s * (get_local_size(0) + 2)].speeds[6] 
          - block[get_local_id(0) + 1 + y_n * (get_local_size(0) + 2)].speeds[4]
          - block[x_e + y_n * (get_local_size(0) + 2)].speeds[7] 
          - block[x_w + y_n * (get_local_size(0) + 2)].speeds[8]) / local_density;

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

        cells[id].speeds[0] = block[get_local_id(0) + 1 + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[0]
          + omega * (d_equ[0] - block[get_local_id(0) + 1 + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[0]);
        cells[id].speeds[1] = block[x_w + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[1]
          + omega * (d_equ[1] - block[x_w + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[1]);
        cells[id].speeds[2] = block[get_local_id(0) + 1 + y_s * (get_local_size(0) + 2)].speeds[2]
          + omega * (d_equ[2] - block[get_local_id(0) + 1 + y_s * (get_local_size(0) + 2)].speeds[2]);
        cells[id].speeds[3] = block[x_e + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[3]
          + omega * (d_equ[3] - block[x_e + (get_local_id(1) + 1) * (get_local_size(0) + 2)].speeds[3]);
        cells[id].speeds[4] = block[get_local_id(0) + 1 + y_n * (get_local_size(0) + 2)].speeds[4]
          + omega * (d_equ[4] - block[get_local_id(0) + 1 + y_n * (get_local_size(0) + 2)].speeds[4]);
        cells[id].speeds[5] = block[x_w + y_s * (get_local_size(0) + 2)].speeds[5]
          + omega * (d_equ[5] - block[x_w + y_s * (get_local_size(0) + 2)].speeds[5]);
        cells[id].speeds[6] = block[x_e + y_s * (get_local_size(0) + 2)].speeds[6]
          + omega * (d_equ[6] - block[x_e + y_s * (get_local_size(0) + 2)].speeds[6]);
        cells[id].speeds[7] = block[x_e + y_n * (get_local_size(0) + 2)].speeds[7]
          + omega * (d_equ[7] - block[x_e + y_n * (get_local_size(0) + 2)].speeds[7]);
        cells[id].speeds[8] = block[x_w + y_n * (get_local_size(0) + 2)].speeds[8]
          + omega * (d_equ[8] - block[x_w + y_n * (get_local_size(0) + 2)].speeds[8]);

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