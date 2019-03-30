#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

kernel void rebound(global float* speed_0,
                    global float* speed_1,
                    global float* speed_2,
                    global float* speed_3,
                    global float* speed_4,
                    global float* speed_5,
                    global float* speed_6,
                    global float* speed_7,
                    global float* speed_8,
                    global float* tmp_speed_0,
                    global float* tmp_speed_1,
                    global float* tmp_speed_2,
                    global float* tmp_speed_3,
                    global float* tmp_speed_4,
                    global float* tmp_speed_5,
                    global float* tmp_speed_6,
                    global float* tmp_speed_7,
                    global float* tmp_speed_8,
                    global int* obstacles,
                    global float *global_sums,
                    local float *local_sums,
                    int nx, int ny,
                    float omega, int groupsz,
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
      && (speed_3[ii + jj* nx] - w3) > 0.f
      && (speed_6[ii + jj* nx] - w4) > 0.f
      && (speed_7[ii + jj* nx] - w4) > 0.f)
  {
    /* increase 'east-side' densities */
    speed_1[ii + jj* nx] += w3;
    speed_5[ii + jj* nx] += w4;
    speed_8[ii + jj* nx] += w4;
    /* decrease 'west-side' densities */
    speed_3[ii + jj* nx] -= w3;
    speed_6[ii + jj* nx] -= w4;
    speed_7[ii + jj* nx] -= w4;
  }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  
  float speeds[NSPEEDS];
  speeds[0] = speed_0[ii + jj*nx];
  speeds[1] = speed_1[x_w + jj*nx];
  speeds[2] = speed_2[ii + y_s*nx];
  speeds[3] = speed_3[x_e + jj*nx];
  speeds[4] = speed_4[ii + y_n*nx];
  speeds[5] = speed_5[x_w + y_s*nx];
  speeds[6] = speed_6[x_e + y_s*nx];
  speeds[7] = speed_7[x_e + y_n*nx];
  speeds[8] = speed_8[x_w + y_n*nx];

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

        tmp_speed_0[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[0]) : (speeds[0] + omega * (d_equ[0] - speeds[0]));
        tmp_speed_1[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[3]) : (speeds[1] + omega * (d_equ[1] - speeds[1]));
        tmp_speed_2[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[4]) : (speeds[2] + omega * (d_equ[2] - speeds[2]));
        tmp_speed_3[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[1]) : (speeds[3] + omega * (d_equ[3] - speeds[3]));
        tmp_speed_4[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[2]) : (speeds[4] + omega * (d_equ[4] - speeds[4]));
        tmp_speed_5[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[7]) : (speeds[5] + omega * (d_equ[5] - speeds[5]));
        tmp_speed_6[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[8]) : (speeds[6] + omega * (d_equ[6] - speeds[6]));
        tmp_speed_7[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[5]) : (speeds[7] + omega * (d_equ[7] - speeds[7]));
        tmp_speed_8[ii + jj * nx] = (obstacles[jj*nx + ii]) ? (speeds[6]) : (speeds[8] + omega * (d_equ[8] - speeds[8]));
      
      
       float local_density_tmp = 0.f;

        local_density_tmp += tmp_speed_0[ii + jj * nx]
                      + tmp_speed_1[ii + jj * nx]
                      + tmp_speed_2[ii + jj * nx]
                      + tmp_speed_3[ii + jj * nx]
                      + tmp_speed_4[ii + jj * nx]
                      + tmp_speed_5[ii + jj * nx]
                      + tmp_speed_6[ii + jj * nx]
                      + tmp_speed_7[ii + jj * nx]
                      + tmp_speed_8[ii + jj * nx];

        float u_x_tmp = (tmp_speed_1[ii + jj * nx] + tmp_speed_5[ii + jj * nx] + tmp_speed_8[ii + jj * nx] - (tmp_speed_3[ii + jj * nx] + tmp_speed_6[ii + jj * nx] + tmp_speed_7[ii + jj * nx])) / local_density_tmp;
        float u_y_tmp = (tmp_speed_2[ii + jj * nx] + tmp_speed_5[ii + jj * nx] + tmp_speed_6[ii + jj * nx] - (tmp_speed_4[ii + jj * nx] + tmp_speed_7[ii + jj * nx] + tmp_speed_8[ii + jj * nx])) / local_density_tmp;
        
        local_sums[local_ii + local_jj * work_items_x] = (obstacles[jj*nx + ii]) ? (0.f) : (sqrt((u_x_tmp * u_x_tmp) + (u_y_tmp * u_y_tmp)));
    
    float sum;
    int i;
    int j;
    
      for (j=work_items_y/2; j>0; j/=2)
      {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_jj<j)
        {
          local_sums[local_ii + local_jj * work_items_x] += local_sums[local_ii + (local_jj + j) * work_items_x];
        }
      }

    if (local_ii == 0 && local_jj == 0) 
    {
      sum = 0.0f;
       for(i=0; i<work_items_x; i++)
       {
            sum += local_sums[i];             
        } 
        global_sums[group_ii + group_jj*groupsz] = sum;
    }

}

