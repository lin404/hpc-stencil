#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9
#define blksz 16

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global int* obstacles,
                            global float* speed_0,
                            global float* speed_1,
                            global float* speed_2,
                            global float* speed_3,
                            global float* speed_4,
                            global float* speed_5,
                            global float* speed_6,
                            global float* speed_7,
                            global float* speed_8,
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
      && (speed_3[ii + jj* nx] - w1) > 0.f
      && (speed_6[ii + jj* nx] - w2) > 0.f
      && (speed_7[ii + jj* nx] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    speed_1[ii + jj* nx] += w1;
    speed_5[ii + jj* nx] += w2;
    speed_8[ii + jj* nx] += w2;
    /* decrease 'west-side' densities */
    speed_3[ii + jj* nx] -= w1;
    speed_6[ii + jj* nx] -= w2;
    speed_7[ii + jj* nx] -= w2;
  }
}

kernel void rebound(global float* speed_0,
                    global float* speed_1,
                    global float* speed_2,
                    global float* speed_3,
                    global float* speed_4,
                    global float* speed_5,
                    global float* speed_6,
                    global float* speed_7,
                    global float* speed_8,
                    global float* temp_speed_0,
                    global float* temp_speed_1,
                    global float* temp_speed_2,
                    global float* temp_speed_3,
                    global float* temp_speed_4,
                    global float* temp_speed_5,
                    global float* temp_speed_6,
                    global float* temp_speed_7,
                    global float* temp_speed_8,
                    global int* obstacles,
                    int nx, int ny, float omega,
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

  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  if (obstacles[jj*nx + ii])
  {
    temp_speed_0[ii + jj*nx] = speed_0[ii + jj*nx];
    temp_speed_1[ii + jj*nx] = speed_3[x_e + jj*nx];
    temp_speed_2[ii + jj*nx] = speed_4[ii + y_n*nx];
    temp_speed_3[ii + jj*nx] = speed_1[x_w + jj*nx];
    temp_speed_4[ii + jj*nx] = speed_2[ii + y_s*nx];
    temp_speed_5[ii + jj*nx] = speed_7[x_e + y_n*nx];
    temp_speed_6[ii + jj*nx] = speed_8[x_w + y_n*nx];
    temp_speed_7[ii + jj*nx] = speed_5[x_w + y_s*nx];
    temp_speed_8[ii + jj*nx] = speed_6[x_e + y_s*nx];
  } else {
        /* compute local density total */
        float local_density = 0.f;

        local_density += speed_0[ii + jj*nx] 
                      + speed_1[x_w + jj*nx]
                      + speed_2[ii + y_s*nx]
                      + speed_3[x_e + jj*nx]
                      + speed_4[ii + y_n*nx]
                      + speed_5[x_w + y_s*nx]
                      + speed_6[x_e + y_s*nx]
                      + speed_7[x_e + y_n*nx]
                      + speed_8[x_w + y_n*nx];

        float u_x = (speed_1[x_w + jj*nx] + speed_5[x_w + y_s*nx] + speed_8[x_w + y_n*nx] - (speed_3[x_e + jj*nx] + speed_6[x_e + y_s*nx] + speed_7[x_e + y_n*nx])) / local_density;
        float u_y = (speed_2[ii + y_s*nx] + speed_5[x_w + y_s*nx] + speed_6[x_e + y_s*nx] - (speed_4[ii + y_n*nx] + speed_7[x_e + y_n*nx] + speed_8[x_w + y_n*nx])) / local_density;

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

        temp_speed_0[ii + jj * nx] = speed_0[ii + jj*nx] + omega * (d_equ[0] - speed_0[ii + jj*nx]);
        temp_speed_1[ii + jj * nx] = speed_1[x_w + jj*nx] + omega * (d_equ[1] - speed_1[x_w + jj*nx]);
        temp_speed_2[ii + jj * nx] = speed_2[ii + y_s*nx] + omega * (d_equ[2] - speed_2[ii + y_s*nx]);
        temp_speed_3[ii + jj * nx] = speed_3[x_e + jj*nx] + omega * (d_equ[3] - speed_3[x_e + jj*nx]);
        temp_speed_4[ii + jj * nx] = speed_4[ii + y_n*nx] + omega * (d_equ[4] - speed_4[ii + y_n*nx]);
        temp_speed_5[ii + jj * nx] = speed_5[x_w + y_s*nx] + omega * (d_equ[5] - speed_5[x_w + y_s*nx]);
        temp_speed_6[ii + jj * nx] = speed_6[x_e + y_s*nx] + omega * (d_equ[6] - speed_6[x_e + y_s*nx]);
        temp_speed_7[ii + jj * nx] = speed_7[x_e + y_n*nx] + omega * (d_equ[7] - speed_7[x_e + y_n*nx]);
        temp_speed_8[ii + jj * nx] = speed_8[x_w + y_n*nx] + omega * (d_equ[8] - speed_8[x_w + y_n*nx]);

  }
}

kernel void av_velocity_kernel(global float* temp_speed_0,
                              global float* temp_speed_1,
                              global float* temp_speed_2,
                              global float* temp_speed_3,
                              global float* temp_speed_4,
                              global float* temp_speed_5,
                              global float* temp_speed_6,
                              global float* temp_speed_7,
                              global float* temp_speed_8,
                              global int *obstacles,
                              global float *global_sums,
                              local float *local_sums,
                              int nx, int ny, int groupsz)
{
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  
  int local_ii = get_local_id(0);
  int local_jj = get_local_id(1);   

  int group_ii = get_group_id(0);
  int group_jj = get_group_id(1);

  int work_items_x  = get_local_size(0);
  int work_items_y  = get_local_size(1);

   if (!obstacles[ii + jj * nx])
      {   
        float local_density = 0.f;

        local_density += temp_speed_0[ii + jj * nx]
                      + temp_speed_1[ii + jj * nx]
                      + temp_speed_2[ii + jj * nx]
                      + temp_speed_3[ii + jj * nx]
                      + temp_speed_4[ii + jj * nx]
                      + temp_speed_5[ii + jj * nx]
                      + temp_speed_6[ii + jj * nx]
                      + temp_speed_7[ii + jj * nx]
                      + temp_speed_8[ii + jj * nx];

        float u_x = (temp_speed_1[ii + jj * nx] + temp_speed_5[ii + jj * nx] + temp_speed_8[ii + jj * nx] - (temp_speed_3[ii + jj * nx] + temp_speed_6[ii + jj * nx] + temp_speed_7[ii + jj * nx])) / local_density;
        float u_y = (temp_speed_2[ii + jj * nx] + temp_speed_5[ii + jj * nx] + temp_speed_6[ii + jj * nx] - (temp_speed_4[ii + jj * nx] + temp_speed_7[ii + jj * nx] + temp_speed_8[ii + jj * nx])) / local_density;
        
        local_sums[local_ii + local_jj * work_items_x] = sqrt((u_x * u_x) + (u_y * u_y));
      } else {
        local_sums[local_ii + local_jj * work_items_x] = 0.f;
      }

  barrier(CLK_LOCAL_MEM_FENCE);

  float sum;                              
  int i;
  int j;

  if (local_ii == 0 && local_jj==0) {                      
      sum = 0.0f;                            
   
      for (j=0; j<work_items_y; j++) {        
          for(i=0; i<work_items_x; i++){
            sum += local_sums[i+j*work_items_x];             
          } 
      }                                      
      global_sums[group_ii+ group_jj*groupsz] = sum;
   }
}