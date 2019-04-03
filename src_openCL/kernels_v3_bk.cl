#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void rebound(global t_speed* cells, 
                    global t_speed* tmp_cells, 
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

float Awrk[NSPEEDS];
  Awrk[0] = cells[ii + jj*nx].speeds[0];
  Awrk[1] = cells[x_w + jj*nx].speeds[1];
  Awrk[2] = cells[ii + y_s*nx].speeds[2];
  Awrk[3] = cells[x_e + jj*nx].speeds[3];
  Awrk[4] = cells[ii + y_n*nx].speeds[4];
  Awrk[5] = cells[x_w + y_s*nx].speeds[5];
  Awrk[6] = cells[x_e + y_s*nx].speeds[6];
  Awrk[7] = cells[x_e + y_n*nx].speeds[7];
  Awrk[8] = cells[x_w + y_n*nx].speeds[8];

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (obstacles[jj*nx + ii])
  {
    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[ii + jj*nx].speeds[1] = Awrk[3];
    cells[ii + jj*nx].speeds[2] = Awrk[4];
    cells[ii + jj*nx].speeds[3] = Awrk[1];
    cells[ii + jj*nx].speeds[4] = Awrk[2];
    cells[ii + jj*nx].speeds[5] = Awrk[7];
    cells[ii + jj*nx].speeds[6] = Awrk[8];
    cells[ii + jj*nx].speeds[7] = Awrk[5];
    cells[ii + jj*nx].speeds[8] = Awrk[6];
  } else {
        /* compute local density total */
        float local_density = 0.f;

        /*
        local_density += Awrk[0]+Awrk[1]+Awrk[2]+Awrk[3]+Awrk[4]+Awrk[5]+Awrk[6]+Awrk[7]+Awrk[8];
        */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += Awrk[kk];
        }

        /* compute x velocity component */
        float u_x = (Awrk[1] + Awrk[5] + Awrk[8] - (Awrk[3] + Awrk[6] + Awrk[7])) / local_density;
        /* compute y velocity component */
        float u_y = (Awrk[2] + Awrk[5] + Awrk[6] - (Awrk[4] + Awrk[7] + Awrk[8])) / local_density;

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

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          cells[ii + jj * nx].speeds[kk] = Awrk[kk] + omega * (d_equ[kk] - Awrk[kk]);
        }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}