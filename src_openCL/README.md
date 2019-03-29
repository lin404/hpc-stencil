# OpenCL

# kernel code for all timestep functions (v0)

31.9s for 128\*128

# combine timestep functions in kernel code (v1)(v2)(v3)

(v1) all functions excluding accelerate_flow()

(v2) all functions including accelerate_flow()
ombine all functions by using CLK_GLOBAL_MEM_FENCE; all in global memory.
27.8s for 128\*128

(v3) exclude accelerate_flow()
28.8s for 128\*128
instead of using temp_cell, using a local list as intermediary to do value exchange.
doesn't help much, bug?

(v3_bk) include accelerate_flow()
not working, bug?

# kernel code for function av_velocity() (v4)

6.8s for 128\*128
move av_velocity() to kernel, calculating in global memory.

# av_velocity() calculation reduction (v5)

not working. why? (time should be the same as v6)

# combine av_velocity() to timestep (v6)

3.8s for 128\*128
not working, bug?

# move av_vels(t) into kernel (v7 based on v4)

108s for 128\*128
clEnqueueReadBuffer() is expensive, so tried this, but very inefficient...

# block cell for propagate(), rebound() and collision() (v9)

2.00s before
OpenCL error during 'waiting for rebound kernel'

# block cell for av_velocity() (v10)

4.00s before

# SoA reduction for av_velocity() (v11)

3.7s for 128\*128

# combine (v12)

2.9s for 128\*128

# read only one float value (v13)

how?

# opemMP (v14)

nothing help

# High Performance Computing - OpenCL
