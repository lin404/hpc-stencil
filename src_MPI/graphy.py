import matplotlib.pyplot as plt
import numpy as numpy
from scipy.interpolate import spline

x = numpy.array([2,4,6,8,10,12,14,16])
y = numpy.array([1.17,0.8,0.78,0.782751,0.754262,0.760647,0.757946,0.773727])

x_2 = numpy.array([4,8,10,12,16])
y_2 = numpy.array([0.316355,0.075324,0.050536,0.040536,0.020536])

x_smooth = numpy.linspace(x.min(), x.max(), 300)
y_smooth = spline(x,y,x_smooth)

x_smooth_2 = numpy.linspace(x_2.min(), x_2.max(), 300)
y_smooth_2 = spline(x_2,y_2,x_smooth_2)

plt.plot(x_smooth,y_smooth)
plt.plot(x_smooth_2,y_smooth_2)
plt.legend(('decompose by rows', 'decompose as tiles'),
           shadow=False, loc=(0.6, 0.8), handlelength=1, fontsize=10)
plt.xlabel('the number of cores')
plt.ylabel('the runtimes of stencil')
plt.show()

