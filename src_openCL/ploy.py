import matplotlib.pyplot as plt
import numpy as np
# plt.xlim(0, 1000)
# plt.ylim(0, 10000)
x = [0.1, 23.3, 100, 1000]
y = [10, 3524, 3524, 3524]
plt.plot(x, y)
plt.xlim([0, 1000])
plt.ylim([0, 10000])
plt.yticks(np.arange(0, 1000, 10))
plt.xticks(np.arange(0, 10000, 10))

# plt.xticks(np.arange(min(x), max(x), 10))
# plt.yticks(np.arange(min(y), max(y), 10))
plt.ylabel('Stimulus average')
plt.xlabel('Time(ms)')
plt.title('Q3: Spike triggered average over a 100ms')
plt.show()
