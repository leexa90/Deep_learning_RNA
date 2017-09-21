import matplotlib.pyplot as plt

import numpy as np

data = np.load('result.npy').item()

plt.plot(range(0,300), [data[x][0] for x in data])
plt.plot(range(0,300), [data[x][1] for x in data])
plt.show()
