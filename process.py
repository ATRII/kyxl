import numpy as np
import matplotlib.pyplot as plt

rwd = np.load("./rwd.npy")
plt.plot(rwd.T)
plt.show()
