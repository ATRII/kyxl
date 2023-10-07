import numpy as np
import matplotlib.pyplot as plt

loss = np.load("./loss.npy")
plt.plot(loss.T)
plt.show()
