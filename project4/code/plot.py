"""Program for plotting results obtained from ising_model. """

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12,12))
plt.title("Test plot")
plt.plot([0.0], [0.0], 'o', label="test")

lattice = ["20"]

for lat in lattice:
    x = np.loadtxt("lattice" + lat)

    plt.plot(x[0, :], x[1, :])

plt.xlabel("temp")
plt.ylabel("expectation")
plt.legend()

plt.show()
