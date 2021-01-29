# http://seaborn.pydata.org/examples/errorband_lineplots.html
# from data dir use ls | entr -rc python3 plot.py to run on any source or data change

import matplotlib.pyplot as plt;
import numpy as np

best_so_far = np.loadtxt("f1.stats")
mean = best_so_far.mean(axis=0)

for line in best_so_far:
    plt.plot(line, color="blue", alpha=0.05)
plt.plot(mean, color="blue")
plt.yscale("log")
# plt.show()
