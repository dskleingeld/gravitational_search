# http://seaborn.pydata.org/examples/errorband_lineplots.html
# from data dir use ls | entr -rc python3 plot.py to run on any source or data change

from typing import Optional
import os
import matplotlib.pyplot as plt;
# import seaborn
import numpy as np

def analyse(path: str):
    best_so_far = []
    with open(path) as file:
        for line in file.readlines():
            best_so_far.append( np.fromstring(line, sep=' '))

    max_cols = max(map(len, best_so_far))
    mean = np.zeros(max_cols)
    std = np.zeros(max_cols)
    for i in range(0, max_cols):
        column = []
        for best in best_so_far:
            if i < len(best):
                column.append(best[i])

        mean[i] = np.array(column).mean()
        std[i] = np.array(column).std()
    n = [len(x) for x in best_so_far]
    n = np.array(n).mean()

    best=9e20;
    for run in best_so_far:
        run = np.array(run)
        run = run[np.nonzero(run)]
        best=min(best,np.min(run))

    return (best_so_far, mean, std, n, best)


def plot(name: str, log:bool =False, ylim=None, xlim=None):
    (best_so_far, mean, std, n, best) = analyse(name+".stats")
    print(f"{mean[-1]:.1E} +- {std[-1]:.1E}, best: {best:.1E}, n: {n}")

    for line in best_so_far:
        plt.plot(line, color="blue", alpha=0.15, linewidth=0.5)
    plt.plot(mean, color="red", linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    if log:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(bottom=1e-15)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.tight_layout(pad=1)
    plt.savefig("../plots/"+name)
    plt.clf()

plot("gsa/f1_gsa", log=True, ylim=1e-15)
plot("gabsa/f1_gsa", log=True)
print()
plot("gsa/f2_gsa", log=True, xlim=(0,300))
plot("gabsa/f2_gsa", log=True, xlim=(0,300))
print()
plot("gsa/f3_gabsa", log=True)
plot("gabsa/f3_gabsa", log=True)
print()
plot("gabsa/f1_gabsa", log=True)
plot("gabsa/f2_gabsa", log=True)
