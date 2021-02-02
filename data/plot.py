# http://seaborn.pydata.org/examples/errorband_lineplots.html
# from data dir use ls | entr -rc python3 plot.py to run on any source or data change

import matplotlib.pyplot as plt;
# import seaborn
import numpy as np


def plot_f1():
    best_so_far = np.loadtxt("f1.stats")
    mean = best_so_far.mean(axis=0)
    std = best_so_far.std(axis=0)
    fitness = best_so_far[::,-1]
    x = np.linspace(0, len(mean), len(mean))
    print(f"{mean[-1]:.1E} +- {std[-1]:.1E}")

    plt.fill_between(x, mean+std, mean-std, alpha=0.4, color="blue")
    # for line in best_so_far:
    #     plt.plot(x, line, color="blue", alpha=0.15, linewidth=0.5)
    plt.plot(x, mean, color="red", linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.yscale("log")
    plt.ylim(bottom=1e-15)
    plt.savefig("f1")
    plt.clf()

def plot_f2():
    best_so_far = np.loadtxt("f2.stats")
    mean = best_so_far.mean(axis=0)
    std = best_so_far.std(axis=0)
    fitness = best_so_far[::,-1]
    x = np.linspace(0, len(mean), len(mean))
    print(f"{mean[-1]:.1E} +- {std[-1]:.1E}")

    for line in best_so_far:
        plt.plot(x, line, color="blue", alpha=0.15, linewidth=0.5)
    plt.plot(x, mean, color="red", linewidth=2)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.yscale("log")
    plt.xlim(0,300)
    plt.savefig("f2")
    plt.clf()

def plot_f3():
    best_so_far = []
    with open("f3.stats") as file:
        for line in file.readlines():
            best_so_far.append( np.fromstring(line, sep=' '))

    max_cols = max(map(len, best_so_far))
    mean = np.zeros(max_cols)
    for i in range(0, max_cols):
        column = []
        for best in best_so_far:
            if i < len(best):
                column.append(best[i])
        mean[i] = np.array(column).mean()

    for line in best_so_far:
        plt.plot(line, color="blue", alpha=0.45, linewidth=0.2)
    plt.plot(mean, color="red", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.xlim(0,150)
    plt.savefig("f3")
    plt.clf()


plot_f1()
plot_f2()
plot_f3()
