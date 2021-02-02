# http://seaborn.pydata.org/examples/errorband_lineplots.html
# from data dir use ls | entr -rc python3 plot.py to run on any source or data change

import matplotlib.pyplot as plt;
# import seaborn
import numpy as np


def plot_f1():
    best_so_far = np.loadtxt("f1.stats")
    mean = best_so_far.mean(axis=0)

    for line in best_so_far:
        plt.plot(line, color="blue", alpha=0.45, linewidth=0.5)
    plt.plot(mean, color="red", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.yscale("log")
    plt.savefig("f1")
    plt.clf()

def plot_f2():
    best_so_far = np.loadtxt("f2.stats")
    mean = best_so_far.mean(axis=0)

    for line in best_so_far:
        plt.plot(line, color="blue", alpha=0.45, linewidth=0.5)
    plt.plot(mean, color="red", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.xlim(0,300)
    plt.yscale("log")
    plt.savefig("f2")
    plt.clf()

def plot_f3():
    best_so_far = np.loadtxt("f3.stats")
    mean = best_so_far.mean(axis=0)

    for line in best_so_far:
        plt.plot(line, color="blue", alpha=0.45, linewidth=0.5)
    plt.plot(mean, color="red", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("function result")
    plt.xlim(0,300)
    # plt.yscale("log")
    plt.savefig("f3")
    plt.clf()


plot_f1()
# plot_f2()
# plot_f3()
