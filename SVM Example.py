import sklearn as skl
import matplotlib.pyplot as plt
import scipy as sns
from scipy.io import loadmat

data1 = loadmat('C:\\Users\\Harera\\Downloads\\Compressed\\ex6data1.mat')

x1 = data1["X"]
y1 = data1["y"]


def plotData(x, y, s):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(x[pos, 0], x[pos, 1], s=s, c='b', marker='o', linewidth=1)
    plt.scatter(x[neg, 0], x[neg, 1], s=s, c='r', marker='x', linewidth=1)
    plt.show()


plotData(x1, y1, 50)

print(data1)
