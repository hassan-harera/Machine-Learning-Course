import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def init_centroids(x, k):
    m, n = x.shape
    centroids = np.zeros((k, n))

    rand = np.random.randint(0, m, k)

    for i in range(rand.size):
        centroids[i] = x[rand[i]]

    return centroids


def clustering(x, cords):
    m, n = x.shape
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(cords.shape[0]):
            dis = np.sum((x[i, :] - cords[j, :]) ** 2)
            if dis < min_dist:
                min_dist = dis
                idx[i] = j

    return idx


def displacement(x, idx, shape):
    new_cords = np.zeros(shape)

    for i in range(shape[0]):
        cluster_elements = np.where(idx == i)
        new_cords[i, :] = (np.sum(x[cluster_elements, :], axis=1) / len(cluster_elements[0])).ravel()

    return new_cords


data1 = loadmat('ex7data2.mat')

x = data1["X"]

init_centroids_var = init_centroids(x, 3)

for i in range(20):
    idxs = clustering(x, init_centroids_var)
    old_centroids = init_centroids_var
    init_centroids_var = displacement(x, idxs, init_centroids_var.shape)

    clusters1 = x[np.where(idxs == 0), :][0]
    clusters2 = x[np.where(idxs == 1), :][0]
    clusters3 = x[np.where(idxs == 2), :][0]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(clusters1[:, 0], clusters1[:, 1], s=10, c="b", label="Cluster 1")
    ax.scatter(init_centroids_var[0, 0], init_centroids_var[0, 1], s=80, c="b")

    ax.scatter(clusters2[:, 0], clusters2[:, 1], s=10, c="r", label="Cluster 2")
    ax.scatter(init_centroids_var[1, 0], init_centroids_var[1, 1], s=80, c="r")

    ax.scatter(clusters3[:, 0], clusters3[:, 1], s=10, c="g", label="Cluster 3")
    ax.scatter(init_centroids_var[2, 0], init_centroids_var[2, 1], s=80, c="g")

    plt.show()

    if (init_centroids_var == old_centroids).all():
        break