import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'E:\\courses\Machine Learning\Machine Learning Hesham Asem\\02 تعليم الآلة , القسم الثاني  التوقع Machine learning , Regression\\1.txt'
data = pd.read_csv(path, header=None, names=["Population", "Profit"])

plt = data.plot(kind='scatter', x="Population", y="Profit", figsize=(5,5))




data.insert(0, "Ones", 1)
data


cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]


x = np.matrix(x.values)
y = np.matrix(y.values)



def computeLossCost(x,y,theta):
    z = np.power((x*theta.T) - y,2)
    return np.sum(z) / (2*len(z))


def GradientDescent(x, y, theta, its, alpha):
    cost = np.zeros(its)

    for i in range(its):
        error = (x * theta.T) - y
        NewTheta = error.T * x
        theta = theta - ((alpha / len(x)) * np.sum(NewTheta))
        cost[i] = computeLossCost(x, y, theta)
    return theta, cost


theta  = np.matrix(np.array([0,0]))
theta, cost = GradientDescent(x, y, theta, 1000, 0.01)

print(theta, cost)
