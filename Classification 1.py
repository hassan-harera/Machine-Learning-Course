from builtins import type

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "3.txt"

data = pd.read_csv(path, names=["Exam 1", "Exam 2", "Admitted"])
# data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# print(data.describe())
# print(data.head())

# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

positive = data[data["Admitted"] == 1]
negative = data[data["Admitted"] == 0]

# print("****************")
# print(positive.head())
# print(negative.head())

fig, ax = plt.subplots(figsize=(5, 5))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')

ax.scatter(negative['Exam 1'], negative['Exam 2'], s=20, c="b", marker="x", label="not admitted")
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=20, c="r", marker="o", label="admitted")

ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


# plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    # return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(nums, sigmoid(nums), 'r')


# plt.show()


# def cost(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
#     return np.sum(first - second) / (len(X))


def cost(theta, X, Y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)

    predicted = sigmoid(X * theta.T)
    logPredicted = np.log(predicted)

    positive = np.multiply(-Y, logPredicted)
    negative = np.multiply((1 - Y), np.log(1 - predicted))
    return np.sum((positive - negative)) / len(X)


# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# print('X.shape = ', X.shape)
# print('theta.shape = ', theta.shape)
# print('y.shape = ', y.shape)

thiscost = cost(np.matrix(theta), np.matrix(X), np.matrix(y))


# print()
# print('cost = ', thiscost)


def gradient(theta, X, Y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    Y = np.matrix(Y)

    predicted = sigmoid(X * theta.T)
    error = predicted - Y

    gradientVal = np.zeros(X.shape[1])

    thetaN = X.shape[1]
    for i in range(thetaN):
        gradientVal[i] = np.sum(np.multiply(error, X[:, i])) / len(X)
    return gradientVal


# def gradient(theta, X, y):
#     parameters = theta.shape[1]
#     grad = np.zeros(parameters)
#
#     error = sigmoid(X * theta.T) - y
#
#     for i in range(parameters):
#         term = np.multiply(error, X[:, i])
#         grad[i] = np.sum(term) / len(X)
#
#     return grad


# print(gradient(theta, X, y))

import scipy.optimize as opt

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

theta = result[0]


costafteroptimize = cost(theta, X, y)


# print(costafteroptimize)


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    predicted = np.zeros(probability.shape[0])
    for x in range(probability.shape[0]):
        predicted[x] = 1 if probability[x, 0] >= 0.5 else 0
    return predicted


# def predict(theta, X):
#     probability = sigmoid(X * theta.T)
#
#     predicted =  [1 if x >= 0.5 else 0 for x in probability]
#     return predicted

probability = sigmoid(X * theta.T)
print(probability)

predicted = predict(theta, X)
print(predicted)
errors = 0
for i in range(len(predicted)):
    if predicted[i] != y[i]:
        errors += 1

accuracy = 1 - (errors / len(y))
accuracy * 100
print(accuracy)

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
