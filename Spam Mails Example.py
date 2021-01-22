import sklearn as skl
import matplotlib.pyplot as plt
import scipy as sns
import numpy as np
from scipy.io import loadmat
from sklearn import svm

spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')



x_test = spam_test["Xtest"]
x_train = spam_train["X"]
y_train = spam_train["y"]
y_test = spam_test["ytest"]

# print(x_test, x_train, y_test, y_train)

svc = svm.SVC()
svc.fit(x_train, y_train.ravel())

print(np.round(svc.score(x_test, y_test.ravel()) * 100, 2))
print(x_test.shape)