import numpy as np


def segmoid(Zs):
    return 1.0 / (1 + np.exp(-Zs))


def segmoid_derivatve(Zs):
    return Zs * (1 - Zs)


class NeuralNetwork():
    def __int__(self, X, Y, neurons, outputSize):
        self.featureSize = X.shape[1]
        self.outputSize = outputSize
        self.X = X
        self.Y = Y

        self.weights1 = np.random.randn(self.featureSize, neurons)
        self.weights2 = np.random.randn(neurons, self.outputSize)

        self.output = np.zeros(Y.shape)

    def forwardProp(self):
        Zs1 = np.dot(self.X, self.weights1)
        self.layer1 = segmoid(Zs1)
        Zs2 = np.dot(self.layer1, self.weights2)
        self.output = segmoid(Zs2)

    def backProp(self):
        weights2 = np.dot(self.layer1.T, (2 * (self.Y - self.output) * segmoid_derivatve(self.output)))

        weights1 = np.dot(self.X.T,
                          (np.dot(2 * (self.Y - self.output) * segmoid_derivatve(self.output), self.weights2.T) *
                           segmoid_derivatve(self.layer1)))

        self.weights2 += weights2
        self.weights1 += weights1

    def predict(self, X):
        Zs1 = np.dot(X, self.weights1)
        layer1 = segmoid(Zs1)
        Zs2 = np.dot(layer1, self.weights2)
        output = segmoid(Zs2)
        return output


X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
Y = np.array(([92], [86], [89]), dtype=float)
xtest = np.array(([4, 8]), dtype=float)

X = X / np.max(X, axis=0)
xtest = xtest / np.max(xtest, axis=0)
Y = Y / 100

# print(X)
# print(X)
# print(Y)

nn = NeuralNetwork()
nn.__int__(X, Y, 5, 1)

for i in range(1000):
    nn.forwardProp()
    nn.backProp()

predicted = nn.predict(X)
print(predicted)
print(np.sum(np.square(Y - predicted)))