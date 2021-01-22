import numpy as np


def segmoid(Zs):
    return 1.0 / (1 + np.exp(-Zs))


def segmoid_derivatve(Zs):
    return Zs * (1 - Zs)


class NeuralNetwork():
    def __int__(self, X, Y, neurons):
        self.input = X
        self.Y = Y

        self.weights1 = np.random.rand(self.input.shape[1], neurons)
        self.weights2 = np.random.rand(neurons, 1)

        self.output = np.zeros(self.Y.shape)

    def forwardProp(self):
        Zs1 = np.dot(self.input, self.weights1)
        self.layer1 = segmoid(Zs1)
        Zs2 = np.dot(self.layer1, self.weights2)
        self.output = segmoid(Zs2)

    def backProp(self):
        weights2 = np.dot(self.layer1.T, (2 * (self.Y - self.output) * segmoid_derivatve(self.output)))

        weights1 = np.dot(self.input.T,
                          (np.dot(2 * (self.Y - self.output) * segmoid_derivatve(self.output), self.weights2.T) *
                           segmoid_derivatve(self.layer1)))

        self.weights2 += weights2
        self.weights1 += weights1


X = np.array([[0, 0, 1],
              [0, 1, 1, ],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

nn = NeuralNetwork()
nn.__int__(X, Y, 4)

for i in range(500):
    nn.forwardProp()
    nn.backProp()

print(nn.output)
