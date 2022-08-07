from math import e
import numpy as np
from scipy import signal
import skimage.measure

# DATASET STUFF
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils

# BINARY CROSS ENTROPY FUNCTIONS
def binary_cross_entropy(actual, predicted):
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

def delta_binary_cross_entropy(actual, predicted):
    return ((1 - actual) / (1 - predicted) - actual / predicted) / np.size(actual)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, outputGradient, learningRate):
        pass

class Dense(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.biases = np.random.randn(outputSize, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, outputGradient, learningRate):
        # The partial derivative with respect to the input. In other words, the amount that the change in the input affected the change in the output
        output = np.dot(self.weights.T, outputGradient)

        # Set the descent gradient to the dot product between the cost and the transpose of the inputs
        # This is the amount that the weights affected the output
        weightsGradient = np.dot(outputGradient, self.input.T)
        self.weights -= learningRate * weightsGradient

        # Since the input for the bias is always one, simply change the biases by the cost
        # This represents how much the biases changed the output
        self.biases -= learningRate * outputGradient

        return output

class Activation(Layer):
    def __init__(self, activation, deltaActivation):
        self.activation = activation
        self.deltaActivation = deltaActivation

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.deltaActivation(self.input))

class Convolution(Layer):
    def __init__(self, inputShape, filterSize, depth):
        """
            - Input Depth: a 3-dimensional tuple containing the depth of the input (number of images), the height of the input, and the width of the input respectively
            - Filter size: an integer denotating the dimensions of the kernel filter (n * n)
            - Depth: the number of kernels
        """
        inputDepth, inputHeight, inputWidth = inputShape
        self.depth = depth
        self.inputShape = inputShape
        self.inputDepth = inputDepth

        # A 3d shape as the number of outputs is equal to the number of kernels
        self.outputShape = (depth, inputHeight - (filterSize - 1), inputWidth - (filterSize - 1))

        # A 4d shape that represents the number of kernels. Each kernel has a filter of size n*n. For each input image, there exists a kernel
        # When performing forward propogation, each input is summed into the final output
        self.kernelsShape = (depth, inputDepth, filterSize, filterSize)
        self.kernels = np.random.randn(*self.kernelsShape)
        self.biases = np.random.randn(*self.outputShape)

    def forward(self, input):
        """
        - Input: an array of images
        """

        self.inputs = input
        self.outputs = np.copy(self.biases)
        for depth in range( self.depth):
            for inputNum in range(self.inputDepth):
                self.outputs[depth] += signal.correlate2d(input[inputNum], self.kernels[depth, inputNum], "valid")
        
        return self.outputs 

    def backward(self, outputGradient, learningRate):
        kernelsGradient = np.zeros(self.kernelsShape)
        inputGradient = np.zeros(self.inputShape)

        for i in range(self.depth):
            for j in range(self.inputDepth):
                kernelsGradient[i, j] = signal.correlate2d(self.inputs[j], outputGradient[i], "valid")
                inputGradient[j] += signal.convolve2d(outputGradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learningRate * kernelsGradient
        self.biases -= learningRate * outputGradient
        return inputGradient

class Reshape(Layer):
    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forward(self, input):
        return np.reshape(input, self.outputShape)

    def backward(self, outputGradient, learningRate):
        return np.reshape(outputGradient, self.inputShape)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def delta_sigmoid(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, delta_sigmoid)

class SoftMax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, outputGradient, learningRate):
        n = np.size(self.output)
        tmp = np.tile(self.output, n)
        return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), outputGradient)

class Pool(Layer):
    def forward(self, input):
        # Compress the image by a factor of 2
        output = skimage.measure.block_reduce(input, block_size=(1, 2, 2), func=np.max)
        self.maxs = output.repeat(2, axis=1).repeat(2, axis=2)
        self.mask = np.equal(input, self.maxs).astype(int)

        return output

    def backward(self, outputGradient, learningRate):
        output = outputGradient.repeat(2, axis=1).repeat(2, axis=2)
        output = np.multiply(output, self.mask)
        return output

class Network:
    def __init__(self, layers, learningRate, epochs):
        self.layers = layers
        self.learningRate = learningRate
        self.epochs = epochs

    def learn(self, xInput, yInput):
        print("Beginning learning for", self.epochs, "iterations with", len(xInput), "training samples...")
        for e in range(self.epochs):
            error = 0
            for x, y in zip(xInput, yInput):
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                error = binary_cross_entropy(y, output)

                gradient = delta_binary_cross_entropy(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.learningRate)

            error /=  len(xInput)
            print("Error:", error)

        print("Learning completed for", self.epochs, "iterations for", len(xInput), "training samples...")

    def test(self, xInput, yInput):
        correct = 0
        for x, y in zip(xInput, yInput):
            output = x
            for layer in self.layers:
                output = layer.forward(output)

            if(np.argmax(output) == np.argmax(y)):
                correct += 1
            print("Predicted:", np.argmax(output), "Actual:", np.argmax(y))
        ratio = float(correct) / len(yInput)
        print("Correct/Total:", ratio)

def mnist_data(x, y, limit):
    allIndices = np.where(y <= 9)[0][:limit]
    allIndices = np.random.permutation(allIndices)

    x, y = x[allIndices], y[allIndices]
    x = np.pad(x, ((0,0), (2,2), (2, 2)), 'constant')
    x = x.reshape(len(x), 1, 32, 32)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, yTrain = mnist_data(xTrain, yTrain, 100)
xTest, yTest = mnist_data(xTest, yTest, 30000)

n = Network([
    Convolution((1, 32, 32), 5, 6),
    Sigmoid(),
    Pool(),
    Convolution((6, 14, 14), 5, 16),
    Sigmoid(),
    Pool(),
    Reshape((16, 5, 5), (16 * 5 * 5, 1)),
    Dense(16 * 5 * 5, 120),
    Sigmoid(),
    Dense(120, 10),
    SoftMax()],
    .05,
    20
)

n.learn(xTrain, yTrain)
n.test(xTest, yTest)