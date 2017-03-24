#!/usr/bin/env python

from numpy import *
import math

class DataProvider:

    def __init__(self, filename):
        f = open(filename, 'r')

    def getOne():
        return map(float, f.readline().strip().split())

class Sigmoid:

    def firstDerivative(w, x):

class DeepNode:
    def __init__(self, dim, activation = Sigmoid()):
        self.w = zeros((1, dim))
        self.rate = 0.5
        self.act = activation

    def cal(self, x):
        return 1 / (1 + math.exp(-dot(self.w, x)))

    def update(self, g):
        for i in xrange(len(self.w)):
            self.w[i] -= self.rate * g;

class DeepModel:

    def __init__(self, hiddenNum, hiddenDim, inputDim, outputDim):
        self.hs = []
        lastDim = inputDim
        for i in xrange(hiddenNum):
            l = []
            for j in xrange(hiddenDim):
                l.append(DeepNode(lastDim))
            lastDim = hiddenDim
            self.hs.append(l)
        l = []
        for j in xrange(outputDim):
            l.append(DeepNode(lastDim))
        self.hs.append(l)

    def train(self, x, y):
        for layer in xrange(self.hs):
            x = [node.cal(x) for node in xrange(layer)]
        loss = sum([0.5 * ((y[i] - x[i]) ** 2) for i in xrange(len(y))])
        print "loss : %f" % loss
        return x

    def update(self, g):


def learn(step = 1, dm, dp):
    for i in xrange(step):
        dm.train(dp.getOne())

if __name__ == '__main__':
    deepLearn(dm = DeepModel(3), dp = DataProvider("train.data"))
