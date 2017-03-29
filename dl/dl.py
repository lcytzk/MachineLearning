#!/usr/bin/env python

from numpy import *
import math

class Example:

    def __init__(self, label, features):
        self.label = label
        self.fs = features

class DataProvider:

    def __init__(self, filename):
        self.f = open(filename, 'r')

    def __iter__(self):
        return self

    def next(self):
        line = self.f.readline().strip()
        if line == '':
            raise StopIteration
        label, rest = line.split(',')
        return (float(label), map(int, rest.split()))

class Sigmoid:

    def firstDerivative(self, y, pred):
        return pred * (1 - pred)

class SqLoss:

    def firstDerivative(self, y, pred):
        return pred - y


class DeepNode:

    def __init__(self, dim, x, activation = Sigmoid()):
        self.w = zeros((1, dim))
        self.rate = 0.001
        self.act = activation
        self.y = None
        self.x = x
        self.g = None

    def cal(self):
        x = [node.y for node in self.x]
        self.y = 1 / (1 + math.exp(-dot(self.w, x)))
        return self.y

    def update(self, g):
        self.g = g
        self.back = self.act.firstDerivative(0, self.y)
        for i in xrange(len(self.w)):
            self.w[i] -= self.rate * self.back * self.x[i].y
            self.back =

class DeepModel:

    def __init__(self, hiddenNum = 1, hiddenDim = 2, inputDim = 2, outputDim = 1):
        self.hs = []
        self.input = [DeepNode(1, None) for i in xrange(inputDim)]
        lastInp = self.input
        lastDim = inputDim
        for i in xrange(hiddenNum):
            l = []
            for j in xrange(hiddenDim):
                l.append(DeepNode(lastDim, lastInp))
            lastDim = hiddenDim
            lastInp = l
            self.hs.append(l)
        l = []
        for j in xrange(outputDim):
            l.append(DeepNode(lastDim, lastInp))
        self.hs.append(l)

    def train(self, x, y):
        for i in xrange(len(x)):
            self.input[i].y = x[i]
        for layer in self.hs:
            x = [node.cal() for node in layer]
            print x
        loss = 0.5 * ((y - x[0]) ** 2)
        self.update(y)
        print "loss : %f" % loss

    def update(self, y):
        for layer in self.hs[::-1]:
            for node in layer:
                g = node.y - y
                node.update(g)
                for i in xrange(len(node.w)):
                        node.w[i] -= node
        return


def learn(step, dm, dp):
    #for i in xrange(step):
    for e in dp:
        #print e
        dm.train(e[1], e[0])

if __name__ == '__main__':
    learn(step = 1, dm = DeepModel(), dp = DataProvider("train.data"))
