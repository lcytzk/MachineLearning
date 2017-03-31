#!/usr/bin/env python

from numpy import *
import math
import sys

passes = 100
BIAS = True

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
        return ([float(label)], map(float, rest.split()))

class Sigmoid:

    def firstDerivative(self, y, pred):
        return pred * (1 - pred)

class SqLoss:

    def firstDerivative(self, y, pred):
        return pred - y


class DeepNode:

    def __init__(self, dim, x, activation = Sigmoid(), bias = True):
        self.bias = bias
        if bias:
            self.w = [0] * (dim + 1)
        else:
            self.w = [0] * dim
        self.rate = 1
        self.act = activation
        self.y = None
        self.x = x

    def cal(self):
        x = [node.y for node in self.x]
        if self.bias:
            x.append(1)
        self.y = 1 / (1 + math.exp(-dot(self.w, x)))
        return self.y

    def update(self, g):
        localg = self.y * (1 - self.y)
        #print localg, g
        back = []
        s = len(self.w)
        if self.bias:
            s -= 1
        for i in xrange(s):
            back.append(g * localg * self.w[i])
            #print self.x[i].y
            self.w[i] -= self.rate * localg * self.x[i].y * g
        if self.bias:
            self.w[-1] -= self.rate * localg * g
        #print self.w
        self.back = back

class DeepModel:

    def __init__(self, hiddenNum = 1, hiddenDim = 2, inputDim = 3, outputDim = 1):
        self.hs = []
        self.input = [DeepNode(1, None, bias = BIAS) for i in xrange(inputDim)]
        lastInp = self.input
        lastDim = inputDim
        for i in xrange(hiddenNum):
            l = []
            for j in xrange(hiddenDim):
                l.append(DeepNode(lastDim, lastInp, bias=BIAS))
            lastDim = hiddenDim
            lastInp = l
            self.hs.append(l)
        l = []
        for j in xrange(outputDim):
            l.append(DeepNode(lastDim, lastInp, bias=BIAS))
        self.hs.append(l)
        self.outputDim = outputDim
        self.g = [0] * outputDim

    def train(self, x, y):
        for i in xrange(len(x)):
            self.input[i].y = x[i]
        for layer in self.hs:
            x = [node.cal() for node in layer]
            #print x
        loss = 0
        for i in xrange(len(x)):
            loss += 0.5 * ((y[i] - x[i]) ** 2)
        #loss =  -log(x[0]) if y > 0 else -log(1-x[0])
        #print "loss : %f" % loss
        self.addG(y)
        return loss

    def addG(self, y):
        ly = self.hs[-1]
        for i in xrange(len(ly)):
            self.g[i] += ly[i].y - y[i]
        #print self.g


    def update(self, y):
        g = self.g
        for layer in self.hs[::-1]:
            gn = None
            #print g
            for i in xrange(len(layer)):
                node = layer[i]
                node.update(g[i])
                if gn == None:
                    gn = [0] * len(node.back)
                for ii in xrange(len(node.back)):
                    gn[ii] += node.back[ii]
            g = gn
        self.g = [0] * self.outputDim


    def pred(self, x):
        for i in xrange(len(x)):
            self.input[i].y = x[i]
        for layer in self.hs:
            x = [node.cal() for node in layer]
        return x


def learn(step, dm, dp):
    batch = 1
    count = 0
    for i in range(passes):
        dp = DataProvider("train.data")
        loss = 0.0
        for e in dp:
            loss += dm.train(e[1], e[0])
            if count % batch == 0:
                dm.update(e[0])
            count += 1
        #print "loss: %f" % (loss / count)
    #print "pred %s" % str(dm.pred([0, 0]))#, 1])
    print "pred %s" % str(dm.pred([0, 1]))#, 0]) # -2557.1
    print "pred %s" % str(dm.pred([0.5, 0.5]))#, 0]) # -2557.1
    print "pred %s" % str(dm.pred([1, 1]))#, 0]) # -2557.1
    print "pred %s" % str(dm.pred([0, 0]))#, 0]) # -2557.1
    print "pred %s" % str(dm.pred([1, 0]))#, 0])
    #print "pred %s" % str(dm.pred([1, 1]))#, 0])

    #print "pred %f" % dm.pred([0, 0, 1])
    #print "pred %f" % dm.pred([0, 1, 0])
    #print "pred %f" % dm.pred([1, 0, 0])

if __name__ == '__main__':
    passes = int(sys.argv[1])
    BIAS = int(sys.argv[2]) == 1
    learn(step = 1, dm = DeepModel(inputDim = 2), dp = DataProvider("train.data"))
