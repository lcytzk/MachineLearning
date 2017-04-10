#!/usr/bin/env python

from numpy import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./', one_hot=True)

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
        return (float(label), array(map(float, rest.split())))

def loss(y, y_):
    l1 = [math.exp(k) for k in y_]
    base = sum(l1)
    l2 = [k / base for k in l1]
    l3 = [-math.log(k) for k in l2]
    l2 -= y
    return sum(l3), l2

def predict(w, b):
    xs = mnist.test.images
    ys = mnist.test.labels
    count = 0
    res = 0
    for x in xs:
        y_ = dot(x, w) + b
        res += 1 if argmax(ys[count]) == argmax(y_) else 0
        count += 1
    print res * 1.0 / count


def test():
    # defination
    w = zeros(7840).reshape(784, 10)
    b = zeros(10)
    step = 0.5

    for i in xrange(10):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        #dp = DataProvider('train.data')
        los = 0
        count = 0
        for x in batch_xs:
            y_ = dot(x, w) + b
            l, g = loss(batch_ys[count], y_)
            los += l
            cnt = 0
            for w_ in w.T:
                w_ -= step * g[cnt] * x
                cnt += 1
            b -= step * g
            count += 1
        print los / count
    print w, b
    predict(w, b)
    
if __name__ == '__main__':
    test()
