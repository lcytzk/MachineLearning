#!/usr/bin/env python

import math
import random
import time

def jdotj(a, b):
	if len(a) != len(b):
		print 'Sizes are not equal!'
		return
	res = 0
	for i in xrange(len(a)):
		res += a[i]*b[i] 
	return res

def ndotj(n, j):
	return [n*i for i in j]

def jaddj(a, b):
	return [a[i]+b[i] for i in xrange(len(a))]

class LinearReg:

	def __init__(self, size):
		self.size = size
		self.w = [0 for i in xrange(size)]
		self.stepSize = 0.1

	def getLoss(self, y, x):
		return y - jdotj(x, self.w)

	def getGradient(self, y, x):
		tmp = [0 for i in xrange(len(x))]
		for j in xrange(len(tmp)):
			tmp[j] += self.getLoss(y, x) * x[j] / len(x)
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(deltaW, self.w)

def getAbs(l):
	res = 0
	for i in l:
		res += i*i
	return math.sqrt(res)

class OGD:

	def __init__(self, loss, x, y, round):
		self.stop = -1
		self.round = round
		self.realRound = 0
		self.loss = loss
		self.x = x
		self.y = y

	def runARound(self, y, x):
		deltaG = self.loss.getGradient(y, x)
		self.loss.updateWeight(deltaG)
		return deltaG

	def learn(self):
		deltaG = self.runARound(self.y[0], self.x[0])
		while self.realRound < self.round and self.stop < getAbs(deltaG):
			deltaG = self.runARound(self.y[self.realRound], self.x[self.realRound])
			self.realRound += 1

class OGD2:

	def __init__(self, loss, x, y, round):
		self.stop = 0.001
		self.round = round
		self.realRound = 0
		self.loss = loss
		self.x = x
		self.y = y
		self.deltaG = None
		self.deltaGN = 1

	def runARound(self, y, x):
		deltaG = self.loss.getGradient(y, x)
		self.loss.updateWeight(deltaG)
		return deltaG

	def learn(self):
		self.deltaG = self.runARound(self.y, self.x)


def genRandomData(featureSize, sampleSize):
	y = []
	x = []
	w = []
	for i in xrange(featureSize):
		w.append(random.random())
	for i in xrange(sampleSize):
		tmp = []
		for j in xrange(featureSize):
			tmp.append(random.random())
		x.append(tmp)
		y.append(jdotj(tmp, w))
	return y,x,w

def runInOneTime(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	ogd = OGD(lg, x, y, sampleSize)
	ogd.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is\n%s" % (str(w))
	print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def runOneByOne(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	ogd = OGD2(lg, x, y, sampleSize)
	for i in xrange(sampleSize):
		ogd.x = x[i]
		ogd.y = y[i]
		ogd.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is\n%s" % (str(w))
	print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def test():
	featureSize = 40
	sampleSize = 20000
	y, x, w = genRandomData(featureSize, sampleSize)
	start = time.clock()
	#for i in range(10):
	runOneByOne(x, y, w, sampleSize, featureSize)
	end = time.clock()
	print "run one by one use: %s" % (end - start)
	start = end
	#for i in range(10):
	runInOneTime(x, y, w, sampleSize, featureSize)
	print "run in one time use: %s" % (time.clock() - start)

if __name__ == '__main__':
	test()