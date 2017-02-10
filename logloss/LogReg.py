#!/usr/bin/env python

import math
import random
import time

def jdotj(a, b):
	if len(a) != len(b):
		print 'Sizes are not equal!'
		return
	res = 0.0
	for i in xrange(len(a)):
		res += a[i]*b[i] 
	return res

def ndotj(n, j):
	return [n*i for i in j]

def jaddj(a, b):
	return [a[i]+b[i] for i in xrange(len(a))]

def getAbs(l):
	res = 0
	for i in l:
		res += i*i
	return math.sqrt(res)

class LogReg:

	def __init__(self, size):
		self.size = size
		self.w = [0 for i in xrange(size)]
		self.stepSize = 0.1

	def getLoss(self, y, x):
		return y - 1.0 / (1 + math.exp(0 - jdotj(x, self.w)))

	def getGradient(self, y, x):
		tmp = [0 for i in xrange(len(x))]
		for j in xrange(len(tmp)):
			tmp[j] += self.getLoss(y, x) * x[j] / len(x)
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(deltaW, self.w)

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

def genRandomData(featureSize, sampleSize):
	y = []
	x = []
	w = []
	for i in xrange(featureSize):
		w.append(random.random())
	for i in xrange(sampleSize):
		tmp = [random.random() for j in xrange(featureSize)]
		x.append(tmp)
		k = jdotj(tmp, w)
		y.append(1 if k > 0.22 else 0)
	return y,x,w

def genRandomData2(featureSize, sampleSize):
	y = []
	x = []
	w = [1.0 / featureSize for i in range(featureSize)]
#	for i in xrange(featureSize):
#		w.append(1 if random.random() > 0.5 else 0)
	for i in xrange(sampleSize):
		tmp = [1 if random.random() > 0.5 else 0 for j in xrange(featureSize)]
		x.append(tmp)
		k = jdotj(tmp, w)
		y.append(1 if k > 0.5 else 0)
	return y,x,w

def runInOneTime(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	ogd = OGD(lg, x, y, sampleSize)
	ogd.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	#print "true model is\n%s" % (str(w))
	#print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def runInOneTime2(x, y, w, sampleSize, featureSize):
	lg = LogReg(featureSize)
	ogd = OGD(lg, x, y, sampleSize)
	ogd.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	#print "true model is\n%s" % (str(w))
	#print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def test():
	featureSize = 10
	sampleSize = 20000
	y, x, w = genRandomData2(featureSize, sampleSize)
	start = time.clock()
	#for i in range(10):
	#runOneByOne(x, y, w, sampleSize, featureSize)
	#end = time.clock()
	#print "run one by one use: %s" % (end - start)
	#for i in range(10):
	runInOneTime(x, y, w, sampleSize, featureSize)
	runInOneTime2(x, y, w, sampleSize, featureSize)
	print "run in one time use: %s" % (time.clock() - start)

if __name__ == '__main__':
	test()