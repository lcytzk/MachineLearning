#!/usr/bin/env python

import math
import random

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
			tmp[j] += self.stepSize * self.getLoss(y, x) * x[j] / len(x)
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(deltaW, self.w)

def getAbs(l):
	res = 0
	for i in l:
		res += i*i
	t = math.sqrt(res)
	#print t
	return t

class OGD:

	def __init__(self, loss, x, y, round):
		self.stop = 0.00001
		self.round = round
		self.realRound = 1
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
		if self.realRound < self.round:
			print "Reach the gap: %f" % getAbs(deltaG)
		else:
			print "Run off round."


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

def test():
	featureSize = 2000
	sampleSize = 30000
	y, x, w = genRandomData(featureSize, sampleSize)
	lg = LinearReg(featureSize)
	ogd = OGD(lg, x, y, sampleSize)
	ogd.learn()
	#print gd.loss.w, gd.realRound
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	#print "true model is %s" % (str(w))
	#print "my model is   %s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

if __name__ == '__main__':
	test()