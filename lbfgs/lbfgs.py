#!/usr/bin/env python

import math
import random

def jdotj(a, b):
	res = 0
	for i in xrange(len(a)):
		res += a[i]*b[i] 
	return res

def ndotj(n, j):
	return [n*i for i in j]

def jaddj(a, b):
	return [a[i]+b[i] for i in xrange(len(a))]

def jminj(a, b):
	return [a[i]-b[i] for i in xrange(len(a))]

class LinearReg:

	def __init__(self, size):
		self.size = size
		self.w = [0 for i in xrange(size)]
		self.stepSize = 0.5

	def getLoss(self, y, x):
		return y - jdotj(self.w, x)

	def getGradient(self, y, x):
		tmp = [0 for i in xrange(len(x[0]))]
		for j in xrange(len(tmp)):
			tmpj = 0
			for i, y_i in enumerate(y):
				x_i = x[i]
				x_j_i = x[i][j]
				tmpj += self.getLoss(y_i, x_i) * x_j_i
			tmp[j] += self.stepSize * tmpj / len(y)
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(self.w, deltaW)

	def updateWeight2(self, w):
		self.w = w

def getAbs(l):
	res = 0
	for i in l:
		res += i*i
	return math.sqrt(res)

class LBFGS:

	def __init__(self, loss, x, y, round):
		self.stop = 0.0001
		self.round = round
		self.realRound = 1
		self.loss = loss
		self.x = x
		self.y = y
		self.h = []
		self.m = 10
		self.s = [] # si = xi - xi-1
		self.t = [] # ti = gi - gi-1
		self.H = 1
		self.lastGrad = [0 for i in xrange(len(self.x[0]))]
		self.grad = 0

	def runARound(self):
		direction = self.getDirection(self.lastGrad)
		thisw = jminj(self.loss.w, ndotj(0.1, direction))
		s = jminj(thisw, self.loss.w)
		self.loss.updateWeight2(thisw)
		self.grad = self.loss.getGradient(self.y, self.x)
		y = jminj(self.grad, self.lastGrad)
		self.updateST(s, y)
		self.H = jdotj(y, s) / jdotj(y, y)
		self.lastGrad = self.grad
		return self.grad

	def updateST(self, s, t):
		if len(self.s) >= self.m:
			del self.s[0]
			del self.t[0]
		self.s.append(s)
		self.t.append(t)
			

	def getDirection(self, q):
		# two-loop
		k = min(self.m, len(self.s))
		s = self.s
		t = self.t
		alpha = [0 for i in xrange(k)]
		rho = [0 for i in xrange(k)]
		for i in range(k)[::-1]:
			rho[i] = 1.0 / jdotj(s[i], t[i])
			alpha[i] = rho[i] * jdotj(s[i], q)
			q = jminj(q, ndotj(alpha[i], t[i]))
		r = ndotj(self.H, q)
		for i in xrange(k):
			beta = rho[i] * jdotj(t[i], r)
			r = jaddj(r, ndotj(alpha[i] - beta, s[i]))
		return r


	def learn(self):
		self.lastGrad = self.loss.getGradient(self.y, self.x)
		while True:
			grad = self.runARound()
			self.realRound += 1
			if self.realRound > self.round or self.stop > getAbs(grad):
				break
		if self.realRound < self.round:
			print "Reach the gap: %f" % getAbs(grad)
		else:
			print "Run off round."


def genRandomData(featureSize, sampleSize):
	y, x, w = [], [], []
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
	featureSize = 10
	sampleSize = 1000
	y, x, w = genRandomData(featureSize, sampleSize)
	lg = LinearReg(featureSize)
	bfgs = LBFGS(lg, x, y, sampleSize)
	bfgs.learn()
	#print gd.loss.w, gd.realRound
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is %s" % (str(w))
	print "my model is   %s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

if __name__ == '__main__':
	test()