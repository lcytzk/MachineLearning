#!/usr/bin/env python

import math
import random
import time

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
		#self.stepSize = 0.5

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
			tmp[j] = tmpj / len(y)
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(self.w, deltaW)

	def updateWeight2(self, w):
		self.w = w

class LogReg:

	def __init__(self, size):
		self.size = size
		self.w = [0 for i in xrange(size)]
		#self.stepSize = 0.5

	def getVal(self, x):
		return 1 if 1.0 / (1 + math.exp(0 - jdotj(x, self.w))) > 0.5 else 0 

	def getLoss(self, y, x):
		try:
			t = jdotj(x, self.w)
			return y - 1.0 / (1 + math.exp(0 - t))
			#return y - jdotj(x, self.w)
		except Exception as e:
			print t
			raise e

	def getGradient(self, y, x):
		tmp = [0 for i in xrange(len(x[0]))]
		for j in xrange(len(tmp)):
			tmpj = 0
			for i, y_i in enumerate(y):
				x_i = x[i]
				x_j_i = x[i][j]
				tmpj += self.getLoss(y_i, x_i) * x_j_i
			tmp[j] = tmpj / len(y)
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
		self.stop = 0.001
		self.round = round
		self.loss = loss
		self.x = x
		self.y = y
		self.h = []
		self.m = 10
		self.s = [] # si = xi - xi-1
		self.t = [] # ti = gi - gi-1
		self.H = 1
		self.grad = 0
		self.stepSize = 0.1
		self.alpha = [0 for i in xrange(self.m)]
		self.rho = [0 for i in xrange(self.m)]

	def runARound(self):
		direction = self.getDirection(self.lastGrad)
		thisw = jminj(self.loss.w, ndotj(self.stepSize, direction))
		self.loss.updateWeight2(thisw)
		s = ndotj(0 - self.stepSize, direction)
		self.grad = self.loss.getGradient(self.y, self.x)
		y = jminj(self.grad, self.lastGrad)
		self.updateST(s, y)
		y_y = jdotj(y, y)
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
		for i in range(k)[::-1]:
			s_t = jdotj(s[i], t[i])
			self.rho[i] = 1.0 / s_t
			self.alpha[i] = self.rho[i] * jdotj(s[i], q)
			q = jminj(q, ndotj(self.alpha[i], t[i]))
		r = ndotj(self.H, q)
		for i in xrange(k):
			beta = self.rho[i] * jdotj(t[i], r)
			r = jaddj(r, ndotj(self.alpha[i] - beta, s[i]))
		return r


	def learn(self):
		self.lastGrad = self.loss.getGradient(self.y, self.x)
		realRound = 1
		while True:
			grad = self.runARound()
			realRound += 1
			if realRound > self.round or self.stop > getAbs(grad):
				break
		#if realRound < self.round:
		#	print "Reach the gap: %f" % getAbs(grad)
		#	print "Run %d rounds." % realRound
		#else:
		#	print "Run off round."

class LBFGS2:

	def __init__(self, loss, x, y, round):
		self.stop = 0.001
		self.round = round
		self.loss = loss
		self.x = x
		self.y = y
		self.h = []
		self.m = 10
		self.s = [] # si = xi - xi-1
		self.t = [] # ti = gi - gi-1
		self.H = 1
		self.lastGrad = [0 for i in xrange(len(self.x[0]))]
		#self.lastGrad = self.loss.getGradient(self.y, self.x)
		self.grad = 0
		self.stepSize = 0.1
		self.alpha = [0 for i in xrange(self.m)]
		self.rho = [0 for i in xrange(self.m)]
		self.gradNorm = 1

	def runARound(self):
		direction = self.getDirection(self.lastGrad)
		thisw = jminj(self.loss.w, ndotj(self.stepSize, direction))
		self.loss.updateWeight2(thisw)
		#s = jminj(thisw, self.loss.w)
		s = ndotj(0 - self.stepSize, direction)
		self.grad = self.loss.getGradient(self.y, self.x)
		y = jminj(self.grad, self.lastGrad)
		self.updateST(s, y)
		y_y = jdotj(y, y)
		if y_y == 0:
			return self.grad
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
		for i in range(k)[::-1]:
			s_t = jdotj(s[i], t[i])
			self.rho[i] = 1.0 / s_t
			self.alpha[i] = self.rho[i] * jdotj(s[i], q)
			q = jminj(q, ndotj(self.alpha[i], t[i]))
		r = ndotj(self.H, q)
		for i in xrange(k):
			beta = self.rho[i] * jdotj(t[i], r)
			r = jaddj(r, ndotj(self.alpha[i] - beta, s[i]))
		return r

	def learn(self):
		self.lastGrad = self.loss.getGradient(self.y, self.x)
		self.lastGrad = self.runARound()

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
		y.append(1 if k > 0.7 else 0)
	return y,x,w

def runInOneTime(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	bfgs = LBFGS(lg, x, y, sampleSize)
	bfgs.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is\n%s" % (str(w))
	print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def runInOneTime2(x, y, w, sampleSize, featureSize):
	lg = LogReg(featureSize)
	bfgs = LBFGS(lg, x, y, sampleSize)
	bfgs.learn()
	allLoss = 0
	count = 0
	for i, x_i in enumerate(x):
		tmp = lg.getVal(x_i)
		loss = y[i] - tmp
		count += 1 if y[i] == tmp else 0
		allLoss += loss
	#print "true model is\n%s" % (str(w))
	#print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)
	print "count: %f" % (count*1.0/sampleSize)

def runOneByOne(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	bfgs = LBFGS(lg, x, y, sampleSize)
	for i in xrange(sampleSize):
		bfgs.x = [x[i]]
		bfgs.y = [y[i]]
		bfgs.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is\n%s" % (str(w))
	print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def runOneByOne2(x, y, w, sampleSize, featureSize):
	lg = LinearReg(featureSize)
	bfgs = LBFGS2(lg, x, y, sampleSize)
	for i in xrange(sampleSize):
		bfgs.x = [x[i]]
		bfgs.y = [y[i]]
		bfgs.learn()
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is\n%s" % (str(w))
	print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

def runOneByOne3(x, y, w, sampleSize, featureSize):
	lg = LogReg(featureSize)
	bfgs = LBFGS2(lg, x, y, sampleSize)
	for i in xrange(sampleSize):
		bfgs.x = [x[i]]
		bfgs.y = [y[i]]
		bfgs.learn()
	allLoss = 0
	count = 0
	for i, x_i in enumerate(x):
		tmp = lg.getVal(x_i)
		loss = y[i] - tmp
		count += 1 if y[i] == tmp else 0
		allLoss += loss
	#print "true model is\n%s" % (str(w))
	#print "my model is\n%s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)
	print "count: %f" % (count*1.0/sampleSize)

def test():
	featureSize = 10
	sampleSize = 2000
	y, x, w = genRandomData2(featureSize, sampleSize)
	start = time.clock()
	#for i in range(10):
	#runInOneTime2(x, y, w, sampleSize, featureSize)
	#end = time.clock()
	#print "#1# run in one time use: %s" % (end - start)
	#start = end
	runOneByOne3(x, y, w, sampleSize, featureSize)
	end = time.clock()
	print "#2# run one by one use: %s" % (end - start)
	#start = end
	#for i in range(10):
	#runInOneTime(x, y, w, sampleSize, featureSize)
	#end = time.clock()
	#print "run in one time use: %s" % (end - start)

if __name__ == '__main__':
	test()