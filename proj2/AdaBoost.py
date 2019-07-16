import numpy as np
import json
from sys import argv

def lt(x, y):
	return x < y
	
def le(x, y):
	return x <= y
	
def gt(x, y):
	return x > y
	
def ge(x, y):
	return x >= y
	
def eq(x, y):
	return x == y
	
COMPARATORS = {'<':lt, '>':gt}

class WeakClassifier:
	def __init__(self, p_array, iter):
		self.p_array = p_array
		self.iter = iter
		self.name = "h%d" % iter
		self.alpha = 0
		self.error = 1
		self.z = 0
		self.comparator = None
		self.thresh = None	
		self.c_str = ''

	def test_func(self, func, means, mapback, s_feat, labels):
		# wrongs, errors, means, s_feat are all in the same order
		errors = []
		wrongs = []
		for m in means:
			e = 0
			w = np.ones(len(s_feat)) * -1 # no errors
			for i,f in enumerate(s_feat):
				pred = -1 # assume we got it wrong
				if func.__call__(f, m):
					pred = 1 # actually got it right
				if pred != labels[mapback[i]]: # indeed got it wrong
					w[i] = 1 # mark the error
					e += self.p_array[mapback[i]] # sum the error
			errors.append(e)
			wrongs.append(w)
			
		least = np.argmin(errors)
		return errors[least], means[least], wrongs[least]
		
	def train(self, features, labels):
		# pick the function and thresh that minimize error
		mapback = np.argsort(features)
		s_feat = sorted(features)
		means = [s_feat[0]-1]
		means.extend(np.mean(s_feat[i:i+2]) for i in range(len(s_feat)-1))
		means.append(s_feat[-1]+1)
		
		self.error = 1 # the highest it could be
		wrongs = None # same order as s_feat
		for s,c in COMPARATORS.items():
			e,t,w = self.test_func(c, means, mapback, s_feat, labels)
			if e < self.error:
				self.error = e
				self.thresh = t
				self.comparator = c
				self.c_str = s
				wrongs = w
		
		# lim x->0 alpha = infinity
		if self.error == 0:
			self.alpha = 10 # close enough to infinity
		else:
			self.alpha = .5 * np.log((1-self.error)/self.error)
		q_array = [np.exp(self.alpha * wrongs[mapback[i]]) for i in range(len(wrongs))]
		self.z = np.dot(self.p_array, q_array)
		self.p_array = np.multiply(self.p_array, q_array) / self.z
		
	def eval(self, x):
		if self.comparator.__call__(x, self.thresh):
			return self.alpha
		else:
			return -self.alpha
			
	def as_dict(self):
		return {self.name:"x{}{}".format(self.c_str,self.thresh), 
				"error":self.error, 
				"alpha":self.alpha, 
				"Z":self.z, 
				"p_i":list(self.p_array)}
			
	def __str__(self):
		return json.dumps(self.as_dict(), indent=2)
		
		
class AdaBoost:
	def __init__(self, T, features, labels, initial_p=None):
		self.classifiers = []
		self.features = features
		self.labels = labels
		self.T = T
		self.initial_p = [1/len(self.features)]*len(self.features) # uniform
		if initial_p is not None:
			self.initial_p = initial_p
		
	def train(self):
		iter = 1
		self.p_array = self.initial_p
		self.classifiers.clear()
		while iter <= self.T:
			self.classifiers.append(WeakClassifier(self.p_array, iter))
			self.classifiers[-1].train(self.features, self.labels)
			print("\niter {}\nweak classifier: {}\nboosted classifier: {}\nboosted error: {}\nboosted bound: {}"
				.format(iter, str(self.classifiers[-1]), str(self), self.error, self.bound))
			self.p_array = self.classifiers[-1].p_array
			iter += 1

	@property
	def error(self):
		return sum([self.p_array[i] for i in range(len(self.features)) if self.eval(self.features[i]) != self.labels[i]])

	@property
	def bound(self):
		return np.prod([c.z for c in self.classifiers])

	def eval(self, x):
		return np.sign(sum([c.eval(x) for c in self.classifiers]))
			
	def __str__(self):
		rv = ""
		for c in self.classifiers:
			rv += "{}*I(x{}{}) + ".format(c.alpha, c.c_str, c.thresh)
		return rv[:-3] # chop off that last +

def factory(fname):
	T = 0
	n = 0
	features = []
	labels = []
	p_array = None
	try:
		with open(fname) as fin:
			line = fin.readline().strip().split()
			T = int(line[0])
			n = int(line[1])
			features = np.array(fin.readline().strip().split()[:n+1], dtype=np.float)
			labels = np.array(fin.readline().strip().split()[:n+1], dtype=np.int)
			p_array = np.array(fin.readline().strip().split()[:n+1], dtype=np.float)
	except:
		print("Error: input format error ({}) {} {} {} {} {}".format(fname, T, n, features, labels, p_array))
	return AdaBoost(T, features, labels, p_array)

if __name__ == "__main__":
	if len(argv) != 2:
		print("Usage: python AdaBoost.py <input file>")
		exit(1)

	ab = factory(argv[1])	
	ab.train()
	
