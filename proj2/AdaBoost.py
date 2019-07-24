import numpy as np
import json
from sys import argv

PREDICTIONS = [-1, 1] # corresponds to [False, True]

def lt(x, y):
	return PREDICTIONS[int(x < y)]
	
def le(x, y):
	return PREDICTIONS[int(x <= y)]
	
def gt(x, y):
	return PREDICTIONS[int(x > y)]
	
def ge(x, y):
	return PREDICTIONS[int(x >= y)]
	
def eq(x, y):
	return PREDICTIONS[int(x == y)]
	
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

	def test_func(self, func, means, features, labels):
		errors = []
		wrongs = []
		for m in means:
			e = 0
			w = np.ones(len(features), dtype=np.int) * -1 # no errors
			for i,f in enumerate(features):
				if func.__call__(f, m) != labels[i]: # predicted wrong
					w[i] = 1 # mark the error
					e += self.p_array[i] # sum the error
			errors.append(e)
			wrongs.append(w)
			
		least = np.argmin(errors)
		return errors[least], means[least], wrongs[least]
		
	def train(self, features, labels): # assume features are sorted ascending and labels correspond
		# pick the function and thresh that minimize error
		means = [features[0]-1]
		means.extend(np.mean(features[i:i+2]) for i in range(len(features)-1))
		means.append(features[-1]+1)
		
		self.error = 1 # the highest it could be
		wrongs = None
		for s,c in COMPARATORS.items():
			e,t,w = self.test_func(c, means, features, labels)
			if e < self.error:
				self.error = e
				self.thresh = t
				self.comparator = c
				self.c_str = s
				wrongs = w
		
		# lim x->0 (alpha) = infinity
		if self.error == 0:
			self.alpha = 10 # close enough to infinity
		else:
			self.alpha = .5 * np.log((1-self.error)/self.error)
		q_array = [np.exp(self.alpha * w) for w in wrongs]
		self.z = np.dot(self.p_array, q_array)
		self.p_array = np.multiply(self.p_array, q_array) / self.z
		
	def eval(self, x):
		return self.comparator.__call__(x, self.thresh) * self.alpha
			
	def as_dict(self):
		return {self.name:"I(x{}{})".format(self.c_str,self.thresh), 
				"error":self.error, 
				"alpha":self.alpha, 
				"Z":self.z, 
				"p":list(self.p_array)}
			
	def __str__(self):
		d = self.as_dict()
		s = "The selected weak classifier {}:{}\n".format(self.name, d[self.name])
		s += "The error of {}: {}\n".format(self.name, d['error'])
		s += "The weight of {}: {}\n".format(self.name, d['alpha'])
		s += "The probabilities normalization factor Z{}: {}\n".format(self.iter, d['Z'])
		s += "The probabilities normalized: "
		for p in d['p']:
			s += "|{}".format(p)
		s += "|"
		return s #json.dumps(self.as_dict(), indent=2)
		
		
class AdaBoost:
	def __init__(self, T, features, labels, initial_p=None):
		self.classifiers = []
		mapback = np.argsort(features) # sort labels and features
		self.features = np.array(sorted(features), dtype=np.float)
		self.labels = np.array([labels[mapback[i]] for i in range(len(labels))], dtype=np.int)
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
			bc = "|"
			for x in self.features:
				bc += "{}|".format(self.eval(x))
			print("\n---Iteration {}----\n{}\nThe boosted classifier: {}\nThe boosted error: {}\nThe error bound: {}"
				.format(iter, str(self.classifiers[-1]), bc, self.error, self.bound))
			self.p_array = self.classifiers[-1].p_array
			iter += 1

	@property
	def error(self):
		return sum([1 for i in range(len(self.features)) if self.predict(self.features[i]) != self.labels[i]]) / len(self.labels)

	@property
	def bound(self):
		return np.prod([c.z for c in self.classifiers])

	def predict(self, x):
		return np.sign(self.eval(x))

	def eval(self, x):
		return sum([c.eval(x) for c in self.classifiers])
			
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
	
