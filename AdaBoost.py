import numpy as np

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
		self.alpha = 0
		self.error = 1
		self.z = 0
		self.comparator = None
		self.thresh = None	
		self.c_str = ''

	def test_func(self, func, means, mapback, s_feat, labels):
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
		wrongs = None
		for s,c in COMPARATORS.items():
			e,t,w = self.test_func(c, means, mapback, s_feat, labels)
			if e < self.error:
				self.error = e
				self.thresh = t
				self.comparator = c
				self.c_str = s
				wrongs = w
		
		self.alpha = .5 * np.log((1-self.error)/self.error)
		q_array = [np.exp(self.alpha * wrongs[mapback[i]]) for i in range(len(wrongs))]
		self.z = np.dot(self.p_array, q_array)
		self.p_array = np.multiply(self.p_array, q_array) / self.z
		
	def eval(self, x):
		if self.comparator.__call__(x, self.thresh):
			return self.alpha
		else:
			return -self.alpha
			
	def __str__(self):
		return "h{}: x{}{}, error: {}, alpha: {}, Z: {}, probabilities: {}".format(self.iter, self.c_str, self.thresh, self.error, self.alpha, self.z, self.p_array)
		
		
class AdaBoost:
	def __init__(self, T, features, labels):
		self.classifiers = []
		self.features = features
		self.labels = labels
		self.T = T
		
	def train(self):
		iter = 1
		p_array = [1/len(self.features)]*len(self.features)
		while iter <= self.T:
			self.classifiers.append(WeakClassifier(p_array, iter))
			self.classifiers[-1].train(self.features, self.labels)
			print("\niter {}\nweak classifier: {}\nboosted classifier: {}".format(iter, str(self.classifiers[-1]), str(self)))
			p_array = self.classifiers[-1].p_array
			iter += 1
		
	def eval(self, x):
		pred = 0
		for c in self.classifiers:
			pred += c.eval(x)
		return pred
			
	def __str__(self):
		rv = ""
		for c in self.classifiers:
			rv += "{}*I(x{}{}) + ".format(c.alpha, c.c_str, c.thresh)
		return rv[:-3] # chop off that last +
		
		
		
