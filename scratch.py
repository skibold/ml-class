import numpy as np

def cond_prob(c1, c2, x, df): # P(c1 | c2=x)
	return prob(c1, df[df[c2]==x])
	
def joint_prob(c1, x1, c2, x2, df): # P(c1=x1, c2=x2)
	return len(df[df[c1]==x1][df[c2]==x2]) / len(df)

def prob(col, df):
	return np.array([len(df[df[col]==u]) / len(df) for u in df[col].unique()])

def log2(n):
	return np.log(n) / np.log(2)
	
def h_vec(vec): # entropy of the 1d vector
	return sum([p * log2(1/p) for p in vec if p > 0])
	
def cross_entropy(p, q):
	assert len(p) == len(q)
	return sum([p[i] * log2(1/q[i]) for i in range(len(p)) if q[i] > 0])
	
def entropy(col, df):
	return h_vec(prob(col,df))
	
def cond_entropy(c1, c2, df):
	pc2 = prob(c2, df)
	p_c1c2 = [cond_prob(c1,c2,x,df) for x in df[c2].unique()]
	h_c1c2 = [h_vec(p) for p in p_c1c2]
	return np.dot(pc2, h_c1c2)
	
def gain(c1, c2, df):
	return entropy(c1, df) - cond_entropy(c1, c2, df)
	
def mutual_info(c1, c2, df):
	joints = np.array([joint_prob(c1,x1,c2,x2,df) for x1 in df[c1].unique() for x2 in df[c2].unique()])
	marginals = np.array([p1*p2 for p1 in prob(c1,df) for p2 in prob(c2,df)])
	ind = joints/marginals
	return sum([joints[i] * log2(ind[i]) for i in range(len(ind)) if ind[i] != 0])
	
def compare(target, df):
	others = [c for c in df.columns if c != target]
	for c2 in others:
		print("{} Gain = {}, I = {}".format(c2, gain(target,c2,df), mutual_info(target,c2,df)))
		
def adam_average(data, beta):
	xbar = [0]
	xhat = [0]
	for t in range(len(data)):
		xb = beta * xbar[t] + (1-beta)*data[t]
		xbar.append(xb)
		xh = xb / (1-pow(beta,t+1))
		xhat.append(xh)
	return xbar, xhat
	
def softmax(data):
	exps = np.exp(data)
	return exps / sum(exps)
	
def ce_loss(data, j):
	sm = softmax(data)
	return np.log(1/sm[j])
	
def ce_gradient(data, j):
	sm = softmax(data)
	def ind(i,j):
		if i==j:
			return 1
		return 0
	return [sm[i] - ind(i,j) for i in range(len(sm))]
	
		