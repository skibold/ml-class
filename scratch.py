import numpy as np

# entropy, conditional entropy
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
	
	
# information gain vs mutual information
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
	
	
# adam running average	
def adam_average(data, beta):
	xbar = [0]
	xhat = [0]
	for t in range(len(data)):
		xb = beta * xbar[t] + (1-beta)*data[t]
		xbar.append(xb)
		xh = xb / (1-pow(beta,t+1))
		xhat.append(xh)
	return xbar, xhat
	
	
# softmax and cross entropy
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
	
	
# gaussian naive bayes
def class_means(x,y):
	mu0 = np.zeros(2)
	len0 = len(np.where(y==0)[0])
	for i in np.where(y==0)[0]:
		mu0[0] += x[i][0]
		mu0[1] += x[i][1]
	mu0 /= len0
	h0 = len0 / len(y)
	
	mu1 = np.zeros(2)
	len1 = len(np.where(y==1)[0])
	for i in np.where(y==1)[0]:
		mu1[0] += x[i][0]
		mu1[1] += x[i][1]
	mu1 /= len1
	h1 = len1 / len(y)
	return mu0, mu1
	
def class_probs(y):
	len0 = len(np.where(y==0)[0])
	h0 = len0/len(y)
	len1 = len(np.where(y==1)[0])
	h1 = len1/len(y)
	return h0, h1
	
def gnb_case1(x, y, sigma, x_pred): # C1 = C2 = s^2 I
	mu0, mu1 = class_means(x,y)
	h0, h1 = class_probs(y)
	print(mu0, mu1)
	w = mu0 - mu1
	b = .5*(np.linalg.norm(mu1) - np.linalg.norm(mu0)) + sigma * (np.log(h0) - np.log(h1))
	print("{}*x0 + {}*x1 + {}".format(w[0], w[1], b))
	return w[0] * x_pred[0] + w[1] * x_pred[1] + b > 0
	
def gnb_case2(x, y, x_pred): # C1 = C2 = C
	h0, h1 = class_probs(y)
	mu0, mu1 = class_means(x,y)
	mu = x.sum(axis=0) / len(x)
	x_centered = x - mu
	cov = np.zeros([2,2])
	for xi in x_centered:
		cov += np.outer(xi,xi)
	cov /= len(y)
	print(cov)
	cov_inv = np.linalg.inv(cov)
	w = np.matmul(cov_inv, (mu0-mu1))
	b = .5*(np.matmul(np.matmul(mu1.transpose(),cov_inv),mu1) - np.matmul(np.matmul(mu0.transpose(),cov_inv),mu0)) + (np.log(h0)-np.log(h1))
	print("{}*x0 + {}*x1 + {}".format(w[0], w[1], b))
	return w[0] * x_pred[0] + w[1] * x_pred[1] + b > 0
	
def gnb_case3(x,y): # arbitrary C1, C2
	h0, h1 = class_probs(y)
	mu0, mu1 = class_means(x,y)
	
	cov0 = np.zeros([2,2])
	for i in np.where(y==0)[0]:
		x_centered = x[i] - mu0
		cov0 += np.outer(x_centered, x_centered)
	cov0 /= len(np.where(y==0)[0])
	c0_inv = np.linalg.inv(cov0)
	c0_det = np.linalg.det(cov0)
	
	cov1 = np.zeros([2,2])
	for i in np.where(y==1)[0]:
		x_centered = x[i] - mu1
		cov1 += np.outer(x_centered, x_centered)
	cov1 /= len(np.where(y==1)[0])
	c1_inv = np.linalg.inv(cov1)
	c1_det = np.linalg.det(cov1)
	
	b = np.log(c1_det) - np.log(c0_det) + 2*(np.log(h0) - np.log(h1))
	
	
	print(cov0, cov1)
	
	
	
		
