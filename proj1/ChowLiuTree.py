from Graph import Graph
import pandas as pd
import numpy as np
from math import gamma
#from MushroomsNaiveBayes import beta
A=1
B=1
class CondTable:
	def __init__(self, evidence, prior, beta_a=0, beta_b=0):
		# variable names
		self.ev = evidence 
		self.pr = prior
		self.table = dict()
		self.a = beta_a
		self.b = beta_b
		
	def beta(self, x):
		try:
			Beta_inv = gamma(self.a+self.b) / (gamma(self.a) * gamma(self.b))
			return Beta_inv * pow(x, (self.a-1)) * pow((1-x), (self.b-1))
		except:
			return 1
			
		
	def add(self, e_val, p_val, prob):
		key = "{}|{}".format(e_val, p_val)
		self.table[key] = prob
		
	def get(self, e_val, p_val):
		key = "{}|{}".format(e_val, p_val)
		prob = 0
		if key in self.table:
			prob = self.table[key]
		#print(prob, self.beta(prob))
		return self.beta(prob) * prob
		
	def __str__(self):
		return "CondTable for P({}|{})\n{}".format(self.ev, self.pr, self.table)

class MSTCluster:
	def __init__(self, id):
		self._id = id
		self._nodes = [id]
		self._edges = []

	@property
	def id(self):
		return self._id

	@property
	def nodes(self):
		return self._nodes

	@property
	def edges(self):
		return self._edges

	def hasNode(self, id):
		return id in self._nodes

	def merge(self, cluster, edge):
		if self._id < cluster.id:
			self._id = cluster.id
		self._nodes.extend(cluster.nodes)
		self._edges.extend(cluster.edges)
		self._edges.append(edge)

	def mwoe(self, graph):
		heaviest = -1000000
		heaviest_idx = None
		heaviest_nbr = None
		for n in self.nodes:
			for nbr in graph.neighbors(n):
				if self.hasNode(nbr):
					continue
				if [n, nbr] in graph.edges:
					idx = graph.edges.index([n, nbr])
					if graph.edgeWeights[idx] > heaviest:
						heaviest = graph.edgeWeights[idx]
						heaviest_idx = idx
						heaviest_nbr = nbr
		return heaviest, heaviest_idx, heaviest_nbr

	def __str__(self):
		return "MSTCluster {}: edges {}:".format(self._id, self._edges)



class ChowLiuTree(Graph):
	def __init__(self, trainingFile):
		self.df = pd.read_csv(trainingFile)
		# construct a complete graph
		numNodes = len(self.df.columns)
		A = []
		for i in range(numNodes):
			a = np.ones(numNodes)
			a[i] = 0
			A.append(a)
		super().__init__(A=A)

		# assign edge weights and sort edges by weight
		self._condTables = [] # there will be two per edge in the same order as self.edges
		self._marginals = [dict() for _ in self.nodes] # there will be one per node	
		self.edgeWeights = self.weights()
		#self._edges = [self.edges[e] for e in np.argsort(self.edgeWeights)]
		#self.edgeWeights = sorted(self.edgeWeights)

		# maximum weight spanning tree
		mstEdges = self.mst().edges
		mstWeights = []
		mstCondTables = []
		for e in mstEdges:
			eidx = self._edges.index(e)
			mstWeights.append(self.edgeWeights[eidx])
			mstCondTables.append(self._condTables[eidx])
			
		# delete all unused edges and weights
		self.edgeWeights = mstWeights
		self._condTables = mstCondTables
		self._edges.clear()
		self._neighbors.clear()
		self.add_edges(mstEdges)
		
		# setup priors
		self.theta = []
		self.visit_neighbors(None, 0, 0)
		for m in self._marginals[0].values():
			self.theta.append([m])
		mu = []
		for _ in range(np.shape(self.theta)[0]):
			mu.append([.5])
		mu = np.array(mu)
		self.theta = np.array(self.theta)
		sigma = np.identity(np.shape(mu)[0])
		variance = 1
		sigma *= variance
		self.prior = np.exp(-.5 * np.matmul(np.transpose(self.theta-mu), np.matmul(np.linalg.inv(sigma), (self.theta-mu))))
		#print(len(self.theta), self.theta)
		#print(self.prior)
		
	def __str__(self):
		rv = ""
		for n in self.nodes:
			rv += "{} : Marginal probabilities {}\n".format(self.df.columns[n], self._marginals[n])
		for e in self._edges:
			idx = self._edges.index(e)
			rv += "[{}, {}] : {}\n{}\n{}\n\n".format(self.df.columns[e[0]], self.df.columns[e[1]],
			self.edgeWeights[idx], self._condTables[idx][0], self._condTables[idx][1])
		return rv
		
	def mst(self):
		clusters = dict()
		for n in self.nodes:
			clusters[n] = MSTCluster(n)

		round =0
		while True:
			clusterKeys = list(clusters.keys()) # avoid concurrent modification
			if len(clusterKeys) == 1:
				return clusters[clusterKeys[0]]
			#print("round {}: {} {}\n".format(round, len(clusterKeys), clusterKeys))
			for k in clusterKeys:
				if k in clusters:
					c1 = clusters[k]
					(weight, edge_idx, nbr) = c1.mwoe(self)
					#print("{} has mwoe {} {} {}".format(c1, weight, edge_idx, nbr))
					for k2, c2 in clusters.items():
						if c2.id != c1.id and c2.hasNode(nbr):
							clusters.pop(k)
							clusters.pop(k2)
							log="merged {} {}".format(c1.id, c2.id)
							c1.merge(c2, self.edges[edge_idx])
							#print("{} to yield {}\n".format(log, c1))
							clusters[c1.id] = c1
							break
			round += 1

	def weights(self):
		eWeights = []
		for e in self.edges:
			print("mutualInfo({})".format(e))
			eWeights.append(self.mutualInfo(e))
		return eWeights
			
	def mutualInfo(self, edge):
		tables = [CondTable(edge[0],edge[1],A,B), CondTable(edge[1],edge[0],A,B)]
		total = len(self.df)
		col0 = self.df.columns[edge[0]]
		col1 = self.df.columns[edge[1]]
		vals0 = self.df[col0].unique()
		vals1 = self.df[col1].unique()
		sumMutual = 0
		for v0 in vals0:
			v0truth = self.df[col0] == v0
			Nv0 = len(self.df[v0truth])
			Pv0 = Nv0 / total
			self._marginals[edge[0]][v0] = Pv0
			
			for v1 in vals1:
				v1truth = self.df[col1] == v1
				Nv1 = len(self.df[v1truth])
				Pv1 = Nv1 / total
				self._marginals[edge[1]][v1] = Pv1
				
				Nmutual = len(self.df[v0truth][v1truth])
				Pmutual = Nmutual / total
				
				tables[1].add(v1, v0, Nmutual / Nv0)
				tables[0].add(v0, v1, Nmutual / Nv1)
				
				if Pmutual == 0: # avoid log 0
					continue # no mutual info for this val
				sumMutual += (Pmutual * (np.log(Pmutual/(Pv0*Pv1)) / np.log(2)))

		#print(sumMutual, beta(sumMutual, 5, 2))
				
		self._condTables.append(tables)
		return sumMutual
		
	def predict(self, testset):
		test = pd.read_csv(testset)
		true_pos = 0
		fals_pos = 0
		true_neg = 0
		fals_neg = 0

		#print(self._marginals[0]['e'] , beta(self._marginals[0]['e'], 5, 2))  # p(e)
		#print(self._marginals[0]['p'] , beta(self._marginals[0]['p'], 5, 2))  # p(p)

		for _, row in test.iterrows():
			p_e, p_p = self.predict_label(row)
			
			if p_e > p_p:  # predicted edible
				#print('e', row[0])
				if row[0] == 'e':
					true_pos += 1
				else:
					fals_pos += 1
			else:  # predicted poison
				#print('p', row[0])
				if row[0] == 'p':
					true_neg += 1
				else:
					fals_neg += 1

		return true_pos, fals_pos, true_neg, fals_neg

	def predict_label(self, sample):
		root = 0
		cond = self.infer(None, root, sample)
		p_e = self._marginals[root]['e'] * cond # p(e)
		p_p = self._marginals[root]['p'] * cond # p(p)
		for n in self.neighbors(root):
			condTable = self.get_cond_table(root, n)
			p_e *= condTable.get(sample[n], 'e')
			p_p *= condTable.get(sample[n], 'p')
			
		#print("P(0=e) = {}, joint = {}".format(self._marginals[root]['e'], p_e))
		#print("P(0=p) = {}, joint = {}".format(self._marginals[root]['p'], p_p))

		return p_e, p_p
		
	def infer(self, parent, n, sample):
		prob = 1
		
		for nbr in self.neighbors(n):
			if nbr != parent:
				prob *= self.infer(n, nbr, sample)
				if parent is not None: # otherwise I'm the root and don't want compute conditionals of child | root since idk what root is; this prediction afterall
					condTable = self.get_cond_table(n, nbr)
					condProb = condTable.get(sample[nbr], sample[n])
					prob *= condProb
					#print(condTable)
					#print("P({}={} | {}={}) = {}".format(nbr, sample[nbr], n, sample[n], condProb))
		
		return prob

	def get_cond_table(self, parent, child):
		edge = [parent, child]
		if edge in self.edges:
			idx = self.edges.index(edge)
			return self._condTables[idx][1]
		else:
			edge = [child, parent]
			idx = self.edges.index(edge)
			return self._condTables[idx][0]

	def visit_neighbors(self, parent, n, depth):
		if parent is None:
			print("DFS rooted at {}".format(n))
		print("{}->\t{}".format((depth)*"\t", n)) 
		for nbr in self.neighbors(n):
			if nbr != parent:
				condTable = self.get_cond_table(n, nbr)
				for c in condTable.table.values():
					self.theta.append([c])
				self.visit_neighbors(n,nbr,depth+1)
		
if __name__ == "__main__":
	clt = ChowLiuTree("full_data.csv")
	print(clt.edges)
	print(clt)
	clt.visit_neighbors(None, 0, 0)
	#tp, fp, tn, fn = clt.predict("mushroom_test.csv")
	#print("\ntrue_pos:{} fals_pos:{} true_neg:{} fals_neg:{}".format(tp, fp, tn, fn))
	#print((tp + tn) / (tp + fp + tn + fn))
