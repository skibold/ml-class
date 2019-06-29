import json
from json.decoder import JSONDecodeError
import numpy as np

class Graph:
	def __init__(self, gDefFile=None, A=None):
		self._nodes = []
		self._edges = []
		self._neighbors = dict()
		if A is not None:
			self.loadMatrix(A)
		else:
			try:
				self.loadjson(gDefFile)
			except JSONDecodeError:
				self.loadtxt(gDefFile)
			
	def loadjson(self, gDefFile):
		with open(gDefFile,'r') as fin:
			j = json.load(fin)

		self._nodes = j['nodes']
		self._edges = j['edges']
		for e in j['edges']:
			for n in e:
				if n not in self._neighbors:
					self._neighbors[n] = set()
			self._neighbors[e[0]].add(e[1])
			self._neighbors[e[1]].add(e[0])

	def loadtxt(self, gDefFile):
		edgeset = []
		with open(gDefFile,'r') as fin:
			line = fin.readline().rstrip()
			self._nodes = line.split(',')
			line = fin.readline().rstrip()
			nidx=0
			
			while len(line) and nidx < len(self._nodes):
				curr_node = self._nodes[nidx]
				self._neighbors[curr_node] = set()
				edges = line.split(',')
				for i,e in enumerate(edges):
					if e == '1':
						neighbor = self._nodes[i]
						self._neighbors[curr_node].add(neighbor)
						newedge = set((curr_node,neighbor))
						if newedge not in edgeset:
							edgeset.append(newedge)
				line = fin.readline().rstrip()
				nidx+=1
		self._edges = [list(e) for e in edgeset]

	def savejson(self, gDefFile):
		gj = dict()
		gj['nodes'] = self._nodes
		gj['edges'] = self._edges
		with open(gDefFile,'w') as fout:
			json.dump(gj,fout)

	def savetext(self, gDefFile):
		indexdict = dict()
		for i,n in enumerate(self._nodes):
			indexdict[n]=i
		
		with open(gDefFile,'w') as fout:
			fout.write("%s\n" % ','.join(self._nodes))
			for i in self._nodes:
				row = np.zeros(len(self._nodes), dtype=np.int)
				for j in self.neighbors(i):
					jdx = indexdict[j]
					row[jdx] = 1
				fout.write("%s\n" % ','.join(row.astype(np.str).tolist()))

	def loadMatrix(self, A):
		edgeset = []
		self._nodes = np.arange(len(A[0])).tolist()
		nidx=0
		for row in A:
			curr_node = self._nodes[nidx]
			self._neighbors[curr_node] = set()
			for i,e in enumerate(row):
				if e == 1:
					neighbor = self._nodes[i]
					self._neighbors[curr_node].add(neighbor)
					newedge = set((curr_node,neighbor))
					if newedge not in edgeset:
						edgeset.append(newedge)
			nidx+=1
		self._edges = [list(e) for e in edgeset]
			
	@property
	def nodes(self):
		return self._nodes

	@property
	def edges(self):
		return self._edges
		
	def node_idx(self, n):
		if n in self._nodes:
			return self._nodes.index(n)
		else:
			return None

	def node_at(self, idx):
		if idx < len(self._nodes):
			return self._nodes[idx]
		else:
			return None

	def __eq__(self, other):
		return np.sort(self.nodes).tolist() == np.sort(other.nodes).tolist() and\
			np.sort(self.edges).tolist() == np.sort(other.edges).tolist()

	def __ne__(self, other):
		return not self.__eq__(other)

	def neighbors(self, node):
		if node not in self._nodes:
			return None # node doesn't exist
		elif node not in self._neighbors:
			return list() # no neighbors
		else:
			return list(self._neighbors[node])

	def remove_node(self, node):
		if node not in self._nodes:
			return None

		# remove node from the neighbor dict, returns all of node's abandoned neighbors
		abandoned = list(self._neighbors.pop(node))
		# remove node from the neighbor list of all node's abandoned neighbors
		for n in abandoned:
			self._neighbors[n].remove(node)
		
		# remove any edge containing node
		reduced = [e for e in self._edges if node not in e]
		self._edges = reduced
		
		# finally remove node from node list
		self._nodes.remove(node)

		return abandoned

	def add_node(self, node, neighbors=[]):
		if node in self._nodes:
			return None
		newedges = []
		self._nodes.append(node)
		for n in neighbors:
			newedges.append([node,n])
		return self.add_edges(newedges)

	def add_edges(self, edges):
		newedges = []
		for e in edges:
			i = e[0]
			j = e[1]
			if i not in self._neighbors:
				self._neighbors[i] = set()
			if j not in self._neighbors:
				self._neighbors[j] = set()
			if i not in self._neighbors[j]:
				self._neighbors[j].add(i)
				self._neighbors[i].add(j)
				self._edges.append(e)
				newedges.append(e)
		return newedges

	def make_clique(self, nodes):
		if nodes is None:
			return None
		newedges = []
		for idx,i in enumerate(nodes):
			for j in nodes[idx+1:]:
				if j not in self.neighbors(i):
					newedges.append([i,j])
		return self.add_edges(newedges)

	def rm_var(self, node):
		abandoned = self.remove_node(node)
		clique = self.make_clique(abandoned)
		nodes = set()
		for e in clique:
			for n in e:
				nodes.add(n)
		return len(nodes), clique
		
	def product_ij(self, psi_ij, x_v):
		product = 1
		for e in self._edges:
			i = self.node_idx(e[0])
			j = self.node_idx(e[1])
			x_i = float(x_v[i])
			x_j = float(x_v[j])
			product *= psi_ij.__call__(x_i,x_j)
		return product
		
	def product_i(self, phi_i, x_v):
		product = 1
		for x_i in x_v:
			product *= phi_i.__call__(float(x_i))
		#print("phi_i({}) = {}]".format(x_v, product))
		return product

