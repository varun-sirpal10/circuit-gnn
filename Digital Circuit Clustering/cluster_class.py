from os import listdir
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import itertools
import time

class DigitalClustering:

	def __init__(self, epochs, learning_rate):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.cluster_nodes = {}

	def train_gnn(self):

		filename = "adaptec1.nets"

		edge_file = "add_edges.py"
		edge_file_handle = open(edge_file, "w+")

		node_count = 0

		with open(filename,encoding="utf8") as f:
		    netSection = False
		    edges = 0
		    lineCount = 0
		    src=[]
		    dest=[]
		    for line in f:
		        lineCount += 1
		        #Newline
		        if len(line) < 2:
		            continue

		        # Comments
		        if line[0] == '#':
		            continue
		        #print(line)
		        splitLine = line.split()
		        #print("Words = ", len(splitLine))
		        if(splitLine[0] == 'NetDegree'):

		#             if edges > 0: #Previous NetDegree not complete!
		#                 print("Missed an edge at line: ", lineCount-1)

		            edges = int(splitLine[2])
		            netSection = True
		            src=[]
		            dest=[]
		            #G.add_nodes(num)

		        elif netSection == True:
		          node_name = splitLine[0][1:]
		          node_val  = int(node_name)
		          if node_count < node_val+1:
		              node_count = node_val+1

		          if(splitLine[1] == 'O'):
		              src.append(node_name)
		              edges -= 1
		          elif(splitLine[1] == 'I'):
		              dest.append(node_name)
		              edges -= 1

		        if netSection == True:
		            if edges == 0:
		                if (len(src)<1):
		#                     print("No source edge, skipping")
		                    continue
		                if (len(dest)<1):
		#                     print("No dest edge, skipping")
		                    continue

		                netSection = False
		                for des_idx in range(len(dest)):
		                    cmd = "G.add_edges("+src[0]+","+dest[des_idx]+")"+"\n"
		                    edge_file_handle.write(cmd)

		edge_file_handle.close()

		G = dgl.DGLGraph()
		G.add_nodes(node_count)
		exec(open(edge_file).read())
		#     print("Nodes = ", G.number_of_nodes())
		#     print("Edges = ", G.number_of_edges())

		embed = nn.Embedding(G.number_of_nodes(),6)

		G.ndata['feat'] = embed.weight

		class GCN(nn.Module):
		    def __init__(self,input_size,hidden_size,num_classes):
		        super(GCN,self).__init__()
		        self.conv1 = GraphConv(input_size,hidden_size)
		        self.conv2 = GraphConv(hidden_size,num_classes)

		    def forward(self,G,inputs):
		        h = self.conv1(G,inputs)
		        h = torch.relu(h) 
		        h = self.conv2(G,h)
		        return h

		net = GCN(6,5,20)

		inputs = embed.weight
		a = np.linspace(0,G.number_of_nodes()-1,20,dtype=int)
		labelled_nodes = torch.tensor(a,dtype=torch.long)
		labels = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],dtype=torch.long)

		"""# Training Model"""

		optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=self.learning_rate)
		all_logits = []

		start_time = time.time()

		for epoch in range(self.epochs):
		    logits = net(G, inputs)
		    all_logits.append(logits.detach())
		    logp = F.log_softmax(logits, 1)
		    loss = F.nll_loss(logp[labelled_nodes], labels)

		    optimizer.zero_grad()
		    loss.backward()
		    optimizer.step()

		#     print('Epoch {} | Loss: {}'.format(epoch + 1, loss.item()))

		end_time = time.time()
		# print("Training Time: {}".format(end_time - start_time)


	def number_of_nodes_per_cluster(self):

		pos = {}
		self.cluster_nodes = {} # Every value in key->value pair includes the cluster centre node as well.
		for v in range(G.number_of_nodes()):
		  pos[v] = all_logits[99][v].numpy()
		  cls = pos[v].argmax()
		  if self.cluster_nodes.get(cls) :
		    self.cluster_nodes[cls].append(v)
		  else:
		    self.cluster_nodes[cls] = [] 
		    self.cluster_nodes[cls].append(v)

		number_dict = {} # Dictionary mapping the node label to the number of nodes in a particular cluster
		for key,value in self.cluster_nodes.items():
			number_dict[key] = len(value)

			number_dict = sorted(number_dict.items()) # Sorting dictionary by keys in increasing order

		for ix in number_dict:
			print("Number of Nodes in Cluster {} = {}\n".format(ix[0],ix[1]))

		return number_dict


	def edges_between_clusters(self):

		m = 10 # Total number of clusters

		edges_between_clusters = {} # Absolute value of number of edges

		for i in range(m):
		#   print("Calculating for cluster {}\n".format(i))
		  nodes_i = cluster_nodes[i]

		  for j in range(i+1,m):
		    nodes_j = np.array(cluster_nodes[j])

		    ij_input_edges = G.in_edges(nodes_i,form='uv')
		    edges_src = np.array(ij_input_edges[0])

		    count = 0

		    for val1 in nodes_j:
		      for val2 in edges_src:
		        if val1 == val2:
		          count += 1

		    # for x in range(np.array(nodes_i).shape[0]):
		    #   if edges_src[x] == nodes_j[x]:
		    #     count += 1

		    edges_between_clusters[(i,j)] = count

		percentage_dict = {} # Edges as a % of the total edges in graph G

		for key,value in edges_between_clusters.items():
		  value = (value/G.number_of_edges())*100
		  value = round(value,4)
		  percentage_dict[key] = value

		return edges_between_clusters,percentage_dict


DC = DigitalClustering(epochs=250,learning_rate=0.01)

start_time = time.time()
DC.train_gnn()

number_dict = DC.number_of_nodes_per_cluster()
print(number_dict.items())

end_time = time.time()
print("\n\nTime taken: {}".format(end_time - start_time))