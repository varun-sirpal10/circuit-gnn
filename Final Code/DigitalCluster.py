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

	def __init__(self, file, epochs, learning_rate, clusters):
		self.filename = file
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.clusters = clusters

	def run(self):

		edge_file = "add_edges.py"
		edge_file_handle = open(edge_file, "w+")

		node_count = 0

		with open(self.filename,encoding="utf8") as f:
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

		net = GCN(6,5,self.clusters)

		inputs = embed.weight
		a = np.linspace(0,G.number_of_nodes()-1,self.clusters,dtype=int)
		labelled_nodes = torch.tensor(a,dtype=torch.long)
		b = np.arange(self.clusters)
		labels = torch.tensor(b,dtype=torch.long)

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
	# def number_of_nodes_per_cluster(self):

		pos = {}
		node_to_cluster_map = np.zeros((G.number_of_nodes(),))

		cluster_nodes = {} # Every value in key->value pair includes the cluster centre node as well.
		for v in range(G.number_of_nodes()):
		  pos[v] = all_logits[99][v].numpy()
		  cls = pos[v].argmax()
		  node_to_cluster_map[v] = cls
		  if cluster_nodes.get(cls) :
		    cluster_nodes[cls].append(v)
		  else:
		    cluster_nodes[cls] = []
		    cluster_nodes[cls].append(v)

		number_dict = {} # Dictionary mapping the node label to the number of nodes in a particular cluster
		for key,value in cluster_nodes.items():
			number_dict[key] = len(value)

		number_list = sorted(number_dict.items()) # Sorting dictionary by keys in increasing order

		for ix in number_list:
			print("Number of Nodes in Cluster {} = {}\n".format(ix[0],ix[1]))

	# def edges_between_clusters(self):

		def find_cluster(node):
  			return node_to_cluster_map[node]

		edges_dict = {}

		for node in range(G.number_of_nodes()):
		  #print("Node {}".format(node))
		  # node is the source node for all out edges

		  out_edges = G.out_edges(node,form='uv')

		  src = out_edges[0]
		  if(len(src) < 1):
		    continue
		    
		  if src[-1] != node:
		    print("Source node is different from node")
		  dst = out_edges[1]

		  src_cluster = find_cluster(node)

		  for ix in range(src.shape[0]):
		    #node_n = src[ix]
		    node_n = node
		    node_m = dst[ix]

		    #cluster_n = find_cluster(node_n)
		    cluster_n = src_cluster
		    cluster_m = find_cluster(node_m)

		    if cluster_n != cluster_m:
		      from_cluster = cluster_n
		      to_cluster = cluster_m
		      if from_cluster > to_cluster:
		        from_cluster = cluster_m
		        to_cluster = cluster_n

		      if (from_cluster,to_cluster) not in edges_dict.keys():
		        edges_dict[(from_cluster,to_cluster)] = 1
		      else:
		        edges_dict[(from_cluster,to_cluster)] += 1


		for key,value in edges_dict.items():
			num = value
			# nodes_1 = number_dict[int(key[0])][1]
			# nodes_2 = number_dict[int(key[1])][1]

			nodes_1 = number_dict.get(int(key[0]))
			nodes_2 = number_dict.get(int(key[1]))

			num2 = value/int((nodes_1 + nodes_2))
			num2 = round(num2,4)

			edges_dict[key] = (num,num2)


		count = 0
		for key,value in edges_dict.items():
			count += 1
		print("Entries in edges_dict = {}".format(count))

		edges_list = sorted(edges_dict.items(),reverse=True,key = lambda x: x[1])

		return number_list,edges_list,edges_dict,cluster_nodes


	def visualise(self,number_list,edges_list,edges_dict,cluster_nodes):

		y_vals = []
		x_vals = []

		for key,value in cluster_nodes.items():
		    y_vals.append(len(value))
		    x_vals.append(key)

		plt.style.use("seaborn")
		plt.bar(x_vals,y_vals)
		plt.xlabel("Node Labels")
		plt.ylabel("Number of Nodes")
		plt.title("Number of Nodes per Cluster Distribution")
		plt.show()

		keys_list = []
		values_list = []
		values2_list = []

		count = 0
		for val in edges_list:
		  count += 1
		  int_val1 = int(val[0][0])
		  int_val2 = int(val[0][1])
		  keys_list.append(str((int_val1,int_val2)))
		  values_list.append(val[1][0])
		  values2_list.append(val[1][1])
		  if(count == 10):
		    break

		# keys_list_int = []
		# for key in keys_list:
		# 	int_val1 = int(float(key[0]))
		# 	int_val2 = int(float(key[1]))
		# 	keys_list_int.append((int_val1,int_val2))

		plt.style.use("seaborn")
		plt.bar(keys_list,values_list)
		plt.ylabel("Number of edges between 2 clusters")
		plt.xlabel("Cluster Pairs")
		plt.title("Top 10 cluster pairs with high interconnectivity")
		plt.show()

		plt.style.use("seaborn")
		plt.bar(keys_list,values2_list)
		plt.ylabel("Intercluster Score")
		plt.xlabel("Cluster Pairs")
		plt.title("Top 10 cluster pairs with high interconnectivity")
		plt.show()