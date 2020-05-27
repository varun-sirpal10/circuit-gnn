import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from DigitalCluster import DigitalClustering

DC = DigitalClustering(file="adaptec4.nets",epochs=250,learning_rate=0.01,clusters=20)

start_time = time.time()

number_list,edges_list,edges_dict,cluster_nodes = DC.run()

DC.visualise(number_list,edges_list,edges_dict,cluster_nodes)

end_time = time.time()

print("\n\nTime taken: {}".format(end_time - start_time))