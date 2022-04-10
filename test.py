import numpy as np 
import pickle 
import networkx as nx 
from scipy.spatial.distance import cosine
import random
from operator import itemgetter
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored
import logging
import copy
import os

import matplotlib.pyplot as plt
import tqdm
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import torch
import build_adjacency_matrix

import dgl.function as fn
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dataset import KarateClubDataset

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
    
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
        
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, 2) # DotProductPredictor()
    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)

dataset = KarateClubDataset()
sco = DotProductPredictor()
graph = dataset[0]
print(sco(graph, graph.ndata['feat']).shape)

node_features = graph.ndata['feat']
edge_label = graph.edata['label']
train_mask = graph.edata['train_mask']
model = Model(2048, 128, 64)
opt = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(300):
    pred = model(graph, node_features)
    #print(pred[train_mask].shape, edge_label[train_mask].shape)
    loss = criterion(pred[train_mask], edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
    print(criterion(pred[~train_mask], edge_label[~train_mask]))