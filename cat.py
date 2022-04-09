import numpy as np 
import pickle 
import torch
from utils import evaluate_ranking_list
from gnn_reranking import gnn_reranking

dataset = 'Market'
num_query = 3368
k1, k2 = 26,7

with open(f'datasets/{dataset}/ids.pkl', 'rb') as f :
    pids = pickle.load(f) # 751 unique id [:100]
with open(f'datasets/{dataset}/camids.pkl', 'rb') as f :
    camids = pickle.load(f)  # 6 unique camids [:100]

pids = torch.from_numpy(pids)
camids = torch.from_numpy(camids)

# query feature, person ids and camera ids
query_pids = pids[:num_query]
query_camids = camids[:num_query]

# gallery features, person ids and camera ids
gallery_pids = pids[num_query:]
gallery_camids = camids[num_query:]

indices = np.concatenate([np.load(f'L{i+1}.npy') for i in range(4)], axis=0)
evaluate_ranking_list(indices, query_pids, query_camids, gallery_pids, gallery_camids)