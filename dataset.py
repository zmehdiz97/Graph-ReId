import dgl
from dgl.data import DGLDataset
import torch
import os
import pickle
import numpy as np
import random

class KarateClubDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='mydataset')

    def process(self):
        dataset = 'Market'

        with open(f'datasets/{dataset}/feat.pkl', 'rb') as f :
            features = pickle.load(f)  # (19281, 2048) [:100,:]
        with open(f'datasets/{dataset}/ids.pkl', 'rb') as f :
            pids = pickle.load(f) # 751 unique id [:100]
        with open(f'datasets/{dataset}/camids.pkl', 'rb') as f :
            camids = pickle.load(f)  # 6 unique camids [:100]
        print('number of unique ids: ', len(np.unique(pids)))
        
        number_ids = 100
        k=10
        selected_ids = np.unique(pids)[:number_ids]
        print(f'Using a subsample of {number_ids} ids')
        def select(id):
            return id in selected_ids

        selector = list(map(select, pids))
        print(sum(selector))
        features = features[selector]
        pids = pids[selector]
        camids = camids[selector]

        features = torch.from_numpy(features)
        pids = torch.from_numpy(pids)
        camids = torch.from_numpy(camids)

        original_score = torch.mm(features, features.t())
        S, initial_rank = original_score.topk(k=k, dim=-1, largest=True, sorted=True)
        src = torch.cat([ torch.tensor(list(range(len(pids)))) for _ in range(k)])
        dst = torch.cat([initial_rank[:,i] for i in range(k)])
        scr = torch.cat([S[:,i] for i in range(k)])

        labels = torch.zeros_like(src)
        n=0
        for s,d in zip(src, dst):
            if pids[s] == pids[d]:
                labels[n] = 1
            n += 1
        
        self.graph = dgl.graph( (src, dst), num_nodes=pids.shape[0])
        self.graph.ndata['feat'] = features
        self.graph.ndata['label'] = pids
        #self.graph.edata['weight'] = edge_features
        self.graph.edata['feat'] = scr[:,None]
        self.graph.edata['label'] = labels#[:,None]
        self.graph.edata['train_mask'] = torch.zeros(len(labels), dtype=torch.bool).bernoulli(0.7)

        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = KarateClubDataset()
graph = dataset[0]

print(graph)