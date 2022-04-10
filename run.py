import numpy as np 
import pickle 
import torch
import random
from utils import evaluate_ranking_list
from gnn_reranking import gnn_reranking

from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored
import logging


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron2,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict): {metric -> score}
    """
    # unordered results cannot be properly printed

    logger = logging.getLogger(__name__)

    dataset_name = results.pop('dataset')
    metrics = ["Dataset"] + [k for k in results]
    csv_results = [(dataset_name, *list(results.values()))]

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=metrics,
        numalign="left",
    )

    print("Evaluation results in csv format: \n" + colored(table, "cyan"))
def eval_metrics(dataset, indices, q_ids, g_ids, q_camids, g_camids, max_rank=10):
    """Evaluation of the queries output 
    """
    num_q, num_g = indices.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0  # number of valid query

    i=0
    for q_idx in range(num_q):
        # get query pid and camid
        q_id = q_ids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_ids[order] == q_id) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_ids[order] == q_id).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            i+=1
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    print('number of queries that do not exist in the gallery :', i)
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    results = OrderedDict()
    results['dataset'] = dataset
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['Rank-{}'.format(r)] = all_cmc[r - 1] * 100
    results['mAP'] = mAP * 100
    results['mINP'] = mINP * 100
    results["metric"] = (mAP + all_cmc[0]) / 2 * 100

    print_csv_format(results)
    return all_cmc, all_AP, all_INP

dataset = 'Market'
num_query = 3368
k1, k2 = 26,1

with open(f'datasets/{dataset}/feat.pkl', 'rb') as f :
    features = pickle.load(f)  # (19281, 2048) [:100,:]
with open(f'datasets/{dataset}/ids.pkl', 'rb') as f :
    pids = pickle.load(f) # 751 unique id [:100]
with open(f'datasets/{dataset}/camids.pkl', 'rb') as f :
    camids = pickle.load(f)  # 6 unique camids [:100]


## query feature, person ids and camera ids
#query_features = features[:1800]
#query_pids = pids[:1800]
#query_camids = camids[:1800]
#
## gallery features, person ids and camera ids
#gallery_features = features[num_query:]
#gallery_pids = pids[num_query:]
#gallery_camids = camids[num_query:]
print('number of unique ids: ', len(np.unique(pids)))
number_ids = 220
selected_ids = np.unique(pids)[:number_ids]
print(f'Using a subsample of {number_ids} ids')
def select(id):
    return id in selected_ids

selector = list(map(select, pids))
print('samples', sum(selector))
features = features[selector,:]
pids = pids[selector]
camids = camids[selector]

features = torch.from_numpy(features)
pids = torch.from_numpy(pids)
camids = torch.from_numpy(camids)

random.seed(10)
gallery_mask = random.sample(range(len(pids)), k=int(round(len(pids)*0.7)))
query_mask = list(set(range(len(pids))) - set(gallery_mask))

query_pids = pids[query_mask]
query_features = features[query_mask, :]
query_camids = camids[query_mask]

gallery_pids = pids[gallery_mask]
gallery_features = features[gallery_mask, :]
gallery_camids = camids[gallery_mask]

query_features = query_features.cuda()
gallery_features = gallery_features.cuda()
print(query_features.shape, gallery_features.shape)
indices = gnn_reranking(query_features, gallery_features, k1, k2)
#evaluate_ranking_list(indices, query_pids, query_camids, gallery_pids, gallery_camids)
query_pids, gallery_pids, query_camids, gallery_camids = \
    query_pids.cpu().numpy(), gallery_pids.cpu().numpy(),\
    query_camids.cpu().numpy(), gallery_camids.cpu().numpy()
eval_metrics(dataset, indices, query_pids, gallery_pids, query_camids, gallery_camids, max_rank=10)