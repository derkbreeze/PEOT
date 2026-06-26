import os
import numpy as np
np.set_printoptions(suppress=True)
import torch
torch.set_printoptions(sci_mode=False)

import matplotlib.pyplot as plt
import utils
import asot


vidname = 'P36_webcam01_P36_salat'  # Breakfast (Salat)
dataset = 'da' if '2020' in vidname else 'salat'

if dataset == 'da':
    ub_weight = 0.09 # DA hparams
    eps = 0.01
    alpha = 0.6
    r = 0.02
else:
    ub_weight = 0.06 # BF (salat) hparams
    eps = 0.01
    alpha = 0.6
    r = 0.04
    
affinity = torch.from_numpy(np.load('data/affinity/{}.npy'.format(vidname)))
affinity = affinity.unsqueeze(0)
matching_cost = 1 - affinity

def process_mapping(x):
    i, nm = x.rstrip().split(' ')
    return nm, int(i)

action_mapping = dict(map(process_mapping, open(os.path.join('data', 'gt', 'mapping_{}.txt'.format(dataset)))))

gt = [line.rstrip() for line in open(os.path.join('data', 'gt', vidname))]
gt = torch.Tensor(list(map(lambda x: action_mapping[x], gt))).int()

# run ASOT
with torch.no_grad():
    B, N, K = matching_cost.shape
    soft_assign, _ = asot.segment_asot(matching_cost, eps=eps, alpha=alpha, radius=r, lambda_actions=ub_weight)
    segmentation = soft_assign.argmax(dim=-1).squeeze()



