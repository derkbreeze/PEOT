import argparse
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

import sys
sys.path.append('src')
import asot

from video_dataset import VideoDataset
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description="Train representation learning pipeline")
parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')

# FUGW OT segmentation parameters
parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6, help='weighting of KOT term on frame features in OT')
parser.add_argument('--ub-frames', '-uf', action='store_true',
                    help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
parser.add_argument('--ub-actions', '-ua', action='store_false',
                    help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (train)')
parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (eval)')
parser.add_argument('--step-size', '-ss', type=float, default=None,
                    help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')
parser.add_argument('--temp', type=float, default=0.1, help='Temperature parameter for Softmax')

# dataset params
parser.add_argument('--exclude', '-x', type=int, default=None, help='classes to exclude from evaluation. use -1 for YTI')
parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')

# representation learning params
parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
parser.add_argument('--batch-size', '-bs', type=int, default=2, help='batch size')
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--learning-rate-cluster', '-lc', type=float, default=1e-3, help='learning rate for clusters')
parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
parser.add_argument('--k-means', '-km', action='store_false', help='do not initialize clusters with kmeans default = True')
parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')

# system/logging params
parser.add_argument('--gpu', '-g', type=int, default=1, help='gpu id to use')
parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')

parser.add_argument('--peot', action='store_true', help='whether to use peot')
parser.add_argument('--prob', '-pr', action='store_true', help='whether or not to use probabilistic embeddings')
parser.add_argument('--conv', '-cv', action='store_true', help='whether or not to use TCN')

args = parser.parse_args()

#python test.py -pr -s -d Breakfast -v
#python test.py -pr -s -d YTI -v
#python test.py -pr -s -d FSeval -v
if args.dataset == 'Breakfast':
    activities = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    n_clusters = [7, 5, 7, 5, 8, 9, 12, 9, 8, 14]

    #full model
    args.rho, args.radius_gw, args.alpha_train, args.alpha_eval, args.ub_actions, args.lambda_actions_train = 0.2, 0.04, 0.4, 0.7, True, 0.1
    #import ipdb;ipdb.set_trace()
elif args.dataset == 'YTI':
    activities = ['changing_tire', 'coffee', 'cpr', 'jump_car', 'repot']
    n_clusters, args.layers, args.exclude = [11, 10, 7, 12, 8], [3000, 32, 32], -1

    #full model
    args.rho, args.radius_gw, args.ub_actions, args.lambda_actions_train, args.lambda_actions_eval = 0.2, 0.02, True, 0.12, 0.01
    #import ipdb;ipdb.set_trace()
elif args.dataset == 'FSeval':
    args.layers = [64, 128, 40]
    #full model 
    activities, n_clusters, args.rho, args.radius_gw, args.ub_actions, args.lambda_actions_train = ['all'], [12, ], 0.05, 0.02, True, 0.1
elif args.dataset == 'desktop_assembly': #python test.py --va -d desktop_assembly -v
    activities, n_clusters, args.layers = ['all'], [22], [512, 128, 40]
    load_epoch = 6

    args.rho, args.radius_gw, args.alpha_eval, args.lambda_actions_eval = 0.25, 0.01, 0.6, 0.01
elif args.dataset == 'IKEA':
    activities = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    n_clusters = [17, 18, 14, 18]

    args.layers, args.n_epochs = [1024, 40, 40], 15
    args.rho, args.radius_gw = 0.2, 0.04 #0.04 in ASOT
    args.alpha_train, args.alpha_eval, args.lambda_actions_train, args.eps_train, args.eps_eval = 0.4, 0.7, 0.1, 0.07, 0.04

tp_seg_agg, pr_seg_agg, gt_seg_agg = 0, 0, 0
n_total_list, MoF_list, F1_list, mIoU_list, video_metrics_total = [], [], [], [], []
for args.activity, args.n_clusters in zip(activities, n_clusters):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    D = args.layers[-1]
    if args.dataset == 'Breakfast':
        load_epoch = 14
        exp_id = 'exp_20260225_192218' 
    elif args.dataset == 'YTI':
        #load_epoch = 29
        exp_id = 'exp_20260227_095905' #the best model
        load_dict = {'changing_tire':29, 'coffee':14, 'cpr':20, 'jump_car':29, 'repot':29}
    elif args.dataset == 'FSeval':
        load_epoch = 69
        exp_id = 'exp_20260227_122502'
        #import ipdb;ipdb.set_trace()
    elif args.dataset == 'desktop_assembly':
        pass

    #import ipdb;ipdb.set_trace()
    save_dir = '{}_/{}/{}'.format(args.dataset, exp_id, args.activity)
    layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(args.layers[:-2], args.layers[1:-1])]
    layers += [nn.Linear(args.layers[-2], args.layers[-1])]
    mlp = nn.Sequential(*layers)
    mlp = mlp.cuda()
    gcn_mean = nn.Linear(D, D).cuda()
    gcn_logstd = nn.Linear(D, D).cuda()

    if args.dataset == 'YTI':
        load_epoch = load_dict[args.activity]
        #import ipdb;ipdb.set_trace()

    checkpoints = torch.load(save_dir + '/epoch_{:02d}.pth'.format(load_epoch))
    mlp.load_state_dict(checkpoints['mlp'])
    gcn_mean.load_state_dict(checkpoints['gcn_mean'])
    gcn_logstd.load_state_dict(checkpoints['gcn_logstd'])
    clusters = checkpoints['clusters']

    mof = ClusteringMetrics(metric='mof')
    f1 = ClusteringMetrics(metric='f1')
    miou = ClusteringMetrics(metric='miou')
    test_cache, video_metrics = [], []

    test_set = VideoDataset('/data', args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    cov_norm_avg, cov_norm_agg = 0, list()
    for batch_idx, batch in enumerate(test_loader):
        features_raw, mask, gt, fname, n_subactions = batch
        features_raw = features_raw.cuda()
        mask = mask.cuda()
        gt = gt.cuda() 
        B, T, _ = features_raw.shape

        with torch.no_grad():
            features = mlp(features_raw)
            features = F.normalize(features, dim=-1) #features: BxTxD
            adj = torch.stack([torch.eye(T).cuda() for _ in range(B)])

            #One-neighbor connection within GCN, baseline
            indices = torch.arange(T - 1)
            adj[:, indices, indices + 1] = 1
            adj[:, indices + 1, indices] = 1

            #two-neighbor connection within GCN
            # indices = torch.arange(T - 2)
            # adj[:, indices, indices + 2] = 1
            # adj[:, indices + 2, indices] = 1

            # indices = torch.arange(T - 1)
            # adj[:, indices, indices + 1] = 1
            # adj[:, indices + 1, indices] = 1

            adj_weighted = features @ features.transpose(1, 2)
            adj_weighted = (adj_weighted + 1) / 2
            adj_weighted = adj * adj_weighted

            node_degrees = torch.pow(adj_weighted.sum(-1), -0.5)
            node_degrees = torch.diag_embed(node_degrees)
            adj_norm = node_degrees @ adj_weighted @ node_degrees
             
            mean = adj_norm.matmul(gcn_mean(features))
            logstd = adj_norm.matmul(gcn_logstd(features))
            logvar = 2 * logstd
            cov_norm = logvar.exp().sum(-1).sqrt()
            features = F.normalize(mean, dim=-1)

            logvar = 2 * logstd
            cov_norm_avg += logvar.exp().sum(-1).sqrt().mean()
            cov_norm_agg.append(logvar.exp().sum(-1).sqrt().squeeze())

            cost_matrix = 1. - features @ clusters.T
            temp_prior = asot.temporal_prior(T, args.n_clusters, args.rho, features.device)
            cost_matrix += temp_prior

            segmentation, _ = asot.segment_asot(cost_matrix, mask, eps=args.eps_eval, alpha=args.alpha_eval, radius=args.radius_gw,
                                                ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames=args.lambda_frames_eval,
                                                lambda_actions=args.lambda_actions_eval, n_iters=args.n_ot_eval, step_size=args.step_size)
            segments = segmentation.argmax(dim=2)

        mof.update(segments, gt, mask)
        f1.update(segments, gt, mask)
        miou.update(segments, gt, mask)

        # log clustering metrics per video
        metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=args.exclude)
        n_frames = gt.shape[1] if args.dataset != 'YTI' else torch.where(gt.squeeze() != -1)[0].__len__() 
        data = {**metrics, **{'n_frames':n_frames, 'fname':fname[0]}}

        video_metrics.append(data)
        video_metrics_total.append(data)
        #import ipdb;ipdb.set_trace()
        if args.visualize:
            test_cache.append([cost_matrix, segmentation, segments, gt, fname, cov_norm])

    mof_activity, pred_to_gt = mof.compute(exclude_cls=args.exclude)
    return_stats, _ = f1.compute(exclude_cls=args.exclude, pred_to_gt=pred_to_gt)

    f1_activity = return_stats['f1']
    tp_seg, pr_seg = return_stats['precision']
    tp_seg, gt_seg = return_stats['recall']

    tp_seg_agg += tp_seg
    pr_seg_agg += pr_seg
    gt_seg_agg += gt_seg

    miou_activity, _ = miou.compute(exclude_cls=args.exclude, pred_to_gt=pred_to_gt)

    #metrics using Hungarian matching at video level  
    mof_video_avg, n_correct, f1_video, miou_video = 0, 0, 0, 0
    n_frames = np.where(np.array(mof.gt_labels) != -1)[0].__len__()
    n_total_list.append(n_frames)
    for ind in range(video_metrics.__len__()):
        n_correct += video_metrics[ind]['mof'] * video_metrics[ind]['n_frames']
        mof_video_avg += video_metrics[ind]['mof']
        f1_video += video_metrics[ind]['f1']
        miou_video += video_metrics[ind]['miou']

    n_videos = video_metrics.__len__()
    mof_video = n_correct / n_frames
    mof_video_avg /= n_videos
    f1_video /= n_videos
    miou_video /= n_videos

    MoF_list.append(mof_activity)
    F1_list.append(f1_activity)
    mIoU_list.append(miou_activity)

    print('{} total frames {} activity level MoF {:.3f} F1 {:.3f} mIoU {:.3f} video level MoF {:.3f} avg {:.3f} F1 {:.3f} mIoU {:.3f}'.format(
                    args.activity, n_frames, mof_activity, f1_activity, miou_activity, mof_video, mof_video_avg, f1_video, miou_video))
    
    if args.visualize:

        test_dir = save_dir + '/test_{}'.format(load_epoch)
        os.makedirs(test_dir, exist_ok=True)

        gt_uniq = np.unique(mof.gt_labels)
        n_class = len(gt_uniq)
        if n_class <= 20:
            cmap = plt.get_cmap('tab20')
        else:  # up to 40 classes
            cmap1 = plt.get_cmap('tab20')
            cmap2 = plt.get_cmap('tab20b')
            cmap = lambda x: cmap1(round(x * n_class / 20., 2)) if x <= 19. / n_class else cmap2(round((x - 20 / n_class) * n_class / 20, 2))

        colors = {}
        for i, label in enumerate(gt_uniq):
            if label == -1:
                colors[label] = (0, 0, 0)
            else:
                colors[label] = cmap(i / n_class)

        # import pickle
        # with open('colors.pkl', 'wb') as f:
        #     pickle.dump(colors, f)

        #import ipdb;ipdb.set_trace()
        for video_idx in range(len(test_cache)):
            if video_idx % 50 == 0:
                print(f'Finished [{video_idx}/{len(test_cache)}] videos')

            cost_matrix, segmentation, pred, gt, fname, cov_norm = test_cache[video_idx]
            pred_, gt_ = pred.squeeze().cpu().numpy(), gt.squeeze().cpu().numpy()
            #pred_, gt_ = filter_exclusions(pred, gt, excl_cls=args.exclude) #retain bg (-1) class for YTI

            # if fname[0] != 'P26_cam01_P26_salat':
            #     continue
            
            for i, label in enumerate(pred_):
                if gt_[i] == -1:
                    pred_[i] = -1
                else:
                    pred_[i] = pred_to_gt[label]

            #import ipdb;ipdb.set_trace()
            n_frames = len(pred_)        
            fig, axes = plt.subplots(5, figsize=(16, 8))

            plot = axes[0].matshow(cost_matrix[0].detach().cpu().T)
            axes[0].set_aspect('auto')

            plot = axes[1].matshow(segmentation[0].detach().cpu().T)
            axes[1].set_aspect('auto')
            axes[1].set_xticklabels([])

            plot = axes[2].plot(cov_norm.squeeze().cpu(), linewidth=2)
            axes[2].set_aspect('auto')
            axes[2].set_xticklabels([])
            axes[2].margins(x=0)
            axes[2].set_ylim(0, cov_norm.max().item() + 0.2)

            axes[3].margins(x=0)
            axes[3].set_ylabel('Pred', fontsize=20, rotation=0, labelpad=20, verticalalignment='center')
            axes[3].set_yticklabels([])
            axes[3].set_xticklabels([])

            pred_segment_boundaries = np.where(pred_[1:] - pred_[:-1])[0] + 1
            pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_)]))

            for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
                label = pred_[start]
                axes[3].axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
                axes[3].axvline(start / n_frames, color='black', linewidth=3)
                axes[3].axvline(end / n_frames, color='black', linewidth=3)

            axes[4].margins(x=0)
            axes[4].set_ylabel('GT', fontsize=20, rotation=0, labelpad=20, verticalalignment='center')
            axes[4].set_yticklabels([])
            axes[4].set_xticklabels([])

            gt_segment_boundaries = np.where(gt_[1:] - gt_[:-1])[0] + 1
            gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

            for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
                label = gt_[start]
                axes[4].axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
                axes[4].axvline(start / n_frames, color='black', linewidth=3)
                axes[4].axvline(end / n_frames, color='black', linewidth=3)
                
            fig.tight_layout()
            plt.savefig(test_dir + '/{}.png'.format(fname[0], bbox_inches='tight'))
            plt.close()

            dic = {'colors':colors, 'pred_':pred_, 'gt_':gt_, 'cov_norm':cov_norm}
            torch.save(dic, '{}.pth'.format(fname[0]))
    
def calc_metrics(n_total_list, MoF_list, F1_list, mIoU_list):
    n_correct = np.sum(np.array(MoF_list) * np.array(n_total_list))
    MoF =  n_correct / np.sum(n_total_list)

    F1 = np.sum(F1_list) / len(F1_list)
    mIoU = np.sum(mIoU_list) / len(mIoU_list)
    mIoU_ = np.sum(np.array(mIoU_list) * np.array(n_clusters)) / np.sum(n_clusters)
    return MoF, F1, mIoU, mIoU_

#import ipdb;ipdb.set_trace()
MoF, F1, mIoU, mIoU_ = calc_metrics(n_total_list, MoF_list, F1_list, mIoU_list) #macro F1-score
precision, recall = tp_seg_agg / pr_seg_agg, tp_seg_agg / gt_seg_agg  #precision and recall over all videos
F1_micro = (2 * precision * recall) / (precision + recall + 1e-7) #micro F1-score

MoF_video_avg, n_correct, F1_video, mIoU_video = 0, 0, 0, 0
n_total_frames, n_total_videos = np.sum(n_total_list), video_metrics_total.__len__()
for ind in range(n_total_videos):
    MoF_video_avg += video_metrics_total[ind]['mof']
    n_correct += video_metrics_total[ind]['mof'] * video_metrics_total[ind]['n_frames']
    F1_video += video_metrics_total[ind]['f1']
    mIoU_video += video_metrics_total[ind]['miou']

MoF_video_avg /= n_total_videos         #This is NOT the correct MoF (per-video, but TWFINCH-CVPR2021 used this.)
MoF_video = n_correct / n_total_frames  #This is the correct MoF (calculated via per-video Hungarian matching)
F1_video /= n_total_videos              #This is the F1 averaged over all videos (calculated via per-video Hungarian matching)
mIoU_video /= n_total_videos            #This is the mIoU averaged over all videos (calculated via per-video Hungarian matching)
print('{} total frames {} activity level MoF {:.2f} F1 {:.2f} F1-micro {:.2f} mIoU {:.2f} mIoU_ {:.2f} video level MoF {:.2f} avg {:.2f} F1 {:.2f} mIoU {:.2f}'.format(
            args.dataset, n_total_frames, MoF * 100, F1 * 100, F1_micro * 100, mIoU * 100, mIoU_ * 100, 
            MoF_video * 100, MoF_video_avg * 100, F1_video * 100, mIoU_video * 100))
