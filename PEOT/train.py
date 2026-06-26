import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)

import warnings
warnings.filterwarnings('ignore')
import yaml

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import time
from datetime import datetime

import sys
sys.path.append('src')
import asot

from video_dataset import VideoDataset
from utils import plot_pair_imgs
from metrics import ClusteringMetrics, indep_eval_metrics

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description="Train representation learning pipeline")
parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')

# FUGW OT segmentation parameters
parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6, help='weighting of KOT term on frame features in OT')

parser.add_argument('--ub-frames', '-uf', action='store_true', help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
parser.add_argument('--ub-actions', '-ua', action='store_true',help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (train)')
parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (eval)')
parser.add_argument('--step-size', '-ss', type=float, default=None, help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')
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

# system/logging params
parser.add_argument('--gpu', '-g', type=int, default=1, help='gpu id to use')
parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')

parser.add_argument('--prob', '-pr', action='store_true', help='whether or not to use probabilistic embeddings')
parser.add_argument('--n-samples', '-ns', type=int, default=3, help='number of samples for uncertainty quantization')

args = parser.parse_args()
if args.dataset == 'Breakfast':
    activities = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    n_clusters = [7, 5, 7, 5, 8, 9, 12, 9, 8, 14]
    #full model
    #args.rho, args.radius_gw, args.alpha_train, args.alpha_eval, args.ub_actions, args.lambda_actions_train = 0.2, 0.04, 0.4, 0.7, True, 0.1

    #sensitivity alpha_train
    #args.rho, args.radius_gw, args.alpha_eval, args.ub_actions, args.lambda_actions_train = 0.2, 0.04, 0.7, True, 0.1
    
    #sensitivity lambda_actions_train
    #args.rho, args.radius_gw, args.alpha_train, args.alpha_eval, args.ub_actions = 0.2, 0.04, 0.4, 0.7, True

    #sensitivity radius_gw
    #args.rho, args.alpha_train, args.alpha_eval, args.ub_actions, args.lambda_actions_train = 0.2, 0.4, 0.7, True, 0.1

    #sensitivity rho
    #args.rho, args.radius_gw, args.alpha_train, args.alpha_eval, args.ub_actions, args.lambda_actions_train = 0.04, 0.4, 0.7, True, 0.1
    #import ipdb;ipdb.set_trace()
elif args.dataset == 'YTI':
    activities = ['changing_tire', 'coffee', 'cpr', 'jump_car', 'repot']
    n_clusters = [11, 10, 7, 12, 8]
    args.layers, args.n_epochs, args.exclude = [3000, 32, 32], 30, -1
    #full model
    #args.rho, args.radius_gw, args.ub_actions, args.lambda_actions_train, args.lambda_actions_eval = 0.2, 0.02, True, 0.12, 0.01

    #sensitivity alpha_train
    #args.rho, args.radius_gw, args.ub_actions, args.lambda_actions_train = 0.2, 0.02, True, 0.12

    #sensitivity lambda_actions_train
    #args.rho, args.radius_gw, args.ub_actions = 0.2, 0.02, True

    #sensitivity radius_gw
    #args.rho, args.ub_actions, args.lambda_actions_train = 0.2, True, 0.12

    #sensitivity rho
    #args.radius_gw, args.ub_actions, args.lambda_actions_train = 0.02, True, 0.12
    #import ipdb;ipdb.set_trace()
elif args.dataset == 'FS':
    activities, n_clusters, args.n_epochs = ['all'], [19,], 70
elif args.dataset == 'FSeval':
    activities, n_clusters, args.n_epochs = ['all'], [12,], 70
elif args.dataset == 'desktop_assembly':
    args.layers, activities, n_clusters = [512, 128, 40], ['all'], [23] #but GT K is 22 :(

print(args)
num_eps = 1e-11

save = False #whether or not to save training hyper-parameters and checkpoints
if save:
    exp_id = 'exp_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = '{}_/{}'.format(args.dataset, exp_id) if args.prob else '{}/{}'.format(args.dataset, exp_id)

    os.makedirs(exp_dir, exist_ok=True)
    with open(exp_dir + '/config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

n_total_list, MoF_list, F1_list, mIoU_list = [], [], [], []
for args.activity, args.n_clusters in zip(activities, n_clusters):
    if args.dataset == 'YTI':
        if args.activity != 'repot':
            args.lambda_actions_train = 0.12 #0.12
        else:
            args.lambda_actions_train = 0.1 #0.1
        print(f'YTI activity: {args.activity} lambda_actions_train: {args.lambda_actions_train}')

    if args.dataset == 'YTI' and args.activity != 'changing_tire':
        continue

    if args.dataset == 'Breakfast' and args.activity != 'coffee':
        continue

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set = VideoDataset('/data', args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    test_set = VideoDataset('/data', args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # initialize evaluation metrics
    mof = ClusteringMetrics(metric='mof')
    f1 = ClusteringMetrics(metric='f1')
    miou = ClusteringMetrics(metric='miou')

    D = args.layers[-1]
    # initialize MLP
    layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(args.layers[:-2], args.layers[1:-1])]
    layers += [nn.Linear(args.layers[-2], args.layers[-1])]
    mlp = nn.Sequential(*layers)
    mlp = mlp.cuda()

    # initialize cluster centers/codebook
    clusters = nn.Parameter(F.normalize(torch.randn(args.n_clusters, D).cuda(), dim=-1)) #making the clusters leaf tensor that can be optimized by Adam
    if args.k_means:
        with torch.no_grad():
            features_full = []
            mlp.eval()
            for features_raw, _,  _, _, _ in train_loader:
                features_raw = features_raw.cuda()
                #import ipdb;ipdb.set_trace()
                B, T, _ = features_raw.shape
                features = F.normalize(mlp(features_raw), dim=-1)
                features_full.append(features)

            features_full = torch.cat(features_full, dim=0).reshape(-1, features.shape[2]).cpu().numpy()
            kmeans = KMeans(n_clusters=args.n_clusters).fit(features_full)
            #kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed).fit(features_full)
            mlp.train()
        #import ipdb;ipdb.set_trace()
        clusters.data = torch.from_numpy(kmeans.cluster_centers_).to(clusters.device)
        clusters.data = F.normalize(clusters.data, dim=-1)

    if not args.prob:
        gcn_mean = nn.Linear(D, D).cuda()
        optimizer = torch.optim.Adam(list(mlp.parameters()) + list(gcn_mean.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_cluster = torch.optim.Adam([clusters], lr=args.learning_rate_cluster, weight_decay=args.weight_decay)
    else:
        gcn_mean = nn.Linear(D, D).cuda()
        gcn_logstd = nn.Linear(D, D).cuda()
        optimizer = torch.optim.Adam(list(mlp.parameters()) + list(gcn_mean.parameters()) + list(gcn_logstd.parameters()),  
                                    lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_cluster = torch.optim.Adam([clusters], lr=args.learning_rate_cluster, weight_decay=args.weight_decay)

    if save:
        save_dir = exp_dir + '/{}'.format(args.activity)
        os.makedirs(save_dir, exist_ok=True)

    loss_list, epoch_loss_list, epoch_mof_activity_list, epoch_f1_activity_list, epoch_miou_activity_list = [], [], [], [], []
    for epoch in range(args.n_epochs):
        mlp.train()
        if not args.prob:
            for batch_idx, batch in enumerate(train_loader):
                features_raw, mask, gt, fname, n_subactions = batch
                features_raw = features_raw.cuda()
                mask = mask.cuda()
                gt = gt.cuda()

                B, T, _ = features_raw.shape

                features = mlp(features_raw)
                features = F.normalize(features, dim=-1) #features: BxTxD
                adj = torch.stack([torch.eye(T).cuda() for _ in range(B)])

                indices = torch.arange(T - 1)
                adj[:, indices, indices + 1] = 1
                adj[:, indices + 1, indices] = 1

                adj_weighted = features @ features.transpose(1, 2)
                adj_weighted = (adj_weighted + 1) / 2
                adj_weighted = adj * adj_weighted

                node_degrees = torch.pow(adj_weighted.sum(-1), -0.5)
                node_degrees = torch.diag_embed(node_degrees)
                adj_norm = node_degrees @ adj_weighted @ node_degrees

                features = adj_norm.matmul(gcn_mean(features))
                features = F.normalize(features, dim=-1)

                with torch.no_grad():
                    cost_matrix = 1. - features @ clusters.T
                    temp_prior = asot.temporal_prior(T, args.n_clusters, args.rho, features.device)
                    cost_matrix += temp_prior
        
                    opt_codes, _ = asot.segment_asot(cost_matrix, mask, eps=args.eps_train, alpha=args.alpha_train, radius=args.radius_gw,
                                                     ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames=args.lambda_frames_train,
                                                     lambda_actions=args.lambda_actions_train, n_iters=args.n_ot_train, step_size=args.step_size)
                        
                codes = F.softmax(features @ clusters.T / args.temp, dim=-1)
                loss = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
                loss_list.append(loss)

                #import ipdb;ipdb.set_trace()
                optimizer.zero_grad()
                optimizer_cluster.zero_grad()
                loss.backward()

                optimizer.step()
                optimizer_cluster.step()
                clusters.data = F.normalize(clusters.data, dim=-1)
        else:
            for batch_idx, batch in enumerate(train_loader):
                features_raw, mask, gt, fname, n_subactions = batch
                features_raw = features_raw.cuda()
                mask = mask.cuda()
                gt = gt.cuda()

                B, T, _ = features_raw.shape

                features = mlp(features_raw)
                features = F.normalize(features, dim=-1) #features: BxTxD
                adj = torch.stack([torch.eye(T).cuda() for _ in range(B)])

                ##One-neighbor connection within GCN (Baseline)
                indices = torch.arange(T - 1)
                adj[:, indices, indices + 1] = 1
                adj[:, indices + 1, indices] = 1

                adj_weighted = features @ features.transpose(1, 2)
                adj_weighted = (adj_weighted + 1) / 2
                adj_weighted = adj * adj_weighted

                node_degrees = torch.pow(adj_weighted.sum(-1), -0.5)
                node_degrees = torch.diag_embed(node_degrees)
                adj_norm = node_degrees @ adj_weighted @ node_degrees

                mean, logstd = adj_norm.matmul(gcn_mean(features)), adj_norm.matmul(gcn_logstd(features))
                loss_list_mc = []
                for _ in range(args.n_samples):
                    noise = torch.randn_like(mean)  
                    features = mean + noise * logstd.exp()
                    features = F.normalize(features, dim=-1)

                    with torch.no_grad():  # pseudo-labels from OT
                        cost_matrix = 1. - features @ clusters.T
                        temp_prior = asot.temporal_prior(T, args.n_clusters, args.rho, features.device)
                        cost_matrix += temp_prior
                        opt_codes, _ = asot.segment_asot(cost_matrix, mask, eps=args.eps_train, alpha=args.alpha_train, radius=args.radius_gw,
                                                        ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames=args.lambda_frames_train,
                                                        lambda_actions=args.lambda_actions_train, n_iters=args.n_ot_train, step_size=args.step_size)
                        
                    codes = F.softmax(features @ clusters.T / args.temp, dim=-1)
                    loss_mc = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
                    loss_list_mc.append(loss_mc)
                    
                loss = torch.stack(loss_list_mc).mean()
                loss_list.append(loss)

                optimizer.zero_grad()
                optimizer_cluster.zero_grad()
                loss.backward()

                optimizer.step()
                optimizer_cluster.step()
                clusters.data = F.normalize(clusters.data, dim=-1)
    
            spacing =  int(train_loader.__len__() / 5)
            if batch_idx > 0 and batch_idx % spacing == 0 and args.visualize:
                n_iter = len(train_loader) * epoch + batch_idx
                #print('{} epoch {} n_iter {} loss {:.2f}'.format(args.activity, epoch, n_iter, loss_list[-1]))

                tsne = TSNE(n_components=2, perplexity=15, random_state=0)
                #import ipdb;ipdb.set_trace()
                embeddings_tsne = tsne.fit_transform(torch.cat((features.view(-1, D).detach().cpu(), clusters.detach().cpu()), dim=0))
                # plt = plot_imgs(features, clusters, cost_matrix, temp_prior, opt_codes, codes, gt, res_list, epoch_loss_list, fname, embeddings_tsne)
                plt = plot_pair_imgs(features, clusters, cost_matrix, opt_codes, codes, gt, [], epoch_loss_list, fname, embeddings_tsne)               

                plt.savefig(save_dir + '/{:04d}.jpg'.format(n_iter), bbox_inches='tight')
                #display.clear_output(wait=True)
                #display.display(plt.gcf())
                plt.close()

        epoch_loss = torch.stack(loss_list).mean()
        epoch_loss_list.append(epoch_loss)
        if save:
            if not args.prob:
                state_dict = {'clusters':clusters.detach(), 'mlp':mlp.state_dict(), 'gcn_mean':gcn_mean.state_dict(), }
            else:
                state_dict = {'clusters':clusters.detach(), 'mlp':mlp.state_dict(), 
                              'gcn_mean':gcn_mean.state_dict(), 'gcn_logstd':gcn_logstd.state_dict(), }
            torch.save(state_dict, save_dir + '/epoch_{:02d}.pth'.format(epoch), )

        #import ipdb;ipdb.set_trace()
        mlp.eval()
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

                indices = torch.arange(T - 1)
                adj[:, indices, indices + 1] = 1
                adj[:, indices + 1, indices] = 1

                adj_weighted = features @ features.transpose(1, 2)
                adj_weighted = (adj_weighted + 1) / 2
                adj_weighted = adj * adj_weighted

                node_degrees = torch.pow(adj_weighted.sum(-1), -0.5)
                node_degrees = torch.diag_embed(node_degrees)
                adj_norm = node_degrees @ adj_weighted @ node_degrees

                features = adj_norm.matmul(gcn_mean(features))
                features = F.normalize(features, dim=-1)

                cost_matrix = 1. - features @ clusters.T
                temp_prior = asot.temporal_prior(T, args.n_clusters, args.rho, features.device)
                cost_matrix += temp_prior
                segmentation, _ = asot.segment_asot(cost_matrix, mask, eps=args.eps_eval, alpha=args.alpha_eval, radius=args.radius_gw,
                                                    ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames=args.lambda_frames_eval,
                                                    lambda_actions=args.lambda_actions_eval, n_iters=args.n_ot_eval, step_size=args.step_size)
                segments = segmentation.argmax(dim=2)

            # log clustering metrics over full epoch
            mof.update(segments, gt, mask)
            f1.update(segments, gt, mask)
            miou.update(segments, gt, mask)

            # log clustering metrics per video (per video Hungarian matching, this is essentially doing action boundary detection, rather than action segmentation)
            # metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=args.exclude)
            # metrics = {**metrics, **{'n_frames':gt.shape[1], 'fname':fname[0]}}
            # video_metrics.append(metrics)

        #import ipdb;ipdb.set_trace()
        mof_activity, pred_to_gt = mof.compute(exclude_cls=args.exclude)

        return_stats, _ = f1.compute(exclude_cls=args.exclude, pred_to_gt=pred_to_gt)
        f1_activity = return_stats['f1']

        miou_activity, _ = miou.compute(exclude_cls=args.exclude, pred_to_gt=pred_to_gt)

        epoch_mof_activity_list.append(mof_activity)
        epoch_f1_activity_list.append(f1_activity)
        epoch_miou_activity_list.append(miou_activity)

        #calculate metrics for per-video hungarian matching.
        # mof_video_avg, n_correct, f1_video, f1_video_macro, miou_video = 0, 0, 0, 0, 0
        # n_videos = video_metrics.__len__()
        n_frames = mof.gt_labels.__len__()
        n_frames_ = np.where(np.array(mof.gt_labels) != -1)[0].__len__() #number of frames without background (-1 is bg for YTI)
        
        # for ind in range(video_metrics.__len__()):
        #     n_correct += video_metrics[ind]['mof'] * video_metrics[ind]['n_frames']
        #     mof_video_avg += video_metrics[ind]['mof']
        #     f1_video += video_metrics[ind]['f1']
        #     #f1_video_macro += video_metrics[ind]['f1_macro']
        #     miou_video += video_metrics[ind]['miou']

        # #import ipdb;ipdb.set_trace()
        # mof_video = n_correct / n_frames
        # mof_video_avg /= n_videos #This is the way TW-FINCH computes MoF, but is not correct in my opinion.
        # f1_video /= n_videos
        # miou_video /= n_videos

        if args.dataset == 'YTI':
            print('PEOT {} {} frames {} epoch {:02d} loss {:.2f} activity level MoF {:.3f} F1 {:.3f} mIoU {:.3f} '
            ''.format(args.dataset, args.activity, n_frames_, epoch, epoch_loss, mof_activity, f1_activity, miou_activity))
            
            # print('PEOT {} {} [{}/{}] frames epoch {:02d} loss {:.2f} activity level MoF {:.3f} F1 {:.3f} mIoU {:.3f} '
            # 'video level MoF {:.3f} avg {:.3f} F1 {:.3f} mIoU {:.3f}'.format(args.dataset, args.activity, n_frames_, n_frames, epoch, epoch_loss, 
            #                                                                 mof_activity, f1_activity, miou_activity, 
            #                                                                 mof_video, mof_video_avg, f1_video, miou_video))
        else:
            print('PEOT {} {} frames {} epoch {:02d} loss {:.2f} activity level MoF {:.3f} F1 {:.3f} mIoU {:.3f}'.format(
                args.dataset, args.activity, n_frames, epoch, epoch_loss, mof_activity, f1_activity, miou_activity))
            #import ipdb;ipdb.set_trace()
            
            # print('PEOT {} {} frames {} epoch {:02d} loss {:.2f} activity level MoF {:.3f} F1 {:.3f} mIoU {:.3f} '
            # 'video level MoF {:.3f} avg {:.3f} F1 {:.3f} mIoU {:.3f}'.format(args.dataset, args.activity, n_frames, epoch, epoch_loss, 
            #                                                                 mof_activity, f1_activity, miou_activity, 
            #                                                                 mof_video, mof_video_avg, f1_video, miou_video))

        mof.reset()
        f1.reset()
        miou.reset()

    # epoch_loss_list = torch.tensor(epoch_loss_list)
    # epoch_mof_activity_list = torch.tensor(epoch_mof_activity_list)
    # epoch_f1_activity_list = torch.tensor(epoch_f1_activity_list)
    # epoch_miou_activity_list = torch.tensor(epoch_miou_activity_list)

#import ipdb;ipdb.set_trace()