import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from metrics import pred_to_gt_match, filter_exclusions

# def plot_uot_imgs(features, loss_list, order_prior, opt_prior, temp_prior, codes, cost_matrix, opt_codes, gt, fname, ind=0):

#     gt_cpu = gt[ind].numpy()
#     gt_change = np.where(np.diff(gt_cpu) != 0)[0] + 1
#     # = np.where(np.diff(twfinch_labels) != 0)[0] + 1   
#     fdists = squareform(pdist(features[ind].detach().cpu().numpy(), 'cosine'))
    
#     plt.clf()
#     fig, axes = plt.subplots(2, 4, figsize=(16, 4))
#     plot = axes[0, 0].imshow(fdists)
#     for ch in gt_change:
#         axes[0, 0].axvline(ch, color='red')
#     for ch in gt_change:
#         axes[0, 0].axhline(ch, color='red')
#     axes[0, 0].set_title('{}'.format(fname))
    
#     plot = axes[0, 1].matshow(order_prior[ind].detach().cpu().T)
#     for ch in gt_change:
#         axes[0, 1].axvline(ch, color='red')
#     axes[0, 1].set_aspect('auto')
#     plt.colorbar(plot, ax=axes[0, 1])
#     axes[0, 1].set_title('order prior')

#     plot = axes[0, 2].matshow(opt_prior[ind].detach().cpu().T)
#     for ch in gt_change:
#         axes[0, 2].axvline(ch, color='red')
#     axes[0, 2].set_aspect('auto')
#     plt.colorbar(plot, ax=axes[0, 2])
#     axes[0, 2].set_title('optimal prior')

#     plot = axes[0, 3].matshow(temp_prior[ind].detach().cpu().T)
#     for ch in gt_change:
#         axes[0, 3].axvline(ch, color='red')
#     axes[0, 3].set_aspect('auto')
#     plt.colorbar(plot, ax=axes[0, 3])
#     axes[0, 3].set_title('optimal prior')

#     plot = axes[1, 0].plot(torch.Tensor(loss_list))
#     axes[1, 0].set_xlabel('Iterations', fontsize=12)
#     axes[1, 0].set_ylabel('Loss', fontsize=12)
#     axes[1, 0].grid()

#     plot = axes[1, 1].matshow(codes[ind].detach().cpu().T)
#     for ch in gt_change:
#         axes[1, 1].axvline(ch, color='red')
#     axes[1, 1].set_aspect('auto')
#     axes[1, 1].set_xticks([])
#     plt.colorbar(plot, ax=axes[1, 1])
#     axes[1, 1].set_title('codes')

#     plot = axes[1, 2].matshow(cost_matrix[ind].detach().cpu().T)
#     for ch in gt_change:
#         axes[1, 2].axvline(ch, color='red')
#     axes[1, 2].set_aspect('auto')
#     axes[1, 2].set_xticks([])
#     plt.colorbar(plot, ax=axes[1, 2])
#     axes[1, 2].set_title('cost matrix')
    
#     plot = axes[1, 3].matshow(opt_codes[ind].cpu().T)
#     for ch in gt_change:
#         axes[1, 3].axvline(ch, color='red')
#     axes[1, 3].set_aspect('auto')
#     axes[1, 3].set_xticks([])
#     plt.colorbar(plot, ax=axes[1, 3])
#     axes[1, 3].set_title('UOT pseudo label')
    
#     return plt

def plot_imgs(features, clusters, cost_matrix, temp_prior, opt_codes, codes, 
              gt, res_list, epoch_loss_list, fname, embeddings_tsne):
    
    gt_cpu = gt[0].cpu().numpy()
    gt_change = np.where(np.diff(gt_cpu) != 0)[0] + 1
    fdists = squareform(pdist(features[0].detach().cpu().numpy(), 'cosine'))

    colors = ['k','r','g','b','c','m','y']
    labels = []
    for i in gt.view(-1).cpu():
        ind = i.item() % len(colors)
        labels.append(colors[ind])

    plt.clf()
    fig, axes = plt.subplots(3, 3, figsize=(16, 8))
    axes[0, 0].matshow(temp_prior.cpu().numpy().T, )
    for ch in gt_change:
        axes[0, 0].axvline(ch, color='red')
    axes[0, 0].set_title('temporal prior')
    
    axes[0, 0].set_ylabel('Action index', fontsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].set_aspect('auto')

    import ipdb;ipdb.set_trace()
    axes[0, 1].matshow((1. - features @ clusters.T.unsqueeze(0))[0].detach().cpu().numpy().T, )
    for ch in gt_change:
        axes[0, 1].axvline(ch, color='red')

    axes[0, 1].set_title('{}'.format(fname[0]))
    axes[0, 1].set_ylabel('Action index', fontsize=12)
    axes[0, 1].set_aspect('auto')

    axes[0, 2].matshow(cost_matrix[0].detach().cpu().numpy().T)
    for ch in gt_change:
        axes[0, 2].axvline(ch, color='red')

    axes[0, 2].set_title('cost matrix')
    axes[0, 2].set_ylabel('Action index', fontsize=12)
    axes[0, 2].set_aspect('auto')

    axes[1, 0].imshow(fdists)
    fig.colorbar(axes[1, 0].imshow(fdists))
    axes[1, 0].set_title('self similarity')
    for ch in gt_change:
        axes[1, 0].axvline(ch, color='red')
    for ch in gt_change:
        axes[1, 0].axhline(ch, color='red')
        
    axes[1, 1].matshow(codes[0].detach().cpu().numpy().T)
    for ch in gt_change:
        axes[1, 1].axvline(ch, color='red')
        
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title('codes')
    axes[1, 1].set_ylabel('Action index', fontsize=12)
    axes[1, 1].set_aspect('auto')

    axes[1, 2].matshow(opt_codes[0].cpu().numpy().T)
    for ch in gt_change:
        axes[1, 2].axvline(ch, color='red')

    #axes[1, 2].set_title( 'pseudo label ub {:.2f}'.format(ub_term[0].item()) )
    axes[1, 2].set_title( 'pseudo label' )
    axes[1, 2].set_xticks([])
    axes[1, 2].set_ylabel('Action index', fontsize=12)
    axes[1, 2].set_aspect('auto')

    #import ipdb;ipdb.set_trace()
    fig.colorbar(axes[2, 0].imshow((clusters @ clusters.T).detach().cpu()))

    n_clusters = codes.shape[-1]
    axes[2, 1].scatter(embeddings_tsne[:-n_clusters, 0], embeddings_tsne[:-n_clusters, 1], c=labels, s=5)
    axes[2, 1].scatter(embeddings_tsne[-n_clusters:, 0], embeddings_tsne[-n_clusters:, 0], s=200, marker='*')

    if len(res_list) > 0:
        #import ipdb;ipdb.set_trace()
        axes[2, 2].plot(np.array(res_list)[:, 0], c='r', label='MoF')
        axes[2, 2].plot(np.array(res_list)[:, 1], c='b', label='F1')
        
        axes[2, 2].plot(torch.Tensor(epoch_loss_list), label='Loss')
        axes[2, 2].set_xlabel('Epochs', fontsize=12)
        #axes[2, 2].set_ylabel('Loss', fontsize=12)

        axes[2, 2].grid()
        axes[2, 2].legend(fontsize=10)
    return plt

def plot_pair_imgs(features, clusters, cost_matrix, opt_codes, codes, 
                   gt, res_list, epoch_loss_list, fname, embeddings_tsne):
    
    gt_cpu = gt[0].cpu().numpy()
    gt_change = np.where(np.diff(gt_cpu) != 0)[0] + 1

    gt_cpu_ = gt[1].cpu().numpy()
    gt_change_ = np.where(np.diff(gt_cpu_) != 0)[0] + 1
    fdists = squareform(pdist(features[0].detach().cpu(), 'cosine'))

    colors = ['k','r','g','b','c','m','y']
    labels = []
    for i in gt.view(-1).cpu():
        ind = i.item() % len(colors)
        labels.append(colors[ind])

    plt.clf()
    fig, axes = plt.subplots(3, 3, figsize=(16, 8))

    axes[0, 0].matshow(codes[0].detach().cpu().T)
    for ch in gt_change:
        axes[0, 0].axvline(ch, color='red')

    axes[0, 0].set_title('codes')
    #axes[0, 0].set_ylabel('Action index', fontsize=12)
    axes[0, 0].set_aspect('auto')

    axes[0, 1].matshow(cost_matrix[0].detach().cpu().T)
    for ch in gt_change:
        axes[0, 1].axvline(ch, color='red')
    axes[0, 1].set_title('cost matrix')
    axes[0, 1].set_aspect('auto')

    axes[0, 2].matshow(opt_codes[0].detach().cpu().T)
    for ch in gt_change:
        axes[0, 2].axvline(ch, color='red')
    axes[0, 2].set_title(fname[0])
    axes[0, 2].set_aspect('auto')


    #The second video
    axes[1, 0].matshow(codes[1].detach().cpu().T)
    for ch in gt_change_:
        axes[1, 0].axvline(ch, color='red')
    axes[1, 0].set_title('codes')
    axes[1, 0].set_aspect('auto')
    axes[1, 0].set_xticks([])

    axes[1, 1].matshow(cost_matrix[1].detach().cpu().T)
    for ch in gt_change_:
        axes[1, 1].axvline(ch, color='red')
    axes[1, 1].set_title('cost matrix')
    axes[1, 1].set_aspect('auto')
    axes[1, 1].set_xticks([])

    axes[1, 2].matshow(opt_codes[1].detach().cpu().T)
    for ch in gt_change_:
        axes[1, 2].axvline(ch, color='red')

    axes[1, 2].set_title(fname[1])
    axes[1, 2].set_xticks([])
    axes[1, 2].set_aspect('auto')

    fig.colorbar(axes[2, 0].imshow((clusters @ clusters.T).detach().cpu()))

    n_clusters = codes.shape[-1]
    axes[2, 1].scatter(embeddings_tsne[:-n_clusters, 0], embeddings_tsne[:-n_clusters, 1], c=labels, s=5)
    axes[2, 1].scatter(embeddings_tsne[-n_clusters:, 0], embeddings_tsne[-n_clusters:, 0], s=200, marker='*')

    if len(res_list) > 0:
        #import ipdb;ipdb.set_trace()
        axes[2, 2].plot(np.array(res_list)[:, 0], c='r', label='MoF')
        axes[2, 2].plot(np.array(res_list)[:, 1], c='b', label='F1')
        
        axes[2, 2].plot(0.5 * torch.Tensor(epoch_loss_list), label='Loss')
        axes[2, 2].set_xlabel('Epochs', fontsize=12)

        axes[2, 2].grid()
        axes[2, 2].legend(fontsize=10)
    return plt
















def plot_segmentation(pred, mask, name=''):
    colors = {}
    cmap = plt.get_cmap('tab20')
    uniq = np.unique(pred[mask].cpu().numpy())
    n_frames = len(pred)

    # add colors for predictions which do not match to a gt class

    for i, label in enumerate(uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / len(uniq))

    fig = plt.figure(figsize=(16, 2))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred[mask].cpu().numpy()[1:] - pred[mask].cpu().numpy()[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred[mask].cpu().numpy()[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_segmentation_gt(gt, pred, mask, gt_uniq=None, pred_to_gt=None, exclude_cls=None, name=''):
    colors = {}

    pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy(), exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())

    for pr_lab, gt_lab in zip(pred_opt, gt_opt):
        pred_[pred_ == pr_lab] = gt_lab
    n_frames = len(pred_)

    # add colors for predictions which do not match to a gt class
    if gt_uniq is None:
        gt_uniq = np.unique(gt_.cpu().numpy())
        
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    n_class = len(gt_uniq)
    if n_class <= 20:
        cmap = plt.get_cmap('tab20')
    else:  # up to 40 classes
        cmap1 = plt.get_cmap('tab20')
        cmap2 = plt.get_cmap('tab20b')
        cmap = lambda x: cmap1(round(x * n_class / 20., 2)) if x <= 19. / n_class else cmap2(round((x - 20 / n_class) * n_class / 20, 2))

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / n_class)

    fig = plt.figure(figsize=(12, 3))
    plt.axis('off')
    plt.title(name, fontsize=45, pad=20)

    # plot predicted segmentation after matching to gt labels w/Hungarian

    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel('Ours', fontsize=45, rotation=0, labelpad=60, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred_[1:] - pred_[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    # plot gt segmentation

    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel('GT', fontsize=45, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    gt_segment_boundaries = np.where(gt_[1:] - gt_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

    for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
        label = gt_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig

def plot_matrix(mat, gt=None, colorbar=True, title=None, figsize=(10, 5), ylabel=None, xlabel=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot1 = ax.matshow(mat)
    if gt is not None: # plot gt segment boundaries
        gt_change = np.where((np.diff(gt) != 0))[0] + 1
        for ch in gt_change:
            ax.axvline(ch, color='red')
    if colorbar:
        plt.colorbar(plot1, ax=ax)
    if title:
        ax.set_title(f'{title}')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.tick_params(axis='x', labelsize=24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=36)
        ax.tick_params(axis='y', labelsize=24)
    
    ax.set_aspect('auto')
    fig.tight_layout()
    return fig

def plot(cost_matrix, segmentation, gt, pred, mask, gt_uniq=None, pred_to_gt=None, exclude_cls=None, name=''):
    colors = {}

    pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy(), exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())

    for pr_lab, gt_lab in zip(pred_opt, gt_opt):
        pred_[pred_ == pr_lab] = gt_lab
    n_frames = len(pred_)

    # add colors for predictions which do not match to a gt class
    if gt_uniq is None:
        gt_uniq = np.unique(gt_.cpu().numpy())
        
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    n_class = len(gt_uniq)
    if n_class <= 20:
        cmap = plt.get_cmap('tab20')
    else:  # up to 40 classes
        cmap1 = plt.get_cmap('tab20')
        cmap2 = plt.get_cmap('tab20b')
        cmap = lambda x: cmap1(round(x * n_class / 20., 2)) if x <= 19. / n_class else cmap2(round((x - 20 / n_class) * n_class / 20, 2))

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / n_class)

    fig = plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    ax = fig.add_subplot(2, 1, 1)
    ax.matshow(cost_matrix.squeeze().detach().cpu().T)
    #ax.set_xlabel('Frame index', fontsize=12)
    #ax.tick_params(axis='x', labelsize=12)

    ax.set_ylabel('Action index', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_aspect('auto')

    ax = fig.add_subplot(2, 1, 2)
    ax.matshow(segmentation.squeeze().detach().cpu().T)
    #ax.set_xlabel('Frame index', fontsize=12)
    #ax.tick_params(axis='x', labelsize=12)
    ax.set_xticks([])

    ax.set_ylabel('Action index', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_aspect('auto')

    fig.tight_layout()
    return fig