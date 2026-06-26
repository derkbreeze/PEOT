import os
import numpy as np
import torch

def parse_action_name(fname, dataset):
    if dataset == 'Breakfast':
        #import ipdb;ipdb.set_trace() P03_cam01_P03_cereals
        return fname.split('_')[-1]
    if dataset == 'IKEA': 
        #import ipdb;ipdb.set_trace()
        #Lack_Side_Table_0043_oak_table_10_04_2019_08_28_15_31_dev2
        return '_'.join(fname.split('_')[:3])
    if dataset == 'YTI':  # ignores _idt files in groundTruth automatically
        return '_'.join(fname.split('_')[:-1])
    if dataset == 'FS':  # only one activity class
        return ''
    if dataset == 'desktop_assembly':  # only one activity class
        return ''
    raise ValueError(f'{dataset} is not a valid dataset!')

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset, n_frames, standardise=True, split=None, random=True, n_videos=None, action_class=['all']):

        self.root_dir = root_dir
        self.dataset = dataset
        if self.dataset == 'FSeval':
            self.dataset = 'FS'
            granularity = 'eval'
        else:
            granularity = None
        self.data_dir = os.path.join(self.root_dir, self.dataset)
        self.video_fnames = sorted([fname for fname in os.listdir(os.path.join(self.data_dir, 'groundTruth'))
                                    if len(fname.split('_')) > 1 or len(fname.split('-')) > 1])
        
        if self.dataset in ['FS', 'desktop_assembly']:
            action_class = ''
        if action_class != ['all']:
            if type(action_class) is list:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) in action_class]
            else:
                self.video_fnames = [fname for fname in self.video_fnames if parse_action_name(fname, self.dataset) == action_class]
        
        if n_videos is not None:
            # inds = np.random.permutation(len(self.video_fnames))[:n_videos]
            # self.video_fnames = sorted([self.video_fnames[ind] for ind in inds])
            self.video_fnames = self.video_fnames[::int(len(self.video_fnames) / n_videos)]

        def prep(x):
            i, nm = x.rstrip().split(' ') #'0000.txt\n'.rstrip() '0000.txt\n'.strip('\n')
            return nm, int(i)

        if granularity is None:  # granularity applies only to 50Salads
            action_mapping = list(map(prep, open(os.path.join(self.data_dir, 'mapping/mapping.txt'))))
        else:
            action_mapping = list(map(prep, open(os.path.join(self.data_dir, f'mapping/mapping{granularity}.txt'))))

        self.action_mapping = dict(action_mapping)
        self.n_subactions = len(self.action_mapping)
        self.n_frames = n_frames
        self.standardise = standardise
        self.random = random

    def __len__(self):
        return len(self.video_fnames)
    
    def __getitem__(self, idx):
        #import ipdb;ipdb.set_trace()
        video_fname = self.video_fnames[idx]
        gt = [line.rstrip() for line in open(os.path.join(self.data_dir, 'groundTruth', video_fname))]
        inds, mask = self._partition_and_sample(n_samples = self.n_frames, n_frames = len(gt))
        gt = torch.Tensor([self.action_mapping[gt[ind]] for ind in inds]).long()
        action = parse_action_name(video_fname, self.dataset)
        feat_fname = os.path.join(self.data_dir, 'features', action, video_fname)
        
        try:
            features = np.loadtxt(feat_fname + '.txt')[inds, :]
        except:
            features = np.load(feat_fname + '.npy')[inds, :]

        if self.standardise:  # normalize features
            zmask = np.ones(features.shape[0], dtype=bool)
            for rdx, row in enumerate(features):
                if np.sum(row) == 0:
                    zmask[rdx] = False

            #import ipdb;ipdb.set_trace()
            z = features[zmask] - np.mean(features[zmask], axis=0)
            z = z / np.std(features[zmask], axis=0)
            features = np.zeros(features.shape)
            features[zmask] = z
            features = np.nan_to_num(features)
            features /= np.sqrt(features.shape[1])
        # mask = torch.from_numpy(mask * zmask)
        features = torch.from_numpy(features).float()
        #import ipdb;ipdb.set_trace()
        return features, mask, gt, video_fname, gt.unique().shape[0]
    
    def _partition_and_sample(self, n_samples, n_frames):
        if n_samples is None:
            indices = np.arange(n_frames)
            mask = np.ones(n_frames, dtype=bool)
        elif n_samples < n_frames:
            if self.random:
                boundaries = np.linspace(0, n_frames-1, n_samples+1).astype(int)
                indices = np.random.randint(low=boundaries[:-1], high=boundaries[1:])
                #import ipdb;ipdb.set_trace()
            else:
                indices = np.linspace(0, n_frames-1, n_samples).astype(int)
            mask = np.ones(n_samples, dtype=bool)
        else:
            indices = np.concatenate((np.arange(n_frames), np.full(n_samples - n_frames, n_frames - 1)))
            mask = np.concatenate((np.ones(n_frames, dtype=bool), np.zeros(n_samples - n_frames, dtype=bool)))
        return indices, mask
