from bdb import set_trace
from email.headerregistry import ContentDispositionHeader
import torch
import os
import numpy as np
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import ipdb


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train'):
        self.root_dir = args.data_root 
        if mode:
            self.data_dir = '%s/train' % self.root_dir
            self.ordered = True #不要亂設變數！！！
        else:
            self.data_dir = '%s/validate' % self.root_dir
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            if not d1.startswith('.'):
                for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                    if not d2.startswith('.'):
                        self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        # ipdb.set_trace()
        self.seed_is_set = True ## Whether the random seed is already set or not
        self.d = 0
        # self defined
        self.seq_len = args.n_past + args.n_future
        self.mode = mode
        self.img_Augmentation = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #[H, W, C] -> [C, H, W] and also scaling to [0.0, 1.0]
        ])
        self.csv_Augmentation = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return 256
        
    def get_seq(self, index):
        # if self.ordered:
        #     d = self.dirs[self.d]
        #     if self.d == len(self.dirs) - 1:
        #         self.d = 0
        #     else:
        #         self.d += 1
        # else:
        #     d = self.dirs[np.random.randint(len(self.dirs))]
        data_dir = self.dirs[index]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (data_dir, i)
            img = Image.open(fname)
            # if self.train:
            # ipdb.set_trace()
            img = self.img_Augmentation(img)
            image_seq.append(img)
        # [(3, 64, 64), (3, 64, 64), ..., (3, 64, 64)] -> (12, 3, 64, 64)
        image_seq = torch.stack(image_seq)
        return image_seq
    
    def get_csv(self, index):
        # d = self.dirs[self.d]
        data_dir = self.dirs[index]
        conds = []
        files = ["%s/%s" % (data_dir, "actions.csv"), \
                "%s/%s" % (data_dir, "endeffector_positions.csv")]
        for csv_file in files:
            with open(csv_file) as f_csv:
                rows = csv.reader(f_csv)
                cond_arr = np.array(list(rows)).astype(float)
                cond_tensor = self.csv_Augmentation(cond_arr)
            conds.append(cond_tensor)
        # [(30, 3), (30, 4)] -> (30 ,7)
        conds = torch.cat(conds, dim=2)[0]
        conds = conds[:self.seq_len,:].float()
        return conds
        # actions = pd.read_csv('%s/%s' % (d, 'actions.csv'), header = None)
        # position = pd.read_csv('%s/%s' % (d, 'endeffector_positions.csv'), header = None)
        # cond = np.concatenate((actions, position), axis=1)
        # # ipdb.set_trace()
        # cond = cond.astype(np.uint8)
        # cond = self.Augmentation(cond)
        # cond = cond.reshape(30, 7)
        # cond_seq = []
        # for i in range(self.seq_len):
        #     cond_seq.append(cond[i].reshape(1, 7))
        # cond_seq = np.concatenate(cond_seq, axis=0)

        # return cond_seq
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq(index)
        cond =  self.get_csv(index)
        return seq, cond