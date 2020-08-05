import numpy as np
import torch
import os

class AttrPredictor(object):

    def __init__(self, cfg, tops_type=[3, 5, 10]):
        """Create the empty array to count true positive(tp),
            true negative(tn), false positive(fp) and false negative(fn).

        Args:
            class_num : number of classes in the dataset
            tops_type : default calculate top3, top5 and top10
        """

        attr_cloth_file = open(cfg.attr_cloth_file).readlines()
        self.attr_idx2name = {}
        for i, line in enumerate(attr_cloth_file[2:]):
            self.attr_idx2name[i] = line.strip('\n').split()[0]

    def print_attr_name(self, pred_idx):
        for idx in pred_idx:            
            #print(self.attr_idx2name[idx],idx)
            with open('results.txt','a') as f:
              f.write(str(idx)+',')
        with open('results.txt','a') as f:
            f.write('\n')
        for idx in pred_idx:
            with open('results.txt','a') as f:
              f.write(str(self.attr_idx2name[idx])+',')
              
    def show_prediction(self, pred,filename):
        if isinstance(pred, torch.Tensor):
            data = pred.data.cpu().numpy()
        elif isinstance(pred, np.ndarray):
            data = pred
        else:
            raise TypeError('type {} cannot be calculated.'.format(type(pred)))
        with open('results.txt','a') as f:
              f.write('\n'+str(filename)+'\n')
        for i in range(pred.size(0)):
            indexes = np.argsort(data[i])[::-1]
            idx5= indexes
            
            #print('[ Top5 Prediction ]')
            self.print_attr_name(idx5)
