import torch
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import sys
import copy
import numpy as np

#Early Stopping on acc
class EarlyStoppingA():
    def __init__(self, patience):
        self.best_model_copy = None
        self.patience = patience
        self.counter = 0

        self.val_acc_max = -1.0
            
    def __call__(self, val_acc, model):
        if val_acc <= self.val_acc_max:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self,  model):
        self.best_model_copy = copy.deepcopy(model).to("cpu")
        
    def load_checkpoint(self):
        return self.best_model_copy

#Early Topping on both loss and acc
class EarlyStoppingLA():
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_model_copy = None

        self.val_loss_min = np.Inf
        self.val_acc_max = -1.0
            
    def __call__(self, val_loss, val_acc, model):
    #todo loss/acc option
        loss_worse = val_loss >= self.val_loss_min
        acc_worse = val_acc <= self.val_acc_max
        if loss_worse and acc_worse:
            self.counter += 1
            if self.counter >= self.patience:
                return self.val_acc_max, self.val_loss_min
        else:
            if not loss_worse:
                self.val_loss_min = val_loss
            if not acc_worse:
                self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self,  model):
        self.best_model_copy = copy.deepcopy(model).to("cpu")
        
    def load_checkpoint(self):
        return self.best_model_copy



#Early Stopping on acc, handeling a list of accs (for multiple alpha/delay combinations)
#returns a list of either settings that are done (len(ls) > len(comb)) or the final accs (len == comb list)
class EarlyStoppingAls():
    def __init__(self, patience, comb_ls):
        n = len(comb_ls)
        self.comb_ls = comb_ls
        self.patience = patience
        self.counts = torch.zeros(n)
        self.val_acc_max = torch.tensor([-1.0]*n)
        self.best_model_copy = {}
                
    def __call__(self, short_accs, model):
        still_running = self.counts <= self.patience
        new_accs = torch.zeros(len(still_running))
        new_accs[still_running] = short_accs        
        update = new_accs > self.val_acc_max
        ixs = still_running.logical_and(update) #double indexing a la a[ix1][ix2] = b[ix2] does not work
        self.save_checkpoint(ixs.nonzero(), model)
        self.val_acc_max[ixs] = new_accs[ixs]
        self.counts = self.counts + 1
        self.counts[ixs] = 0

        tmp = self.counts>self.patience
        if tmp.all():
            return self.val_acc_max

        r = []
        tmp2 = tmp*still_running
        if tmp2.any():
            r = tmp2.nonzero() #tell train which ones are still running
        return r

    def save_checkpoint(self, ixs, model):
        for ix in ixs:
            nam = self.comb_ls[ix]
            if type(nam) == tuple:
                nam = "_".join([str(x) for x in nam])
            self.best_model_copy[nam] = copy.deepcopy(model).to("cpu")
        
    def load_checkpoint(self, ix):
        return self.best_model_copy[ix]

    def get_ixs(self):
        return list(self.best_model_copy.keys())


def collect_res_dat(p = Path("../results/test/")):
    dfls = []
    for d in tqdm(list(p.glob("res*.pkl"))):
        dfls.append(pd.read_pickle(str(d)))
    return pd.concat(dfls, ignore_index=True)

def combine_results(name, res_path, tmp_res_path):
    p2 = Path("../results_comb/"+name+".pkl")
    df = collect_res_dat(tmp_res_path)
    df.to_pickle(p2)
    shutil.make_archive(str(res_path), 'xztar', tmp_res_path)
    shutil.rmtree(str(tmp_res_path)) #pathlib does not support deleting non empty directories


if __name__ == "__main__":
    print("combine results", sys.argv[-1])
    combine_results(sys.argv[-1])
