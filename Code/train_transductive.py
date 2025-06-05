import torch
import pandas as pd
from pathlib import Path
import time

import warnings

from train_n2v import train_epoch #train is the same (just always on whole graph), val -> early stopping works differently

from utils import EarlyStoppingAls


#eval_df = train_eval_model(model, ds, settings, model_path)
def train_eval_n2v_transductive(model, ds, settings, rep):
    loader = model.loader(batch_size=settings["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=settings["lr"])

    combinations = ["145", "24", "43", "62", "81"]
    if settings["split"] != "trans":
        combinations = [settings["split"].split("_")[1]]
    else:
        assert settings["save_embeds"] == False
    ES = EarlyStoppingAls(settings["patience"], combinations)
    
    data_train = ds.get_train()
    train_splits = ds.get_all_splits("train")#get train splits (all 145, 24, ...) )for train 
    val_splits = ds.get_all_splits("valid")#get val splits for val

    losses = []
    still_running = [True]*len(combinations)
    ends = [-1]*len(combinations)

    for e in range(settings["max_epochs"]):
        # if e%10==0:
        #     print("epoch", e)
        l_n2v, l_ms, l_ndiv = train_epoch(model, loader, optimizer, settings["device"], settings["prob_replace"], settings["w_ms"], settings["w_ndiv"])
        losses.append((l_n2v, l_ms, l_ndiv))

        feats = model().data
        accs = []
        for i in range(len(combinations)):
            if still_running[i]:
                split = combinations[i]
                accs.append(model.test_tr(feats, data_train.y, train_splits[split], val_splits[split]))
        accs = torch.tensor(accs)
        
        ret = ES(accs, model)
        if len(ret) == len(combinations):
            if settings["save_embeds"] != False:
                name=settings["name"]
                Path("../chkp/"+name).mkdir(parents=True, exist_ok=True)
                ls =  ES.get_ixs()
                assert len(ls)==1
                model = ES.load_checkpoint(ls[0]).to(settings["device"])
                feats = model().data
                torch.save(feats, "../chkp/"+name+"/emb_rep_"+str(rep)+".pt")
                
            for ix in range(len(combinations)):
                if still_running[ix]:
                    ends[ix] = e-settings["patience"]
            #d = {"split_val":combinations, "trained_epochs":ends, "val_acc":ret}
            #print(d)
            return {"split_val":combinations, "trained_epochs":ends, "val_acc":ret}
        for ix in ret:
            ends[ix] = e-settings["patience"]
            still_running[ix] = False
        
        
        #return {"split":["145", "24", "43", "62", "81"], "trained_epochs":ends, "val_acc":r, "loss_hist":[losses]*len(combinations)}
        
        
    raise NotImplementedError("training took over max_epochs epochs, dealing with this is not implemented")    
    #get all accs from ES/eval and return those
    #warnings.warn("training reached the end of max_epochs without EarlyStopping")
    


