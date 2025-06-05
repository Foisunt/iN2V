import torch
import pandas as pd
from pathlib import Path
import time

import warnings

from utils import EarlyStoppingLA
from train_n2v import train_eval_n2v
from train_transductive import train_eval_n2v_transductive
from eval_orig import eval_in2v_graphfeats

def do_train(model, ds, settings, rep):
    if settings["model"] == "N2V":
        if str(settings["split"]).split("_")[0]=="trans":
            return train_eval_n2v_transductive(model, ds, settings, rep)
        return train_eval_n2v(model, ds, settings, rep)
    elif settings["model"] == "feats":
        return eval_in2v_graphfeats(model, ds, settings, rep)
    else:
        return train_eval_gnn(model, ds, settings)

#eval_df = train_eval_model(model, ds, settings, model_path)
def train_eval_gnn(model, ds, settings):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["wd"])
    ES = EarlyStoppingLA(settings["patience"])
    loss_fn = torch.nn.NLLLoss()

    data_train = ds.get_train() #inductive train data or transductive whole dataset
    data_val = ds.get_val() #inductive val data or same as train

    for e in range(settings["max_epochs"]):

        model.train()
        optimizer.zero_grad()
        preds = model(data_train)
        t_loss = loss_fn(preds[data_train.train_mask],data_train.y[data_train.train_mask])
        t_loss.backward()
        optimizer.step()
        
        model.eval()
        preds = model(data_val)
        loss = float(loss_fn(preds[data_val.val_mask],data_val.y[data_val.val_mask]))
        acc = float((preds[data_val.val_mask].argmax(-1) == data_val.y[data_val.val_mask]).sum()/data_val.val_mask.sum())

        best_acc_loss = ES(loss, acc, model)
        if best_acc_loss != False:
            model = ES.load_checkpoint().to(settings["device"])
            model.eval()
            data_test = ds.get_test()
            preds = model(data_test)
            test_loss = float(loss_fn(preds[data_test.test_mask],data_test.y[data_test.test_mask]))
            test_acc = float((preds[data_test.test_mask].argmax(-1) == data_test.y[data_test.test_mask]).sum()/data_test.test_mask.sum())
            
            return {"trained_epochs":[e-settings["patience"]], "val_loss":[best_acc_loss[1]], "val_acc":[best_acc_loss[0]], "test_loss":[test_loss], "test_acc":[test_acc]}

    preds = model(data_val)
    loss = float(loss_fn(preds[data_val.val_mask],data_val.y[data_val.val_mask]))
    acc = float((preds[data_val.val_mask].argmax(-1) == data_val.y[data_val.val_mask]).sum()/data_val.val_mask.sum())
        
    data_test = ds.get_test()
    preds = model(data_test)
    test_loss = float(loss_fn(preds[data_test.test_mask],data_test.y[data_test.test_mask]))
    test_acc = float((preds[data_test.test_mask].argmax(-1) == data_test.y[data_test.test_mask]).sum()/data_test.test_mask.sum())
    return {"trained_epochs":[settings["max_epochs"]], "val_loss":[loss], "val_acc":[acc], "test_loss":[test_loss], "test_acc":[test_acc]}

            
        