import torch
import pandas as pd
from pathlib import Path
import time
import warnings

#use train feats of original grap (no n2v)
#hack into in2v to get extension and eval from there
def eval_in2v_graphfeats(model, ds, settings, rep):
    combinations = []
    if settings.get("alpha", False) != False:
        for y in settings["alpha"]:
            if y == 1:
                combinations.append((settings["delay"][0], y))
                
            else:
                combinations.extend([(x, y) for x in settings["delay"]])
    if settings.get("fp", False) != False: #feature propoagation
        combinations.extend([(x, "fp") for x in settings["fp"]])
    if settings.get("baselines", False) != False: #baseline methods
        combinations.extend([(int(x.split("_")[1]), x.split("_")[0]) for x in settings["baselines"]])
    assert len(combinations) > 0
    #combinations = [(x,y) for y in settings["alpha"] for x in settings["delay"]]
    
    comb_str = ["_".join([str(a) for a in x]) for x in combinations]
    model.set_extend_m(combinations, settings["max_iter"]) # alpha and delay are lists to save unneccesary computation

    data_train = ds.get_train()
    data_val = ds.get_val()
    
    feats_ex = model.extend_m(data_train.x, data_val)
    acc = model.test_m(feats_ex, data_val, "val_mask")

    if settings["save_embeds"] != False:
        assert len(acc) == 1
        name=settings["name"]
        data_test = ds.get_test() #need test graph to predict all embeddings
        Path("../chkp/"+name).mkdir(parents=True, exist_ok=True)
        
        feats = model.extend_m(data_train.x, data_test)[0] #just one delay alpha comb -> unpack from list
        
        torch.save(feats, "../chkp/"+name+"/emb_rep_"+str(rep)+".pt")
        
    return {"delay_alpha":comb_str, "val_acc":acc}
            





