import torch
import pandas as pd
from pathlib import Path
import time
import warnings
from utils import EarlyStoppingAls


#eval_df = train_eval_model(model, ds, settings, model_path)
def train_eval_n2v(model, ds, settings, rep):
    loader = model.loader(batch_size=settings["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=settings["lr"])

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
    ES = EarlyStoppingAls(settings["patience"], combinations)
    model.set_extend_m(combinations, settings["max_iter"]) # alpha and delay are lists to save unneccesary computation

    data_val = ds.get_val()

    running_combs = combinations.copy()
    ends = [-1]*len(combinations)

    #losses = []
    
    for e in range(settings["max_epochs"]):
        #print("epoch", e)
        l_n2v, l_ms, l_ndiv = train_epoch(model, loader, optimizer, settings["device"], settings["prob_replace"], settings["w_ms"], settings["w_ndiv"])
        #losses.append((l_n2v, l_ms, l_ndiv))
        
        feats_ex = model.extend_m(model().data, data_val)
        acc = model.test_m(feats_ex, data_val, "val_mask")
        
        #test multiple

        r = ES(acc, model)
        if len(r) == len(combinations):
            ends = [e-settings["patience"] if x==-1 else x-settings["patience"] for x in ends]
            if settings["save_embeds"] != False:
                name=settings["name"]
                data_test = ds.get_test() #need test graph to predict all embeddings
                Path("../chkp/"+name).mkdir(parents=True, exist_ok=True)
                ls =  ES.get_ixs()
                if len(ls)==1: #rename to just rep
                    model = ES.load_checkpoint(ls[0]).to(settings["device"])
                    feats = model.extend_m(model().data, data_test)[0] #just one delay alpha comb -> unpack from list
                    torch.save(feats, "../chkp/"+name+"/emb_rep_"+rep+".pt")
                else: #keep name with delay alpha; might save all combinations for each combination (bc extend_m might return a list of all of them)-> todo check
                    raise NotImplementedError("select correct delay alpha comb according the the path loaded (when multiple delay alpha combs are available)")
                #     for p in ls: #parent removes the rep_ appendix of the path
                #         model.load_state_dict(torch.load(p))
                #         feats = model.extend_m(model().data, data_test) #todo select the one according to p (others might be at wrong epoch)
                #         torch.save(feats, "../chkp/"+model_path.parts[2]+"/emb_"+ls[0].parts[-1])
                #         #save feats to normal path
                #         p.unlink()
                # #raise NotImplementedError("saving embeds / copying from tmp to fixed path not implemented")    
                #copy from tmp to fix path
            return {"delay_alpha":comb_str, "trained_epochs":ends, "val_acc":r}#, "loss_hist":[losses]*len(combinations)}
            
        if len(r)>0:
            updated = True
            for d in r:
                ends[d] = e
                running_combs.remove(combinations[d])
            model.set_extend_m(running_combs, settings["max_iter"]) # no need of continuing generating embeds for out of patience settings

    raise NotImplementedError("training took over max_epochs epochs, dealing with this is not implemented")    
    #get all accs from ES/eval and return those
    #warnings.warn("training reached the end of max_epochs without EarlyStopping")
    



def train_epoch(model, loader, optimizer, device, prob_replace, w_ms, w_ndiv):
    model.train()
    ix_ls = list(torch.arange(model.num_nodes, device=device).chunk(len(loader))) # get one of each ix in random order in chunks; this is more efficient thatn using pos_rw (mean is taken anyway)
                                                                                  # usually len(pos_rw) >> len(ix) (depending on walks/node, batch size, walk len-window size, 
    t_loss_n2v, t_loss_ms, t_loss_ndiv = 0, 0, 0
    loss_ms, loss_ndiv = torch.tensor([0], device=device), torch.tensor([0], device=device)
    
    for pos_rw, neg_rw in loader:
        ix = ix_ls.pop(0) #pop first element
        optimizer.zero_grad()
        if prob_replace >0:
            loss_n2v = model.loss_emb(pos_rw.to(device), neg_rw.to(device), prob_replace, device)
        else:
            loss_n2v = model.loss(pos_rw.to(device), neg_rw.to(device))
        if w_ms > 0:
            loss_ms = model.loss_mean_self(ix, device)
        if w_ndiv > 0:
            loss_ndiv = model.neighbor_loss_full(ix)
            #loss_ndiv = model.neighbor_loss_perm(pos_rw[:, 0].to(device))
        loss = loss_n2v + w_ms*loss_ms + w_ndiv*loss_ndiv
        loss.backward()
        optimizer.step()
        t_loss_n2v += loss_n2v.item()
        t_loss_ms  += loss_ms.item()
        t_loss_ndiv += loss_ndiv.item()
    return t_loss_n2v/len(loader), t_loss_ms/len(loader), t_loss_ndiv/len(loader)

