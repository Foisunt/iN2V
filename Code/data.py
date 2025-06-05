import torch
from pathlib import Path
from torch_geometric.data import Data
import torch_geometric.transforms as T

from numpy import load #Chameleon and Squirrel filtered
from torch_geometric.datasets import Planetoid, Amazon, Actor, WikipediaNetwork, HeterophilousGraphDataset, WikiCS
from ogb.nodeproppred import PygNodePropPredDataset


class Dataset:
    def __init__(self, settings):
        transform_ls = []
        if settings["ds_make_undir"]:
            transform_ls.append(T.ToUndirected())
        if settings["ds_add_self"]:
            transform_ls.append(T.AddRemainingSelfLoops())
        tmp = str(settings["split"]).split("_")
        if tmp[0] == "trans": #transductive, but needs an exsisting split to work with load_ds; use test split for graph and get_all_splits for actual val
            self.split_name = "145"
            if len(tmp) == 2:
                self.split_name = tmp[1]
            self.transductive = True
        else:
            self.split_name = str(settings["split"])
            self.transductive = False
        self.ds_name = settings["dataset"]
        self.ds, self.splits = self.load_ds(transform_ls)
        self.device = settings["device"]
        self.mode = settings["datamode"]

    def load_ds(self, transform_ls):
        transform = None
        if transform_ls != []:
            transform = T.Compose(transform_ls)
        ds_name = self.ds_name
        splits = torch.load(str(Path("dataset/"+ds_name+"/own_"+self.split_name+"_splits.pt")), weights_only=True)
        if ds_name in ["Cora", "Citeseer", "Pubmed"]:
            ds = Planetoid(root='dataset/'+ds_name+"/", name=ds_name, transform=transform)[0]
        elif ds_name in ["Roman-empire", "Amazon-ratings"]:
            ds = HeterophilousGraphDataset(root='dataset/'+ds_name+"/", name=ds_name, transform=transform)[0]
        elif ds_name in ["Photo", "Computers"]:
            ds = Amazon(root='dataset/'+ds_name+"/", name=ds_name, transform=transform)[0]
        elif ds_name == "WikiCS":
            ds = WikiCS(root='dataset/'+ds_name+"/", is_undirected=True, transform=transform)[0]
        elif ds_name in ["Chameleon-f", "Squirrel-f"]:
            np_data = load({"Chameleon-f":"dataset/Chameleon-f/chameleon_filtered.npz", "Squirrel-f":"dataset/Squirrel-f/squirrel_filtered.npz"}[ds_name])
            ds = Data(x=torch.from_numpy(np_data["node_features"]), y=torch.from_numpy(np_data["node_labels"]), edge_index=torch.from_numpy(np_data["edges"]).transpose(0,1))
            raise NotImplementedError("transforms are not implemented for the filtered verions")
        elif ds_name in ["Chameleon", "Squirrel"]:
            ds = WikipediaNetwork(root="dataset/"+ds_name+"/", name=ds_name, transform=transform)[0]
        elif ds_name == "Actor":
            ds = Actor(root="dataset/Actor/", transform=transform)[0]
        elif ds_name == "ogbn-arxiv":
            data = PygNodePropPredDataset(root="dataset/ogbn-arxiv/", name = "ogbn-arxiv", transform = transform)[0]
            ds = Data(x=data.x, y=data.y.squeeze(), edge_index=data.edge_index)
        else:
            raise NotImplementedError("Dataset"+ds_name+"is not implemented")
        return ds, splits

    def set_rep(self, rep):
        self.rep = rep
        if self.mode == "gra": #use the original features
            x = self.ds.x.to(self.device)
        else: #load generated embeddings
            use, extend = self.mode.split("_")
            save_name = {"tr":"base", "ba":"base", "po":"post", "lo":"loss", "re":"prob", "transd":"tr", "fpbase":"base_fp", "fploss":"loss_fp", "fpprob":"prob_fp", "exfbase":"exf_base", "exffp":"exf_fp", "exfph":"exf_ph"}[extend] #tr was used as abbreviation for train, before transductive was implemented 
            emb =  torch.load("../chkp/E_"+self.ds_name+"_"+self.split_name+"_"+save_name+"/emb_rep_"+str(rep)+".pt", map_location = self.device)
            if type(emb)==tuple:
                emb = emb[0]
            if extend == "tr":
                emb[~self.splits["train"][:,rep]]=0 # set all non train to 0            
            if use == "emb":
                x = emb
            elif use == "cat":
                x = torch.cat([self.ds.x.to(self.device), emb], dim=1)
                
        tmp_data = Data(x=x.to(self.device), y=self.ds.y.to(self.device), edge_index=self.ds.edge_index.to(self.device), 
                        train_mask=self.splits["train"][:,rep].to(self.device), val_mask=self.splits["valid"][:,rep].to(self.device), test_mask=self.splits["test"][:,rep].to(self.device))
        if self.transductive:
            self.data_train = tmp_data
            self.data_val = tmp_data
            self.data_test = tmp_data
        else:
            split_val = self.splits["train"][:,rep].logical_or(self.splits["valid"][:,rep])
            self.data_train = tmp_data.subgraph(self.splits["train"][:,rep].to(self.device))
            self.data_val = tmp_data.subgraph(split_val.to(self.device))
            self.data_test = tmp_data
        
    def get_train_mask(self):
        if self.transductive:
            return torch.ones_like(self.splits["train"][:,self.rep])
        return self.splits["train"][:,self.rep]
    def get_train(self):
        return self.data_train
    def get_val(self):
        return self.data_val
    def get_test(self):
        return self.data_test
        
    #for transductive training we val with all splits (145, 24, ...) at once
    #get after setting rep
    def get_all_splits(self, tvt ="valid"):
        splits = {}
        for s in ["145", "24", "43", "62", "81"]:
            tmp = torch.load(str(Path("dataset/"+self.ds_name+"/own_"+s+"_splits.pt")), weights_only = True)
            splits[s]= tmp[tvt][:,self.rep]
        return splits

        


