import torch
from torch import nn
from torch.nn import Dropout, Linear, LayerNorm, GELU
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from inductiveN2V import InductiveN2V


def get_model(data_train, train_mask, settings):
    if settings["model"][:3] == "N2V":
        return InductiveN2V(data_train.edge_index, train_mask, data_train.x.shape[0], settings["embedding_dim"], settings["walk_len"], settings["context_size"], settings["walks_per_node"], 
                            settings["p"], settings["q"], settings["num_negative_samples"], settings["device"])
    elif settings["model"][:3].upper() == "MLP":
        return MLP(data_train.x.shape[1], int(data_train.y.max())+1, settings)
    elif settings["model"][:4].upper() == "SAGE":
        return SAGE(data_train.x.shape[1], int(data_train.y.max())+1, settings)
    elif settings["model"][:5].upper() == "FEATS":
        return InductiveN2V(data_train.edge_index, train_mask, data_train.x.shape[0], 1, -1, -2, -1, -1, -1, -1, settings["device"])
    else:
        raise NotImplementedError("model "+settings["model"]+" is not implemented")
        

#apply layernorm, activation, dropout
class Trans(nn.Module):
    def __init__(self, drop, norm_shape):
        super().__init__()
        self.d = Dropout(drop)
        self.a = GELU()
        self.n = LayerNorm(norm_shape)
    def forward(self, x):
        return self.d(self.a(self.n(x)))

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, settings):
        super().__init__()
        tmp = [int(x) for x in settings["model"].split("_")[1:]] #SAGE_numlayers_hid
        self.num_convs = tmp[0]
        self.h_in = in_channels
        self.h_out = out_channels
        self.h_hid = tmp[1]
        
        self.drop_model_p = settings["drop_model"]
        
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        if settings["model_type"] == "jk":
            self.jk_init(0)
            self.fwd = self.fwd_jk
        if settings["model_type"] == "jkb":
            self.jk_init(self.h_in)
            self.fwd = self.fwd_jkb
        else:
            self.lin_init()
            self.fwd = self.fwd_lin
        
    def jk_init(self, add):
        self.layers.append(SAGEConv(self.h_in, self.h_hid))
        for _ in range(self.num_convs-1):
            self.norms.append(Trans(self.drop_model_p, self.h_hid))
            self.layers.append(SAGEConv(self.h_hid, self.h_hid))
        hid_jk = self.num_convs * self.h_hid + add
        self.layers.append(Linear(hid_jk, self.h_out))
    
    def lin_init(self):
        if self.num_convs == 1:
            self.layers.append(SAGEConv(self.h_in, self.h_out))
        else:
            self.layers.append(SAGEConv(self.h_in, self.h_hid))
            for _ in range(self.num_convs-2):
                self.norms.append(Trans(self.drop_model_p, self.h_hid))
                self.layers.append(SAGEConv(self.h_hid, self.h_hid))
            self.norms.append(Trans(self.drop_model_p, self.h_hid))
            self.layers.append(SAGEConv(self.h_hid, self.h_out))

    #fwd method with jk
    def fwd_jk(self, data):
        x, e = data.x, data.edge_index
        outs = [self.layers[0](x, e)]
        for i in range(self.num_convs-1):
            outs.append(self.layers[i+1](self.norms[i](outs[-1]), e))
        return F.log_softmax(self.layers[-1](torch.cat(outs, dim=1)), dim=1)

    #fwd method with jk including the original feature
    def fwd_jkb(self, data):
        outs, e = [data.x], data.edge_index
        outs.append(self.layers[0](outs[-1], e))
        for i in range(self.num_convs-1):
            outs.append(self.layers[i+1](self.norms[i](outs[-1]), e))
        return F.log_softmax(self.layers[-1](torch.cat(outs, dim=1)), dim=1)

    #fwd method without residual connection for vertex classification
    def fwd_lin(self, data):
        x, e = data.x, data.edge_index
        x = self.layers[0](x, e)
        for i in range(self.num_convs-1):
            x = self.layers[i+1](self.norms[i](x), e)
        return F.log_softmax(x, dim=1)
    
    def forward(self, data):
        return self.fwd(data)
        
    @property
    def device(self):
        return next(self.parameters()).device


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, settings):
        super().__init__()
        tmp = [int(x) for x in settings["model"].split("_")[1:]] #SAGE_numlayers_hid
        self.num_convs = tmp[0]
        self.h_in = in_channels
        self.h_out = out_channels
        self.h_hid = tmp[1]
        
        self.drop_model_p = settings["drop_model"]

        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        if settings["model_type"] == "jk":
            self.jk_init(0)
            self.fwd = self.fwd_jk
        if settings["model_type"] == "jkb":
            self.jk_init(self.h_in)
            self.fwd = self.fwd_jkb
        else:
            self.lin_init()
            self.fwd = self.fwd_lin

    def lin_init(self):
        if self.num_convs == 1:
            self.layers.append(Linear(self.h_in, self.h_out))
        else:
            self.layers.append(Linear(self.h_in, self.h_hid))
            for _ in range(self.num_convs-2):
                self.norms.append(Trans(self.drop_model_p, self.h_hid))
                self.layers.append(Linear(self.h_hid, self.h_hid))
            self.norms.append(Trans(self.drop_model_p, self.h_hid))
            self.layers.append(Linear(self.h_hid, self.h_out))

    def jk_init(self, add):
        self.layers.append(Linear(self.h_in, self.h_hid))
        for _ in range(self.num_convs-1):
            self.norms.append(Trans(self.drop_model_p, self.h_hid))
            self.layers.append(Linear(self.h_hid, self.h_hid))
        hid_jk = self.num_convs * self.h_hid + add
        self.layers.append(Linear(hid_jk, self.h_out))


    def fwd_lin(self, data):
        x = data.x
        x = self.layers[0](x)
        for i in range(self.num_convs-1):
            x = self.layers[i+1](self.norms[i](x))
        return F.log_softmax(x, dim=1)

    def fwd_jk(self, data):
        outs = [self.layers[0](data.x)]
        for i in range(self.num_convs-1):
            outs.append(self.layers[i+1](self.norms[i](outs[-1])))
        return F.log_softmax(self.layers[-1](torch.cat(outs, dim=1)), dim=1)

    def fwd_jkb(self, data):
        outs = [data.x, self.layers[0](data.x)]
        for i in range(self.num_convs-1):
            outs.append(self.layers[i+1](self.norms[i](outs[-1])))
        return F.log_softmax(self.layers[-1](torch.cat(outs, dim=1)), dim=1)


    def forward(self, data):
        return self.fwd(data)
    
    @property
    def device(self):
        return next(self.parameters()).device

