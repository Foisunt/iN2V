from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch_geometric.nn import SimpleConv
from torch_geometric.data import Data
import torch_geometric.transforms as T


from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index, to_dense_adj, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr
from sklearn.linear_model import LogisticRegression

class InductiveN2V(torch.nn.Module):
    def __init__(self, edge_index: Tensor, train_mask, num_nodes: int, embedding_dim: int, walk_length: int, context_size: int, 
                 walks_per_node: int = 1,  p: float = 1.0, q: float = 1.0, 
                 num_negative_samples: int = 1, device = "cuda:0"):
        super().__init__()

        self.sc = SimpleConv(aggr="sum")
        self.mean_conv = SimpleConv("mean")

        if WITH_PYG_LIB and p == 1.0 and q == 1.0:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            if p == 1.0 and q == 1.0:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires either the 'pyg-lib' or "
                                  f"'torch-cluster' package")
            else:
                raise ImportError(f"'{self.__class__.__name__}' "
                                  f"requires the 'torch-cluster' package")

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        self.edge_index = edge_index

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col
        self.rowptr_c, self.col_c = self.rowptr.to(device), self.col.to(device)
        

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        #make sure that the transductive and inductive versions are initialized with the same embedding, that the resulting embeddings are compareable
        embedding_full = Embedding(len(train_mask), embedding_dim, sparse=True)
        self.embedding = Embedding(self.num_nodes, embedding_dim, sparse=True)
        self.embedding.weight.data = embedding_full.weight[train_mask]
        #self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()


    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]


    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    @torch.jit.export
    def loss_emb(self, pos_rw: Tensor, neg_rw: Tensor, prob_replace, device) -> Tensor:
        emb = self.embedding(torch.tensor([x for x in range(self.num_nodes)], device=device))
        mean = self.mean_conv(emb, self.edge_index)
        replace = torch.rand((self.num_nodes)) < prob_replace
        emb[replace]= mean[replace]
        
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = emb[start].view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = emb[rest.view(-1)].view(pos_rw.size(0), -1, self.embedding_dim)
    
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
    
        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
    
        h_start = emb[start].view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = emb[rest.view(-1)].view(neg_rw.size(0), -1, self.embedding_dim)
    
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
    
        return pos_loss + neg_loss

    # pull mean neighbor and self together
    @torch.jit.export
    def loss_mean_self(self, ixs, device):
        if ixs == None:
            ixs = torch.tensor([i for i in range(self.num_nodes)])
        emb = self.embedding(torch.tensor([x for x in range(self.num_nodes)], device=device))
        mean = self.mean_conv(emb, self.edge_index)

        h_start = emb[ixs]
        h_rest = mean[ixs]

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        return pos_loss

    def matrix_cos_sim(self, A):
        A = F.normalize(A)
        return torch.mm(A,A.transpose(0,1))
    
    # #rowptr, col
    # @torch.jit.export
    # def neighbor_loss_perm(self, batch_ix):
    #     drop_short = self.rowptr_c.index_select(0, batch_ix+1) - self.rowptr_c.index_select(0, batch_ix) > 1
    #     start, end = (self.rowptr_c.index_select(0, batch_ix)[drop_short], self.rowptr_c.index_select(0, batch_ix+1)[drop_short])
    #     acc = 0
    #     l = len(start)
    #     for i in range(l):
    #         n = self.col_c[start[i]:end[i]]
    #         e = F.normalize(self.embedding(n))
    #         acc = acc + F.pairwise_distance(e, e[torch.randperm(len(n))]).mean()/l
    #     return acc
            
    @torch.jit.export
    def neighbor_loss_full(self, batch_ix):
        drop_short = self.rowptr_c.index_select(0, batch_ix+1) - self.rowptr_c.index_select(0, batch_ix) > 1
        start, end = (self.rowptr_c.index_select(0, batch_ix)[drop_short], self.rowptr_c.index_select(0, batch_ix+1)[drop_short])
        acc = torch.tensor([0], device=batch_ix.device)
        l = len(start)
        for i in range(l):
            n = self.col_c[start[i]:end[i]]
            acc = acc + self.matrix_cos_sim(self.embedding(n)).mean()/l
        return acc

    def set_extend(self, alpha, delay, max_iter):
        self.alpha = alpha
        self.delay = delay
        self.max_iter = max_iter
        if alpha ==1:
            self.ext = self.extend2ind_fix
        else:
            self.ext = self.extend2ind_update

    def extend(self, embeds, data):
        return self.ext(embeds, data)

    def set_extend_m(self, comb, max_iter):
        self.combinations = comb
        self.max_iter = max_iter
        
    def extend_m(self, embeds, data):
        ls = []
        for d,a in self.combinations:
            self.alpha = a
            self.delay = d
            if a == "fp":
                ls.append(self.extend2ind_featureprop(embeds, data))
            elif a in ["no", "only", "all"]:
                ls.append(self.extend2ind_baseline(embeds, data))
            elif a==1:
                ls.append(self.extend2ind_fix(embeds, data))
            else:
                ls.append(self.extend2ind_update(embeds, data))
        return ls
        
    # ds is either val or test ds
    # embeds are the train embeds
    # return features for all nodes
    @torch.jit.export
    def extend2ind_fix(self, embeds, data):
        embed_size = embeds.shape[1]
        ds_size = data.x.shape[0]

        feats = torch.zeros((ds_size, embed_size), device=embeds.device)
        feats[data.train_mask] = embeds
        has_value = torch.zeros(ds_size, dtype=torch.bool, device=embeds.device)
        has_value[data.train_mask] = True

        for j in range(self.max_iter):
            feats_new = self.sc(feats, data.edge_index)
            num_n_with_value = self.sc(has_value.type(torch.int).unsqueeze(1), data.edge_index) #count neighbors that have a value
            tmp = (feats_new!=0).any(1) #all nodes wich now have an embedding #bug (unconnected old nodes)
            if has_value.logical_or(tmp).sum() == has_value.sum():#no changes -> can stop
                break
            additions = tmp.logical_and(torch.logical_not(has_value)) #only nodes which just got an embedding
            has_value = has_value.logical_or(tmp)
            #return feats_new, num_n_with_value, additions
            feats[additions] = feats_new[additions]/num_n_with_value[additions] #add new nodes to embedding; index first, then divide to prevent /0
        return feats, has_value

    #embdes[add] = mean[add]
    #embdes[existing&new!=0] = (1-a)mean[] + a existing[]
    #delay = number of iters to continue updating after all values are set
    @torch.jit.export
    def extend2ind_update(self, embeds, data):
        delay = self.delay
        embed_size = embeds.shape[1]
        ds_size = data.x.shape[0]

        feats = torch.zeros((ds_size, embed_size), device=embeds.device)
        feats[data.train_mask] = embeds
        has_value = torch.zeros(ds_size, dtype=torch.bool, device=embeds.device)
        has_value[data.train_mask] = True

        for j in range(self.max_iter):
            feats_new = self.sc(feats, data.edge_index)
            num_n_with_value = self.sc(has_value.type(torch.int).unsqueeze(1), data.edge_index) #count neighbors that have a value
            tmp = (feats_new!=0).any(1) #all nodes wich now have an embedding; unconnected old nodes get 0 bc of sc
            if has_value.logical_or(tmp).sum() == has_value.sum():#no changes -> can stop
                if delay > 0:
                    delay -= 1
                else:
                    break
            additions = tmp.logical_and(torch.logical_not(has_value)) #only nodes which just got an embedding
            feats[additions] = feats_new[additions]/num_n_with_value[additions] #add new nodes to embedding; index first, then divide to prevent /0
            updates = tmp.logical_and(has_value)            
            feats[updates] = self.alpha*feats[updates]+(1-self.alpha)*feats_new[updates]/num_n_with_value[updates] #update exisint with a*old+(1-a)*mean
            has_value = has_value.logical_or(tmp)                
        return feats, has_value

    @torch.jit.export
    def extend2ind_featureprop(self, embeds, data):
        embed_size = embeds.shape[1]
        ds_size = data.x.shape[0]
        feats = torch.zeros((ds_size, embed_size), device=embeds.device)
        feats[data.train_mask] = embeds
        
        data_fp = Data(x=feats, edge_index=data.edge_index)
        missing = ~data.train_mask.unsqueeze(1).expand(-1, embeds.shape[1]) #repeat second dimension
        fp = T.FeaturePropagation(missing_mask = missing, num_iterations = self.delay)
        data_fp = fp(data_fp)
        return data_fp.x, False

    @torch.jit.export
    def extend2ind_baseline(self, embeds, data):
        embed_size = embeds.shape[1]
        ds_size = data.x.shape[0]
        feats = torch.zeros((ds_size, embed_size), device=embeds.device)
        feats[data.train_mask] = embeds
        
        rescale_mask = feats.sum(1)!=0
        old_scales = feats.sum(1)
        scaler = torch.ones_like(old_scales)
        mi, ma = feats.min(), feats.max()
        A = torch.nn.functional.normalize(to_dense_adj(add_remaining_self_loops(data.edge_index)[0])[0,:,:], 1) #e +self loops -> A -> row normalize
        for i in range(self.delay):
            feats = torch.mm(A, feats)
            if self.alpha=="all":
                feats = torch.nn.functional.normalize(feats)
            elif self.alpha == "only":
                scaler[rescale_mask] = (old_scales[rescale_mask]/feats.sum(1)[rescale_mask])
                feats = (feats*scaler.unsqueeze(-1).expand(-1, feats.shape[1])).clamp(min=mi, max=ma)
        return feats, False


    def test_m(self, feats, data, mask_str, solver: str = 'lbfgs'):
        accs = []
        for f,_ in feats:
            accs.append(self.test(f, data, mask_str, solver))
        return torch.tensor(accs)
    
    @torch.no_grad()
    def test(self, feats, data, mask_str, solver: str = 'lbfgs'):
        train_z = feats[data.train_mask]
        train_y = data.y[data.train_mask]
        test_z = feats[data[mask_str]]
        test_y = data.y[data[mask_str]]
        
        clf = LogisticRegression(solver=solver, max_iter = 500, #max iter default is 100
                                 ).fit(train_z.detach().cpu().numpy(),train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    @torch.no_grad()
    def test_tr(self, feats, lbls, train_mask, test_mask, solver: str = 'lbfgs'):
        train_z = feats[train_mask]
        train_y = lbls[train_mask]
        test_z = feats[test_mask]
        test_y = lbls[test_mask]
        
        clf = LogisticRegression(solver=solver, max_iter = 500, #max iter default is 100
                                 ).fit(train_z.detach().cpu().numpy(),train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')

