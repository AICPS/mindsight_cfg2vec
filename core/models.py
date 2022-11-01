'''
    This material is based upon work supported by the
    Defense Advanced Research Projects Agency (DARPA)
    and Naval Information Warfare Center Pacific
    (NIWC Pacific) under Contract Number N66001-20-C-4024.

    The views, opinions, and/or findings expressed are
    those of the author(s) and should not be interpreted
    as representing the official views or policies of
    the Department of Defense or the U.S. Government.

    Distribution Statement "A" (Approved for Public Release,
    Distribution Unlimited) 
'''

import torch
import torch.nn as nn
from torch.nn import Sequential

from torch_geometric.nn import GCNConv, GATConv, global_add_pool

class cfg2vecCFG(torch.nn.Module):
    def __init__(self, num_layers, layer_spec, initial_dim, dropout, pcode2vec='None'):
        super(cfg2vecCFG, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_spec = layer_spec.split(',')
        self.num_features_bb = initial_dim + 12 if 'bb' in pcode2vec else 12
        self.num_features_func = initial_dim if 'func' in pcode2vec else 0
        
        self.convs = []
        self.fc_dim = sum([int(dim) for dim in self.layer_spec])+self.num_features_bb
        for layer_idx in range(num_layers):
            in_dim = int(self.layer_spec[layer_idx-1]) if layer_idx > 0 else self.num_features_bb
            out_dim= int(self.layer_spec[layer_idx])
            self.convs.append(GCNConv(in_dim, out_dim))
        self.fc = nn.Linear(self.fc_dim, self.fc_dim)
        self.gconv_layers = Sequential(*self.convs)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc_final = nn.Linear(self.fc_dim + self.num_features_func, self.fc_dim)
    
    def forward(self, x, edge_index, batch, edge_index_cg, func_pcode_x=None):
        outputs = [x]
        for layer_idx in range(self.num_layers):
            #import pdb; pdb.set_trace()
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        if func_pcode_x is not None:
            #import pdb; pdb.set_trace()
            x = torch.cat((x, func_pcode_x.float()), dim=-1)
            x = self.fc_final(x)
        # x = torch.cat(x, funcsssss)
        # x = self.fc1(x)
        return x

class cfg2vecGoG(torch.nn.Module):
    def __init__(self, num_layers, layer_spec, initial_dim, dropout, pcode2vec='None'):
        super(cfg2vecGoG, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_spec = layer_spec.split(',') # ['32', '32']
        self.num_features_bb = initial_dim + 12 if 'bb' in pcode2vec else 12
        self.num_features_func = initial_dim if 'func' in pcode2vec else 0
        
        self.convs = []
        self.fc_dim = sum([int(dim) for dim in self.layer_spec])+self.num_features_bb

        for layer_idx in range(num_layers):
            in_dim = int(self.layer_spec[layer_idx-1]) if layer_idx > 0 else self.num_features_bb
            out_dim= int(self.layer_spec[layer_idx])
            self.convs.append(GCNConv(in_dim, out_dim))
        # self.convs = [GCNConv(12,32), GCNConv(32, 32)]
        self.acg_embed = GATConv(self.fc_dim, self.fc_dim) # GATConv(64, 64, heads=1)
        self.fc = nn.Linear(2*self.fc_dim, self.fc_dim) # Linear(in_features=128, out_features=64, bias=True)
        #import pdb; pdb.set_trace()
        self.gconv_layers = Sequential(*self.convs) # Sequential( (0): GCNConv(12, 32) (1): GCNConv(32, 32))
        self.dropout = nn.Dropout(p=self.dropout) # Dropout(p=0.0, inplace=False)
        self.fc_final = nn.Linear(self.fc_dim + self.num_features_func, self.fc_dim)
    
    def forward(self, x, edge_index, batch, edge_index_cg, func_pcode_x=None):
        outputs = [x]
        for layer_idx in range(self.num_layers):
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x_context = self.acg_embed(x, edge_index_cg).tanh()
        x = torch.cat([x, x_context], axis=1)
        x = self.fc(x)
        if func_pcode_x is not None:
            #import pdb; pdb.set_trace()
            x = torch.cat((x, func_pcode_x.float()), dim=-1)
            x = self.fc_final(x)
        return x