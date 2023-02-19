# --------------------------------------------------------
# Most borrow from Query2Label

# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class, dataname):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (28 for Intentonomy).
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        self.dataname = dataname

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        hidden_dim = transfomer.d_model
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc_num_class = GroupWiseLinear(num_class, hidden_dim, bias=True)
        if self.dataname == 'intentonomy':
            self.fc_middle = GroupWiseLinear(15, hidden_dim, bias=True)
            self.fc_coarse = GroupWiseLinear(9, hidden_dim, bias=True)


    def forward(self, input):
        src, pos = self.backbone(input)
        src, pos = src[-1], pos[-1]

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos)[0] # B,K,d
        out = []
        out.append(self.fc_num_class(hs[0][-1]))
        if self.dataname == 'intentonomy':
            out.append(self.fc_middle(hs[1][-1]))
            out.append(self.fc_coarse(hs[2][-1]))

        return out

    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class,
        dataname = args.dataname
    )

    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    

    return model
        
        
