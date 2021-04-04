from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_res_block(args):
    in_channel = args.res_channel
    layer = []
    if args.model_type == "CNN":
        layer.append(nn.Conv1d(in_channel, in_channel, 3, padding=1))
    elif args.model_type == "MLP":
        layer.append(nn.Linear(in_channel, in_channel, padding=1))
    layer.append(args.norm(in_channel))
    layer.append(args.activation())
    if args.dropout:
        layer.append(nn.Dropout(0.5))
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block1 = nn.Sequential(
            *get_res_block(args))
        self.block2 = nn.Sequential(
            *get_res_block(args))

    def forward(self, x):
        residual = x
        output = self.block1(x)
        output = self.block2(output)
        final = residual + output
        return final
