#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[b, D, N]
    :param p2: size[b, D, M]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, :, None] - p2[:, :, None, :]
    dist = torch.sum(diff*diff,  dim=1) #[B, N, M]
    # dist1 = dist
    # dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist, dim=1)
    dist_min2, _ = torch.min(dist, dim=2)

    return (torch.sum(dist_min1)/dist.shape[1] + torch.sum(dist_min2)/dist.shape[2])
    # return dist_min1, dist_min2

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
