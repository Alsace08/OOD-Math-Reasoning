import os
import argparse
import scipy.spatial
import json
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import math
from scipy.special import comb

from ID_info import *


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


class OODScore:
    def __init__(self, OOD_info, ID_info):
        self.total_layer_num = len(OOD_info["hidden_state"][0])
        self.ID_info = ID_info
        self.OOD_info = OOD_info

        self.ID_mean_hs_all_layer = np.array([self.ID_info[layer]["mean_hs"][0] for layer in range(self.total_layer_num)])
        self.ID_var_hs_all_layer = np.array([self.ID_info[layer]["var_hs"][0] for layer in range(self.total_layer_num)])
        self.OOD_hs_all_layer = np.array(self.OOD_info["hidden_state_mean_pos"])

    def get_tv_score(self, max_order):
        emb_dim = len(self.ID_mean_hs_all_layer[0])
        md_all_layer = []
        for layer_num in range(1, self.total_layer_num - max_order - 1):
            ### OOD
            OOD_layer_hs = []
            for order in range(max_order + 1):
                OOD_layer_hs.append(self.OOD_hs_all_layer[layer_num + order])
            OOD_layer_hs_diff = []
            for order in range(max_order + 1):
                tmp_diff = np.array([0] * emb_dim).astype(np.float64)
                for j in reversed(range(order + 1)):
                    tmp_diff += ((-1) ** (order - j)) * comb(order, order - j) * OOD_layer_hs[j]
                OOD_layer_hs_diff.append(tmp_diff)

            ### ID
            ID_mean_layer_hs, ID_var_layer_hs = [], []
            for order in range(max_order + 1):
                ID_mean_layer_hs.append(self.ID_mean_hs_all_layer[layer_num + order])
                ID_var_layer_hs.append(self.ID_var_hs_all_layer[layer_num + order])
            ID_mean_layer_hs_diff, ID_var_layer_hs_diff = [], []
            for order in range(max_order + 1):
                tmp_mean_diff, tmp_var_diff = np.array([0] * emb_dim).astype(np.float64), np.array([0] * emb_dim).astype(np.float64)
                for j in reversed(range(order + 1)):
                    tmp_mean_diff += ((-1) ** (order - j)) * comb(order, order - j) * ID_mean_layer_hs[j]
                    tmp_var_diff += comb(order, order - j) * ID_var_layer_hs[j]
                ID_mean_layer_hs_diff.append(tmp_mean_diff)
                ID_var_layer_hs_diff.append(tmp_var_diff)

            ### Mahalanobis Distance
            md_per_layer = []
            for i in range(max_order + 1):
                 md_per_layer.append(np.linalg.norm(
                [(OOD_layer_hs_diff[i][dim] - ID_mean_layer_hs_diff[i][dim]) ** 2 / ID_var_layer_hs_diff[i][dim]
                 for dim in range(len(ID_mean_layer_hs_diff[i]))], ord=2))
            md_all_layer.append(md_per_layer)

        tv_score_all = []
        tv_score_per_layer = []
        md_all_layer = np.array(md_all_layer)
        for i in range(max_order + 1):
            tv_score_per_layer_i = [abs(md_all_layer[:, i][j+1] - md_all_layer[:, i][j]) for j in range(len(md_all_layer[:, i]) - 1)]
            tv_score_all_i = np.mean(tv_score_per_layer_i)
            tv_score_per_layer.append(tv_score_per_layer_i)
            tv_score_all.append(tv_score_all_i)

        return tv_score_all, tv_score_per_layer

