import os
import sys
import argparse
import scipy.spatial
import json
import torch
import numpy as np
import pickle
import time

from ID_info import InDistInfo
from score_utils import OODScore

project_root_path = os.environ['PROJECT_PATH']
sys.path.append(project_root_path)
from Data.Raw_Dataset.load_data import DatasetInfo


class OutOfDistInfo:
    def __init__(self, inference_filepath):
        f = open(inference_filepath, 'rb')
        self.info = pickle.load(f)
        f.close()

        self.sample_num = len(self.info)
        self.seq_length = 6
        self.layer_num = len(self.info[0]["hidden_state"][0])
        print(f'sample number: {self.sample_num} --- sequence length: {self.seq_length} --- layer number: {self.layer_num}')

    def update_OOD_info(self):
        for i in range(self.sample_num):
            sample = self.info[i]
            hs_all_layer = []

            for j in range(self.layer_num):
                all_pos_hs = np.array([sample["hidden_state"][pos][j][0][0] for pos in range(0, self.seq_length)])
                hs_all_layer.append(np.mean(all_pos_hs, axis=0))
            self.info[i]["hidden_state_mean_pos"] = hs_all_layer

    def get_score(self, idx_list, ID_info, OOD_info, max_order):
        TVscore = []
        for i in idx_list:
            input_data, output_data = OOD_info.load_single_sample(i)
            print(f"******** idx: {i} Info ********")
            print(f"Input: {input_data}\nTrue Output: {output_data}")
            print(f'Pred Output: {self.info[i]["output_seq"]}')

            OODscore = OODScore(self.info[i], ID_info)
            tv_score_all, tv_score_per_layer = OODscore.get_tv_score(max_order=5) ### layer number x max_order
            tmp = []
            print(f"******** idx: {i} Score (k = 0: w/o DiSmo, k > 0: w/ DiSmo) ********")
            for k in range(max_order + 1):
                tmp.append(round(tv_score_all[k], 2))
                print(f'TV_k={k}: {round(tv_score_all[k], 2)}')
            TVscore.append(tmp)

        return TVscore


def get_score_file(args):
    start_time = time.time()
    datasetinfo = DatasetInfo(args.dataset, args.category)
    data_size = datasetinfo.data_size
    end_time = time.time()
    print(f'Loading Dataset: {round(end_time - start_time, 4)}s')

    start_time = time.time()
    ID_info_class = InDistInfo(os.path.join(project_root_path, 'Data/Inference_Data', args.model_name, 'MultiArith_X.pkl'))
    end_time = time.time()
    print(f'Loading ID Inference Data: {round(end_time - start_time, 4)}s')

    start_time = time.time()
    ID_info = ID_info_class.get_IDinfo()
    end_time = time.time()
    print(f'Computer ID Gaussian Values: {round(end_time - start_time, 4)}s')

    start_time = time.time()
    OOD_info_class = OutOfDistInfo(os.path.join(project_root_path, 'Data/Inference_Data', args.model_name, args.dataset + '_' + args.category + '.pkl'))
    OOD_info_class.update_OOD_info()
    end_time = time.time()
    print(f'Loading OOD Inference Data: {round(end_time - start_time, 4)}s')

    start_time = time.time()
    idx_list = [i for i in range(0, data_size)]
    TVscore = OOD_info_class.get_score(idx_list=idx_list, ID_info=ID_info, OOD_info=datasetinfo, max_order=args.max_order)
    end_time = time.time()
    print(f'Computer OOD Scores: {round(end_time - start_time, 4)}s')

    score_info = []
    for i in range(len(idx_list)):
        tmp = {"id": idx_list[i]}
        for j in range(args.max_order + 1):
            tmp["TV"+str(j)] = TVscore[i][j]
        score_info.append(tmp)

    filename = args.dataset + "_" + args.category + ".pkl"
    f = open(os.path.join(project_path, "Data/Score_Data", args.model_name, filename), 'wb')
    pickle.dump(score_info, f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ID_OOD_Score")
    parser.add_argument("--model_name", type=str, default="llama2-7b", choices=["llama2-7b", "gpt2-xl"])
    parser.add_argument("--dataset", type=str, default="MATH",
                        choices=["MultiArith", "GSM8K", "SVAMP", "AddSub", "SingleEq", "SingleOp", "MATH"])
    parser.add_argument("--category", type=str, default="algebra",
                        choices=["algebra", "counting_and_probability", "geometry", "number_theory", "precalculus", "X"])
    parser.add_argument("--max_output_token_num", type=int, default=16)
    parser.add_argument("--max_order", type=int, default=5)
    args = parser.parse_args()

    if args.dataset != "MATH":
        args.category = "X"

    get_score_file(args)

