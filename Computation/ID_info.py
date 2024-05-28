import pickle
import numpy as np


class InDistInfo:
    def __init__(self, filepath):
        f = open(filepath, 'rb')
        self.info = pickle.load(f)
        f.close()

        self.sample_num = len(self.info)
        self.seq_length = 6
        self.layer_num = len(self.info[0]["hidden_state"][0])

    def get_hs_all_sample_all_layer(self):
        '''
        :return: hs_all_sample_all_layer: sample_num x layer_num
        '''
        hs_all_sample_all_layer = []
        for i in range(self.sample_num):
            if i % 100 >= 60:
                continue
            sample = self.info[i]
            sample_hs = []
            for j in range(self.layer_num):
                all_pos_hs = np.array([sample["hidden_state"][pos][j][0][0] for pos in range(0, self.seq_length)])
                sample_hs.append(np.mean(all_pos_hs, axis=0))
            hs_all_sample_all_layer.append(sample_hs)
        return hs_all_sample_all_layer

    def get_mean_var_hs_all_layer(self):
        hs_all_sample_all_layer = self.get_hs_all_sample_all_layer()
        mean_hs_all_layer = []
        var_hs_all_layer = []
        for i in range(self.layer_num):
            layer_hs = [hs_all_sample_all_layer[j][i] for j in range(len(hs_all_sample_all_layer))]
            mean_layer_hs = np.mean(np.array(layer_hs), axis=0, keepdims=True)
            var_layer_hs = np.var(np.array(layer_hs), axis=0, keepdims=True)
            mean_hs_all_layer.append(mean_layer_hs)
            var_hs_all_layer.append(var_layer_hs)
        return mean_hs_all_layer, var_hs_all_layer

    def get_input_hs(self):
        input_hs_all_sample = []
        for i in range(self.sample_num):
            if i % 100 >= 60:
                continue
            input_hs_per_sample = []
            for j in range(self.layer_num):
                input_hs_per_sample.append(np.mean(np.array(self.info[i]["hidden_state"][0][j][0]), axis=0))
            input_hs_per_sample = np.mean(input_hs_per_sample, axis=0)
            input_hs_all_sample.append(input_hs_per_sample)
        return np.mean(input_hs_all_sample, axis=0), np.var(input_hs_all_sample, axis=0)

    def get_IDinfo(self):
        mean_hs_all_layer, var_hs_all_layer = self.get_mean_var_hs_all_layer()
        mean_input_hs , var_input_hs = self.get_input_hs()

        data_info = []
        data_info.append({"layer": 0,
                          "mean_input_hs": mean_input_hs,
                          "var_input_hs": var_input_hs,
                          "mean_hs": [[0] * len(mean_hs_all_layer[1][0])],
                          "var_hs": [[0] * len(var_hs_all_layer[1][0])]})
        for i in range(1, self.layer_num):
            data_info.append({"layer": i,
                              "mean_hs": mean_hs_all_layer[i],
                              "var_hs": var_hs_all_layer[i]})
        return data_info
