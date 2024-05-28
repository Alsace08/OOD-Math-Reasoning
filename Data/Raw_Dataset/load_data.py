import os
import json
import re

project_path = os.environ['PROJECT_PATH']


class DatasetInfo:
    def __init__(self, dataset_name, category):
        self.dataset_name = dataset_name
        self.category = category
        self.data_path = os.path.join(project_path, "Data/Raw_Dataset", self.dataset_name + ".json")

        if self.dataset_name == "MATH":
            self.filepath = os.path.join(project_path, "Data/Raw_Dataset/MATH", self.category + ".jsonl")
        else:
            self.filepath = os.path.join(project_path, "Data/Raw_Dataset/", self.dataset_name + "_" + self.category + ".jsonl")

        with open(self.filepath, "r", encoding="latin1") as f:
            self.data = [json.loads(line) for line in f]
        self.data_size = len(self.data)

    def load_single_sample(self, idx):
        input_data = self.data[idx]["question"]
        output_data = self.data[idx]["answer"]
        
        return input_data, output_data
