import os
import json
import numpy as np
from torch.utils.data import Dataset
import pickle

class Zod(Dataset):
    def __init__(self, data_root, length):
        self.data_root = data_root
        self.preloaded = {}
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index not in self.preloaded:
            self.preloaded[index] = self.load_item(index)
        return self.preloaded[index]

    def load_item(self, index, prefix='val'):
        file_path = os.path.join(self.data_root, f'{prefix}_{index}.npy')
        with open(file_path, 'rb') as f:
            item = np.load(f)
        return item 

    def load_query_samples(self, sample_list):
        for sample in sample_list:
            self.preloaded[sample] = self.load_item(sample)

    def unload_query_samples(self, sample_list):
        for sample in sample_list:
            del self.preloaded[sample]

    def get_samples(self, id_list):
        data = []
        labels = []
        for id in id_list:
            item = self.get_item(id)
            data.append(item)
            labels.append(None)
        return data, labels

    def get_item(self, index):
        return self.preloaded[index]

    def get_item_count(self):
        return self.length

    def get_item_loc(self, index):
        file_path = os.path.join(self.data_root, f'val_{index}.pkl')
        return file_path

def preprocess(list_data):
    pass

class PostProcessZod:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        # TODO: Init Postprocess parameters
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        processed_results = []
        for idx in range(len(content_id)):
            processed_results.append([])
        processed_results[idx].append(results)
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict

'''class PostProcessZod:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        processed_results = []
        for idx in range(len(content_id)):
            processed_results.append(torch.stack(results).cpu())
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict'''