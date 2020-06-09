import torch
import torch.utils.data as data

__all__ = ['DatasetFromDict']

class DatasetFromDict(data.Dataset):
    def __init__(self, data):
        self._data = data
        self._key = list(data.keys())

    def __len__(self):
        return len(self._data[self._key[0]])

    def __getitem__(self, idx):
        ret_dict = dict()
        for key in self._key:
            ret_dict[key] = self._data[key][idx]
        return ret_dict
