import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import *
from loader.DBP15k import DBP15kLoader
from script.preprocess.get_token import Token


class MyRawdataset(Dataset):
    def __init__(self, id_features_dict_1, adj_tensor_dict_1, id_features_dict_2, adj_tensor_dict_2,pls,is_neighbor=True):
        super(MyRawdataset, self).__init__()
        self.num = len(pls)  # number of samples
        self.nei_num = adj_tensor_dict_1[0].shape[0]
        dim = (768 + self.nei_num) * 2
        self.train_data = torch.zeros((len(pls), self.nei_num, dim))
        count = 0
        for i, j in pls:
            a = torch.cat((torch.Tensor(id_features_dict_1[i]), adj_tensor_dict_1[i]), dim=1)
            b = torch.cat((torch.Tensor(id_features_dict_2[j]), adj_tensor_dict_2[j]), dim=1)
            c = torch.cat((a, b), dim=-1)
            self.train_data[count] = c
            count=count+1

    # indexing
    def __getitem__(self, index):
        return self.train_data[index]
    def __len__(self):
        return self.num




