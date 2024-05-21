from settings import *
import csv
import pandas as pd
import torch
import pickle
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from settings import *
class DBP15KRawNeighbors(Dataset):
    def __init__(self, language, doc_id,seed_map):
        super(DBP15KRawNeighbors, self).__init__()
        All_Neibor=True
        self.language = language
        self.doc_id = doc_id
        # os.makedirs(join(PROJ_DIR, 'saved_embed', self.language, self.doc_id), exist_ok=True)
        self.seed_map = seed_map
        self.nei_hop = 2
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        # self.id_neighbor_loader = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.entity_neighbors_dict = {}
        self.all_neighbors_dict={}
        self.load()

        self.num=len(self.id_entity)
        #
        # self.entity_neighbors_dict = {}
        # self.get_neighbors()
        # self.sample_nei_dict={}
        # hop_num=2
        # self.sample_mutihop_neighbors(hop_num=2)
        # self.sample_nei_emded={}
        # self.get_nei_emded()
        # 保存文件
        # pickle.dump(self.sample_nei_emded, join(PROJ_DIR, 'saved_embed', self.language, self.doc_id, "embed.pkl"))
        # with open(join(PROJ_DIR, 'saved_embed', self.language, self.doc_id, "embed.pkl"), "wb") as tf:
        #     pickle.dump(self.sample_nei_emded, tf)
        # # 读取文件
        # #
        if All_Neibor==True:
            self.get_all_neighbors()
            self.sample_nei_emded=self.all_neighbors_dict
        else:
            with open(join(PROJ_DIR, 'saved_embed', self.language, self.doc_id, "embed.pkl"), "rb") as tf:
                self.sample_nei_emded= pickle.load(tf)



        self.get_center_adj()
        self.concat_adj()

    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)
    def get_all_neighbors(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]

            if not int(row['head']) in self.entity_neighbors_dict.keys():
                self.entity_neighbors_dict[int(row['head'])] = []
                self.all_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.entity_neighbors_dict[int(row['head'])]:
                self.entity_neighbors_dict[int(row['head'])].append(tail_str)
                self.all_neighbors_dict[int(row['head'])].append(tail_str)
            if not int(row['tail']) in self.entity_neighbors_dict.keys():
                self.entity_neighbors_dict[int(row['tail'])] = []
                self.all_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.entity_neighbors_dict[int(row['tail'])]:
                self.entity_neighbors_dict[int(row['tail'])].append(head_str)
                self.all_neighbors_dict[int(row['tail'])].append(head_str)

    def get_neighbors(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = int(row['head'])
            tail_str = int(row['tail'])

            if not int(row['head']) in self.entity_neighbors_dict.keys():
                self.entity_neighbors_dict[int(row['head'])] = []
                self.all_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.entity_neighbors_dict[int(row['head'])]:
                self.entity_neighbors_dict[int(row['head'])].append(tail_str)
                self.all_neighbors_dict[int(row['head'])].append(tail_str)
            if not int(row['tail']) in self.entity_neighbors_dict.keys():
                self.entity_neighbors_dict[int(row['tail'])] = []
                self.all_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.entity_neighbors_dict[int(row['tail'])]:
                self.entity_neighbors_dict[int(row['tail'])].append(head_str)
                self.all_neighbors_dict[int(row['tail'])].append(head_str)

            # if not self.id_neighbor_loader.__contains__(head_str):
            #     self.id_neighbor_loader[head_str] = {}
            #     self.id_neighbors_dict[head_str] = [head_str]
            # if not self.id_neighbor_loader[head_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[head_str][row['relation']] = []
            # if not tail_str in self.id_neighbor_loader[head_str][row['relation']]:
            #     self.id_neighbor_loader[head_str][row['relation']].append(tail_str)
            #     self.id_neighbors_dict[head_str].append(tail_str)

            # tail-rel-head, head is a neighbor of tail
            # if not self.id_neighbor_loader.__contains__(tail_str):
            #     self.id_neighbor_loader[tail_str] = {}
            #     self.id_neighbors_dict[tail_str] = [tail_str]
            # if not self.id_neighbor_loader[tail_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[tail_str][row['relation']] = []
            # if not head_str in self.id_neighbor_loader[tail_str][row['relation']]:
            #     self.id_neighbor_loader[tail_str][row['relation']].append(head_str)
            #     self.id_neighbors_dict[head_str].append(head_str)
    def sample_neighbors(self,cur_node,p,seed_map,nei_dict):

        nei_list = nei_dict[cur_node]
        cur_seed_list = []
        cur_notseed_list = []
        sample_pool=[]
        for cur_nei in nei_list:

            if (seed_map[int(cur_nei)] == 0):
                temp_nei = np.repeat(cur_nei,(10-int(p * 10))).tolist()
                cur_notseed_list.append(temp_nei)
            else:
                temp_nei=np.repeat(cur_nei,int(p * 10)).tolist()
                # cur_nei = cur_nei * p * 10
                cur_seed_list.append(temp_nei)
        sample_pool = cur_notseed_list + cur_seed_list
        sample_pool = sum(sample_pool, [])
        # print('sum result:', sample_pool)

        sample_node = random.sample(sample_pool, 1)
        return sample_node[0]
    def sample_mutihop_neighbors(self,hop_num):
        max_Iter=10
        p=0.80
        nei_dict = self.entity_neighbors_dict

        seed_map = self.seed_map
        entity_num = sum(seed_map)

        for cur_node in nei_dict.keys():
            sample_nei_list = []
            for i in range(max_Iter):

                for cur_hop in range(hop_num):
                    if cur_hop==0:
                        temp_nei=cur_node

                    temp_nei=self.sample_neighbors(temp_nei,p,seed_map,nei_dict)
                    sample_nei_list.append(temp_nei)
            self.sample_nei_dict[cur_node]=set(sample_nei_list)

    def get_nei_emded(self):
        for key in self.sample_nei_dict:
            nei_list=self.sample_nei_dict[key]
            temp_embd=[]
            temp_embd.append(self.id_entity[key])
            for id in nei_list:
                temp_embd.append(self.id_entity[id])
            temp_embd=sum(temp_embd, [])
            self.sample_nei_emded[key]=temp_embd









    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}

        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]

            if not int(row['head']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.id_neighbors_dict[int(row['head'])]:
                self.id_neighbors_dict[int(row['head'])].append(tail_str)

            if not int(row['tail']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.id_neighbors_dict[int(row['tail'])]:
                self.id_neighbors_dict[int(row['tail'])].append(head_str)

            # if not self.id_neighbor_loader.__contains__(head_str):
            #     self.id_neighbor_loader[head_str] = {}
            #     self.id_neighbors_dict[head_str] = [head_str]
            # if not self.id_neighbor_loader[head_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[head_str][row['relation']] = []
            # if not tail_str in self.id_neighbor_loader[head_str][row['relation']]:
            #     self.id_neighbor_loader[head_str][row['relation']].append(tail_str)
            #     self.id_neighbors_dict[head_str].append(tail_str)

            # tail-rel-head, head is a neighbor of tail
            # if not self.id_neighbor_loader.__contains__(tail_str):
            #     self.id_neighbor_loader[tail_str] = {}
            #     self.id_neighbors_dict[tail_str] = [tail_str]
            # if not self.id_neighbor_loader[tail_str].__contains__(row['relation']):
            #     self.id_neighbor_loader[tail_str][row['relation']] = []
            # if not head_str in self.id_neighbor_loader[tail_str][row['relation']]:
            #     self.id_neighbor_loader[tail_str][row['relation']].append(head_str)
            #     self.id_neighbors_dict[head_str].append(head_str)

    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.sample_nei_emded.items():
            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.sample_nei_emded[k] = v + [[0] * LaBSE_DIM] * (NEIGHBOR_SIZE - len(v))
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.sample_nei_emded[k] = v[:NEIGHBOR_SIZE]
    def concat_adj(self,is_neighbor=True):

        id_features_dict = self.sample_nei_emded  # number of samples
        self.x_train = []
        self.y_train=[]
        self.x_train_adj = None
        for k in id_features_dict:
            if is_neighbor:
                if self.x_train_adj==None:
                    self.x_train_adj = self.id_adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.x_train_adj = torch.cat((self.x_train_adj, self.id_adj_tensor_dict[k].unsqueeze(0)), dim=0)
            self.x_train.append(id_features_dict[k])
            self.y_train.append([k])
        # transfer to tensor
        # if type(self.x_train[0]) is list:
        self.x_train = torch.Tensor(self.x_train)
        if is_neighbor:
            self.x_train = torch.cat((self.x_train, self.x_train_adj), dim=2)
        self.y_train = torch.Tensor(self.y_train).long()

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    def __len__(self):
        return self.num