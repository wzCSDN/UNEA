from settings import *
import pandas as pd
import torch
import pickle


class DBP15KRawNeighbors():
    def __init__(self, language, doc_id):
        self.language = language
        self.doc_id = doc_id

        self.path = join(DATA_DIR, 'DBP15K', self.language)

        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity_emb = pickle.load(f)
        self.id_neighbors_dict = {}
        self.id_neighbor_features_dict = {}
        self.id_adj_tensor_dict = {}
        self.load_neighbors_with_relations()
        self.prepare_features_and_adj()

    def load_neighbors_with_relations(self):
        triples_path = join(self.path, 'triples_' + self.doc_id)
        data = pd.read_csv(triples_path, header=None, sep='\t')
        data.columns = ['head', 'relation', 'tail']

        for _, row in data.iterrows():
            head_id = int(row['head'])
            rel_id = int(row['relation'])
            tail_id = int(row['tail'])
            if head_id not in self.id_neighbors_dict:
                self.id_neighbors_dict[head_id] = []
            self.id_neighbors_dict[head_id].append((tail_id, rel_id))
            if tail_id not in self.id_neighbors_dict:
                self.id_neighbors_dict[tail_id] = []

            self.id_neighbors_dict[tail_id].append((head_id, rel_id))  # 暂时用相同rel_id

    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        if valid_len > 0:
            # 自身连接
            for i in range(valid_len):
                adj[i, i] = True
            # 中心节点与所有邻居相连
            adj[0, 1:valid_len] = True
            adj[1:valid_len, 0] = True
        return adj

    def prepare_features_and_adj(self):
        all_entity_ids = set(self.id_neighbors_dict.keys()) | set(self.id_entity_emb.keys())

        for entity_id in all_entity_ids:
            center_feature = self.id_entity_emb.get(entity_id, [[0] * LaBSE_DIM])
            neighbors = self.id_neighbors_dict.get(entity_id, [])
            if len(neighbors) > NEIGHBOR_SIZE - 1:
                neighbors = neighbors[:NEIGHBOR_SIZE - 1]

            valid_len = 1 + len(neighbors)  # 1 for center entity
            feature_list = [torch.tensor(center_feature[0], dtype=torch.float)]
            relation_list = [torch.tensor([0], dtype=torch.long)]  # 关系0代表自环

            for neighbor_id, rel_id in neighbors:
                neighbor_feature = self.id_entity_emb.get(neighbor_id, [[0] * LaBSE_DIM])
                feature_list.append(torch.tensor(neighbor_feature[0], dtype=torch.float))
                relation_list.append(torch.tensor([rel_id], dtype=torch.long))

            padding_count = NEIGHBOR_SIZE - valid_len
            if padding_count > 0:
                pad_feature = torch.zeros(LaBSE_DIM, dtype=torch.float)
                pad_relation = torch.tensor([0], dtype=torch.long)
                for _ in range(padding_count):
                    feature_list.append(pad_feature)
                    relation_list.append(pad_relation)

            features_tensor = torch.stack(feature_list)
            relations_tensor = torch.cat(relation_list).unsqueeze(1).float()  # unsqueeze to NEIGHBOR_SIZE x 1

            combined_tensor = torch.cat([features_tensor, relations_tensor], dim=1)

            self.id_neighbor_features_dict[entity_id] = combined_tensor
            self.id_adj_tensor_dict[entity_id] = self.get_adj(valid_len)

