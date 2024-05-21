# coding: UTF-8
import os
from posixpath import join
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from settings import *
import torch.utils.data as Data
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
from script.preprocess.deal_my_raw_dataset  import MyRawdataset
from script.preprocess.deal_raw_dataset  import Rawdataset
import random
import faiss
import pandas as pd
import argparse
import logging
from .PLS import *
from datetime import datetime
# using labse
# from transformers import *
import torch
import numpy
# Labse embedding dim
MAX_LEN = 88


def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    # parser.add_argument('--language', type=str, default='ja_en')
    # parser.add_argument('--model_language', type=str, default='ja_en')
    parser.add_argument('--language', type=str, default='ja_en')
    parser.add_argument('--model_language', type=str, default='ja_en')
    parser.add_argument('--model', type=str, default='LaBSE')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--queue_length', type=int, default=32)

    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch + 1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()  # 64，2049
        label = torch.zeros([batch_size]).to(self.device).long()  # 64 0000000000
        loss = self.criterion(x, label)
        return loss


class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(self.args.device)

        self.attn = BatchMultiHeadGraphAttention(self.device, self.args)

        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )



        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2):  #64 768
        bsz = pos_1.shape[0]  #以及归一化后了
        logits = (torch.mm(pos_1, pos_2.t())) / self.args.t
        label = torch.arange(bsz).to(self.device).long()

        loss = self.criterion(logits, label)
        return loss

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch_in = batch[:, :, :LaBSE_DIM]
        adj = batch[:, :, LaBSE_DIM:]

        center = batch_in[:, 0].to(self.device)
        center_neigh = batch_in.to(self.device)

        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)

        center_neigh = center_neigh[:, 0]

        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args.combine:
            out_hat = torch.cat((center, center_neigh), dim=1)
            out_hat = self.attn_mlp(out_hat)
            if self.args.emb_norm:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        # logging.info("attn: ", attn)
        # logging.info("attn.shape: ", attn.shape)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        self.loader1 = DBP15KRawNeighbors(self.args.language, "1")
        self.loader2 = DBP15KRawNeighbors(self.args.language, "2")

        myset1 = Rawdataset(self.loader1.id_neighbors_dict, self.loader1.id_adj_tensor_dict)
        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )
        myset2 =Rawdataset(self.loader2.id_neighbors_dict, self.loader2.id_adj_tensor_dict)
        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset2
        del myset1

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode, valid=False,test=False):
            link = {}
            if valid:
                f="valid.ref"
            elif test:
                f="test.ref"
            else:
                f = 'ref_ent_ids'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), f), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
            return link

        self.link = link_loader(self.args.language)

        self.SI=load_SI(self.args.language)
        link_left = list(self.link.keys())
        link_right = list(self.link.values())
        self.new_pair=UBEA(link_left,link_right,self.SI)

        train_data = MyRawdataset(self.loader1.id_neighbors_dict, self.loader1.id_adj_tensor_dict,self.loader2.id_neighbors_dict,self.loader2.id_adj_tensor_dict,self.new_pair)

        self.loader = Data.DataLoader(
            dataset=train_data,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=False,
            drop_last=True,
        )

        self.val_link = link_loader(self.args.language, True)
        self.test_link = link_loader(self.args.language, test=True)
        # queue for negative samples for 2 language sets
        self.neg_queue1 = None
        # self.neg_valid_num_queue1 = None
        self.neg_queue2 = None
        # self.neg_valid_num_queue2 = None

        self.id_list1 = []

        if training:
            # self.writer = SummaryWriter(
            #     log_dir=join(PROJ_DIR, 'log', self.args.model, self.args.model_language, self.args.time),
            #     comment=self.args.time)

            self.model = MyEmbedder(self.args, VOCAB_SIZE).to(self.device)

            emb_dim = LaBSE_DIM
            self.iteration = 0
            self.lr = self.args.lr
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save_model(self, model, epoch):
        os.makedirs(join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language), exist_ok=True)
        torch.save(model, join(PROJ_DIR, 'checkpoints', self.args.model, self.args.model_language,
                               "model"
                               + "_neighbor_" + "True"

                               + "_epoch_" + str(epoch)
                               + "_batch_size_" + str(BATCH_SIZE)
                               + "_neg_queue_len_" + str(self.args.queue_length - 1)
                               + '.ckpt'))

    def evaluate(self, step):
        logging.info("Evaluate at epoch {}...".format(step))
        print("Evaluate at epoch {}...".format(step))

        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        inverse_ids_2 = dict()
        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (token_data_1, id_data_1) in tqdm(enumerate(self.eval_loader1)):
                entity_vector_1 = self.model(token_data_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in tqdm(enumerate(self.eval_loader2)):
                entity_vector_2 = self.model(token_data_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2.squeeze().tolist())
                vector_2.append(entity_vector_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx
        #update pair
        a = numpy.array(list(np.concatenate(vector_1)))
        aa = numpy.array(list(np.concatenate(vector_2)))
        emb_all = [0] * (a.shape[0] + aa.shape[0])

        for i in range(len(ids_1)):
            emb_all[ids_1[i]] = a[i]
        for j in range(len(ids_2)):
            emb_all[ids_2[j]] = aa[j]
        emb_all = numpy.array(emb_all)
        link_left = list(self.link.keys())
        link_right = list(self.link.values())
        new_pair = UBEA(link_left, link_right, emb_all)

        def cal_hit(v1, v2, link):
            source = [_id for _id in ids_1 if _id in link]
            target = np.array(
                [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 99999 for _id in source])

            src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
            v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
            v2 = np.concatenate(tuple(v2), axis=0)
            index = faiss.IndexFlatL2(v2.shape[1])
            index.add(np.ascontiguousarray(v2))
            D, I = index.search(np.ascontiguousarray(v1), 10)
            hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
            hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
            logging.info("#Entity: {}".format(len(source)))
            logging.info("Hit@1: {}".format(round(hit1, 3)))
            logging.info("Hit@10:{}".format(round(hit10, 3)))
            print("#Entity: {}".format(len(source)))
            print("Hit@1: {}".format(round(hit1, 3)))
            print("Hit@10:{}".format(round(hit10, 3)))
            return round(hit1, 3), round(hit10, 3)

        logging.info('========Validation========')
        print('========Validation========')
        hit1_valid, hit10_valid = cal_hit(vector_1, vector_2, self.val_link)
        logging.info('===========Test===========')
        print('===========Test===========')
        hit1_test, hit10_test = cal_hit(vector_1, vector_2, self.test_link)
        return new_pair,hit1_valid, hit10_valid, hit1_test, hit10_test

    def train(self, start=0):
        fix_seed(self.seed)
        if not os.path.exists(join(PROJ_DIR, 'log')):
            os.mkdir(join(PROJ_DIR, 'log'))
            if not os.path.exists(join(PROJ_DIR, 'log', 'layers_LaBSE_neighbor')):
                os.mkdir(join(PROJ_DIR, 'log', 'layers_LaBSE_neighbor'))

        logging.basicConfig(
            filename=join(PROJ_DIR, 'log', 'layers_LaBSE_neighbor/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
                self.args.batch_size,
                self.args.queue_length,
                self.args.center_norm,
                self.args.neighbor_norm,
                self.args.emb_norm,
                self.args.combine,
                self.args.gat_num,
                self.args.t,
                self.args.momentum,
                self.args.lr
            )), level=logging.INFO)
        tot_loss = 0

        self.all_data_batches = []
        for batch_id, (token_data) in enumerate(self.loader):
            self.all_data_batches.append([token_data])

        # 1,64*20*788  64*1


        neg_queue = []
        # neg_valid_num = []
        logging.info("*** Evaluate at the very beginning ***")
        print("*** Evaluate at the very beginning ***")
        # self.evaluate(0)

        best_hit1_valid_epoch = 0
        best_hit10_valid_epoch = 0
        best_hit1_test_epoch = 0
        best_hit10_test_epoch = 0
        best_hit1_valid = 0
        best_hit10_valid = 0
        best_hit1_valid_hit10 = 0
        best_hit10_valid_hit1 = 0
        best_hit1_test = 0
        best_hit10_test = 0
        best_hit1_test_hit10 = 0
        best_hit10_test_hit1 = 0
        record_hit1 = 0
        record_hit10 = 0
        record_epoch = 0
        record_batch_id = 0
        for epoch in range(start, self.args.epoch):
            adjust_learning_rate(self.optimizer, epoch, self.lr)
            for batch_id, (token_data) in tqdm(enumerate(self.all_data_batches)):
                left = token_data[0][:, :, 788:]
                right = token_data[0][:, :, :788]
                pos_batch = None
                # 准备负样本 队列的方式


                self.optimizer.zero_grad()

                emb_left = self.model(left)  # 64 20 788-- 64*788
                emb_right = self.model(right)
                neg_value_list = []


                # contrastive
                contrastive_loss = self.model.contrastive_loss(emb_left, emb_right)
                if epoch>=20:
                    link_left = list(self.new_pair[:, 0])
                    link_right = list(self.new_pair[:, 1])
                    intial_emb_left = torch.Tensor(self.SI[link_left])
                    intial_emb_right = torch.Tensor(self.SI[link_right])
                    intial_emb = torch.cat((intial_emb_left, intial_emb_right), dim=0)
                    emb = torch.cat((emb_left, emb_right), dim=0)
                    contrastive_loss = 0.6*contrastive_loss+self.model.contrastive_loss(emb, intial_emb.cuda())
                SI=self.SI[:,0]
                new_pair=self.new_pair #又回到之前的样子了    心态崩了

                self.iteration += 1

                contrastive_loss.backward(retain_graph=True)
                self.optimizer.step()
                if batch_id == len(self.all_data_batches) - 1:
                    # if batch_id % 200 == 0 or batch_id == len(all_data_batches) - 1:
                    logging.info('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                       contrastive_loss.detach().cpu().data / self.args.batch_size))
                    print('epoch: {} batch: {} loss: {}'.format(epoch, batch_id,
                                                                contrastive_loss.detach().cpu().data / self.args.batch_size))

            if epoch % 10 == 0:
                self.new_pair,hit1_valid, hit10_valid, hit1_test, hit10_test = self.evaluate(
                    str(epoch) + ": batch " + str(batch_id))
                if hit1_valid > best_hit1_valid:
                    best_hit1_valid = hit1_valid
                    best_hit1_valid_hit10 = hit10_valid
                    best_hit1_valid_epoch = epoch
                    record_epoch = epoch
                    record_batch_id = batch_id
                    record_hit1 = hit1_test
                    record_hit10 = hit10_test
                if hit10_valid > best_hit10_valid:
                    best_hit10_valid = hit10_valid
                    best_hit10_valid_hit1 = hit1_valid
                    best_hit10_valid_epoch = epoch
                    if hit1_valid == best_hit1_valid:
                        record_epoch = epoch
                        record_batch_id = batch_id
                        record_hit1 = hit1_test
                        record_hit10 = hit10_test

                if hit1_test > best_hit1_test:
                    best_hit1_test = hit1_test
                    best_hit1_test_hit10 = hit10_test
                    best_hit1_test_epoch = epoch
                if hit10_test > best_hit10_test:
                    best_hit10_test = hit10_test
                    best_hit10_test_hit1 = hit1_test
                    best_hit10_test_epoch = epoch
                logging.info(
                    'Test Hit@1(10)    = {}({}) at epoch {} batch {}'.format(hit1_test, hit10_test, epoch,
                                                                             batch_id))
                logging.info(
                    'Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10,
                                                                    best_hit1_valid_epoch))
                logging.info(
                    'Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid, best_hit10_valid_hit1,
                                                                    best_hit10_valid_epoch))
                logging.info('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10,
                                                                                      record_epoch,
                                                                                      record_batch_id))

                logging.info(
                    'Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10,
                                                                    best_hit1_test_epoch))
                logging.info(
                    'Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test, best_hit10_test_hit1,
                                                                    best_hit10_test_epoch))
                logging.info("====================================")

                print('Test Hit@1(10)    = {}({}) at epoch {} batch {}'.format(hit1_test, hit10_test, epoch,
                                                                               batch_id))
                print('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10,
                                                                      best_hit1_valid_epoch))
                print('Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid, best_hit10_valid_hit1,
                                                                      best_hit10_valid_epoch))
                print('Test @ Best Valid = {}({}) at epoch {} batch {}'.format(record_hit1, record_hit10,
                                                                               record_epoch, record_batch_id))

                print('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10,
                                                                      best_hit1_test_epoch))
                print('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test, best_hit10_test_hit1,
                                                                      best_hit10_test_epoch))
                print("====================================")

                train_data = MyRawdataset(self.loader1.id_neighbors_dict, self.loader1.id_adj_tensor_dict,
                                          self.loader2.id_neighbors_dict, self.loader2.id_adj_tensor_dict, self.new_pair)

                self.loader = Data.DataLoader(
                    dataset=train_data,  # torch TensorDataset format
                    batch_size=self.args.batch_size,  # all test data
                    shuffle=False,
                    drop_last=True,
                )
                self.all_data_batches = []
                for batch_id, (token_data) in enumerate(self.loader):
                    self.all_data_batches.append([token_data])




