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
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
import pandas as pd
import argparse
import logging
from .PLS import *
from datetime import datetime
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist


LaBSE_DIM = 768
REL_DIM = 100
TREE_DEPTH = 2
TREE_WIDTH = 5


def get_orthogonal_matrix(vectors_for_reflection):
    d = vectors_for_reflection.shape[1]
    vectors_normalized = F.normalize(vectors_for_reflection, p=2, dim=1)
    vvt = torch.bmm(vectors_normalized.unsqueeze(2), vectors_normalized.unsqueeze(1))
    return torch.eye(d, device=vectors_for_reflection.device).unsqueeze(0) - 2 * vvt


def sample_personalized_tree(root_id, neighbors_dict, degree_penalties, embs_for_sampling):
    tree = {}
    root_neighbors = neighbors_dict.get(root_id, [])
    if not root_neighbors: return {}
    root_vec = embs_for_sampling[root_id]

    child_l1_ids = np.array([n[0] for n in root_neighbors])
    rel_l1_ids = np.array([n[1] for n in root_neighbors])

    if len(child_l1_ids) > TREE_WIDTH:
        child_vecs = embs_for_sampling[child_l1_ids]
        attn_scores = (root_vec @ child_vecs.T) * degree_penalties[child_l1_ids]
        probs = np.exp(attn_scores - np.max(attn_scores))
        probs_sum = np.sum(probs)
        if probs_sum > 1e-8:
            probs /= probs_sum
        else:
            probs = np.ones(len(child_l1_ids)) / len(child_l1_ids)
        chosen_indices = np.random.choice(len(child_l1_ids), TREE_WIDTH, replace=False, p=probs)
        child_l1_ids = child_l1_ids[chosen_indices]
        rel_l1_ids = rel_l1_ids[chosen_indices]

    if len(child_l1_ids) > 0:
        parent_l1_ids = np.array([root_id] * len(child_l1_ids))
        tree[1] = (parent_l1_ids, child_l1_ids, rel_l1_ids)

    if TREE_DEPTH >= 2 and 1 in tree:
        all_parents_l2, all_children_l2, all_rels_l2 = [], [], []
        for parent_l2_id in child_l1_ids:
            l2_neighbors = neighbors_dict.get(parent_l2_id, [])
            if not l2_neighbors: continue

            child_l2_ids = np.array([n[0] for n in l2_neighbors])
            rel_l2_ids = np.array([n[1] for n in l2_neighbors])

            if len(child_l2_ids) > TREE_WIDTH:
                child_l2_vecs = embs_for_sampling[child_l2_ids]
                parent_l2_vec = embs_for_sampling[parent_l2_id]
                root_sim = root_vec @ child_l2_vecs.T
                parent_sim = parent_l2_vec @ child_l2_vecs.T
                attn_scores = (0.5 * root_sim + 0.5 * parent_sim) * degree_penalties[child_l2_ids]
                probs = np.exp(attn_scores - np.max(attn_scores))
                probs_sum = np.sum(probs)
                if probs_sum > 1e-8:
                    probs /= probs_sum
                else:
                    probs = np.ones(len(child_l2_ids)) / len(child_l2_ids)
                chosen_indices = np.random.choice(len(child_l2_ids), TREE_WIDTH, replace=False, p=probs)
                child_l2_ids = child_l2_ids[chosen_indices]
                rel_l2_ids = rel_l2_ids[chosen_indices]

            if len(child_l2_ids) > 0:
                all_parents_l2.extend([parent_l2_id] * len(child_l2_ids))
                all_children_l2.extend(child_l2_ids)
                all_rels_l2.extend(rel_l2_ids)

        if all_children_l2:
            tree[2] = (np.array(all_parents_l2), np.array(all_children_l2), np.array(all_rels_l2))
    return tree


def parse_options(parser):
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='fr_en')
    parser.add_argument('--model_language', type=str, default='ja_en')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()


class MyEmbedder(nn.Module):
    def __init__(self, args, num_relations, initial_entity_embs, initial_rel_embs):
        super(MyEmbedder, self).__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.entity_embs = nn.Embedding.from_pretrained(initial_entity_embs.float(), freeze=False)
        self.ent_dim = self.entity_embs.embedding_dim
        self.relation_embs = nn.Embedding(num_relations, REL_DIM)
        self.relation_embs.weight.data.copy_(initial_rel_embs.float())
        self.rel_dim = self.relation_embs.embedding_dim
        self.rel_proj = nn.Linear(self.rel_dim, self.ent_dim, bias=False)
        self.w_att = nn.Linear(self.ent_dim * 3, 1, bias=False)
        self.align_criterion = nn.CrossEntropyLoss()
        self.mi_criterion = nn.CrossEntropyLoss()

    def _compose_relations(self, r1_embs, r2_embs):
        return r1_embs * r2_embs

    def _tree_attention_encoding(self, root_ids, tree_data):
        all_unique_ids, level_data = tree_data
        if not all_unique_ids.numel(): return self.entity_embs(root_ids)
        id_to_idx = {uid.item(): i for i, uid in enumerate(all_unique_ids)}
        node_embs = self.entity_embs(all_unique_ids)
        comp_rel_embs_cache = {}

        for l in range(1, TREE_DEPTH + 1):
            if l not in level_data: continue
            parents, children, rels_pc, _ = level_data[l]
            if l == 1:
                for i in range(len(parents)):
                    comp_rel_embs_cache[(parents[i].item(), children[i].item())] = self.relation_embs(rels_pc[i])

        for l in range(TREE_DEPTH, 0, -1):
            if l not in level_data: continue
            parents, children, rels_pc, batch_indices = level_data[l]
            parent_indices = torch.tensor([id_to_idx[p.item()] for p in parents], device=self.device, dtype=torch.long)
            child_indices = torch.tensor([id_to_idx[c.item()] for c in children], device=self.device, dtype=torch.long)
            batch_root_ids = root_ids[batch_indices]
            root_indices = torch.tensor([id_to_idx[r.item()] for r in batch_root_ids], device=self.device,
                                        dtype=torch.long)
            rel_pc_embs = self.relation_embs(rels_pc)

            rel_comp_embs_list = []
            if l == 1:
                rel_comp_embs_list = rel_pc_embs
            else:
                for i in range(len(parents)):
                    root_id, parent_id = batch_root_ids[i].item(), parents[i].item()
                    rel_rp_emb = comp_rel_embs_cache.get((root_id, parent_id),
                                                         torch.zeros(self.rel_dim, device=self.device))
                    rel_pc_emb = self.relation_embs(rels_pc[i])
                    comp_emb = self._compose_relations(rel_rp_emb, rel_pc_emb)
                    rel_comp_embs_list.append(comp_emb)
                if not rel_comp_embs_list: continue  # 如果l=2但没有路径，跳过
                rel_comp_embs_list = torch.stack(rel_comp_embs_list)

            rel_pc_embs_proj = self.rel_proj(rel_pc_embs)
            rel_comp_embs_proj = self.rel_proj(rel_comp_embs_list)
            W_k, W_p = get_orthogonal_matrix(rel_pc_embs_proj), get_orthogonal_matrix(rel_comp_embs_proj)
            e_i, e_x, e_y = node_embs[root_indices], node_embs[parent_indices], node_embs[child_indices]
            e_i_proj = torch.bmm(W_p.transpose(1, 2), e_i.unsqueeze(2)).squeeze(2)
            e_x_proj = torch.bmm(W_k.transpose(1, 2), e_x.unsqueeze(2)).squeeze(2)
            b = self.leaky_relu(self.w_att(torch.cat([e_i_proj, e_x_proj, e_y], dim=1))).squeeze(-1)

            unique_parent_indices, parent_inverse_indices = torch.unique(parent_indices, return_inverse=True)
            num_unique_parents = unique_parent_indices.size(0)
            b_max_by_parent = torch.full((num_unique_parents,), -float('inf'), device=self.device).scatter_reduce_(0,
                                                                                                                   parent_inverse_indices,
                                                                                                                   b,
                                                                                                                   reduce="amax",
                                                                                                                   include_self=True)
            b_shifted = b - b_max_by_parent[parent_inverse_indices]
            b_exp = torch.exp(b_shifted)
            b_exp_sum_by_parent = torch.zeros(num_unique_parents, device=self.device).scatter_reduce_(0,
                                                                                                      parent_inverse_indices,
                                                                                                      b_exp,
                                                                                                      reduce="sum",
                                                                                                      include_self=True)
            b_exp_sum_by_parent.clamp_(min=1e-10)
            attn_weights_a = b_exp / b_exp_sum_by_parent[parent_inverse_indices]

            update_info = torch.bmm(W_k.transpose(1, 2), e_y.unsqueeze(2)).squeeze(2) * attn_weights_a.unsqueeze(1)
            aggregated_info = torch.zeros_like(node_embs).scatter_add_(0, parent_indices.unsqueeze(1).expand_as(
                update_info), update_info)
            node_embs = self.leaky_relu(node_embs + aggregated_info)

        root_indices_in_embs = torch.tensor([id_to_idx[rid.item()] for rid in root_ids], device=self.device,
                                            dtype=torch.long)
        return node_embs[root_indices_in_embs]

    def forward(self, batch_root_ids, tree_data):
        return self._tree_attention_encoding(batch_root_ids, tree_data)

    def alignment_loss(self, emb1, emb2):
        emb1, emb2 = F.normalize(emb1, p=2, dim=1), F.normalize(emb2, p=2, dim=1)
        logits = torch.mm(emb1, emb2.t()) / self.args.t
        label = torch.arange(emb1.shape[0], device=self.device).long()
        return self.align_criterion(logits, label)

    def mi_loss(self, current_embs, initial_embs):
        current_embs, initial_embs = F.normalize(current_embs, p=2, dim=1), F.normalize(initial_embs, p=2, dim=1)
        logits = torch.mm(current_embs, initial_embs.t()) / self.args.t
        label = torch.arange(current_embs.shape[0], device=self.device).long()
        return self.mi_criterion(logits, label)


class Trainer(object):
    def __init__(self, training=True, seed=37):
        fix_seed(seed)
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)
        self.device = torch.device(self.args.device)
        self.start_time = time.time()

        print("Loading data...")
        self.loader1 = DBP15KRawNeighbors(self.args.language, "1")
        self.loader2 = DBP15KRawNeighbors(self.args.language, "2")
        self.neighbors_dict_complete = {**self.loader1.id_neighbors_dict, **self.loader2.id_neighbors_dict}

        print("Preparing data for attention sampling...")
        self.entity_degrees = {eid: len(neighbors) for eid, neighbors in self.neighbors_dict_complete.items()}
        all_entity_ids = list(self.loader1.id_entity_emb.keys()) + list(self.loader2.id_entity_emb.keys())
        max_id = max(all_entity_ids) if all_entity_ids else 0
        self.degree_penalties = np.ones(max_id + 1)
        for eid, degree in self.entity_degrees.items():
            if degree > 0 and eid <= max_id: self.degree_penalties[eid] = 1.0 / np.log(degree + 1)

        def link_loader(mode, v=False, t=False):
            f = 'ref_ent_ids'
            if v:
                f = "valid.ref"
            elif t:
                f = "test.ref"
            return pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), f), sep='\t', header=None).values

        self.link, self.val_link, self.test_link = link_loader(self.args.language), link_loader(self.args.language,
                                                                                                v=True), link_loader(
            self.args.language, t=True)

        print("Initializing embeddings...")
        self.SI = load_SI(self.args.language)
        self.initial_entity_embs_torch = torch.from_numpy(self.SI).to(self.device).float()

        all_rels = set(r for n in self.neighbors_dict_complete.values() for _, r in n)
        self.num_relations = max(all_rels) + 1 if all_rels else 1
        print(f"Found {self.num_relations} unique relations.")
        self.initial_rel_embs_torch = torch.randn(self.num_relations, REL_DIM).to(self.device).float()

        print("Generating initial pseudo-labels...")
        self.new_pair_np = UBEA(list(self.link[:, 0]), list(self.link[:, 1]), self.SI)

        if training:
            self.model = MyEmbedder(self.args, self.num_relations, self.initial_entity_embs_torch,
                                    self.initial_rel_embs_torch).to(self.device)
            self.momentum_model = MyEmbedder(self.args, self.num_relations, self.initial_entity_embs_torch,
                                             self.initial_rel_embs_torch).to(self.device)
            for param_k in self.momentum_model.parameters(): param_k.requires_grad = False
            self.sync_momentum_model()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            self.args.m, self.args.lambda_val = 10, 0.8

    @torch.no_grad()
    def sync_momentum_model(self):
        for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()): param_k.data.copy_(
            param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.model.parameters(),
                                    self.momentum_model.parameters()): param_k.data = param_k.data * self.args.momentum + param_q.data * (
                    1. - self.args.momentum)

    def collate_trees(self, tree_batch):
        all_unique_ids, level_data = set(), {}
        for i, tree in enumerate(tree_batch):
            if not tree or 1 not in tree: continue
            root_id = tree[1][0][0]
            all_unique_ids.add(root_id)
            for l in range(1, TREE_DEPTH + 1):
                if l in tree:
                    p, c, r = tree[l]
                    all_unique_ids.update(p);
                    all_unique_ids.update(c)
                    if l not in level_data: level_data[l] = ([], [], [], [])
                    level_data[l][0].extend(p);
                    level_data[l][1].extend(c);
                    level_data[l][2].extend(r);
                    level_data[l][3].extend([i] * len(p))

        if not all_unique_ids: return torch.LongTensor([]).to(self.device), {}
        all_ids_tensor = torch.LongTensor(list(all_unique_ids)).to(self.device)
        for l in level_data:
            p, c, r, b = level_data[l]
            level_data[l] = (torch.LongTensor(p).to(self.device), torch.LongTensor(c).to(self.device),
                             torch.LongTensor(r).to(self.device), torch.LongTensor(b).to(self.device))
        return all_ids_tensor, level_data

    def evaluate(self, step, is_test=False):
        print(f"\nEvaluate at step {step}...")
        self.model.eval()
        print("Using trained entity embeddings directly for evaluation (fast mode).")
        with torch.no_grad(): emb_all = self.model.entity_embs.weight.cpu().numpy()
        link_data = self.test_link if is_test else self.val_link
        print(f"--- {'Final Test' if is_test else 'Validation'} Performance ---")
        eval_entity_alignment(list(link_data[:, 0]), list(link_data[:, 1]), emb_all, len(link_data))
        print("Updating pseudo-labels for the next training cycle...")
        return UBEA(list(self.link[:, 0]), list(self.link[:, 1]), emb_all)

    def train(self, start=0):
        print("\nStarting training...")
        bce_loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(start, self.args.epoch):
            if epoch % self.args.m == 0:
                self.new_pair_np = self.evaluate(str(epoch))
                print("\nSampling personalized trees for training...")
                print("Pre-calculating data for attention sampling (this is now fast)...")
                with torch.no_grad():
                    embs_for_sampling = self.momentum_model.entity_embs.weight.cpu().numpy()
                    embs_for_sampling /= (np.linalg.norm(embs_for_sampling, axis=1, keepdims=True) + 1e-8)

                self.sampled_trees = {}
                all_training_entities = np.unique(self.new_pair_np.flatten())
                for entity_id in tqdm(all_training_entities, desc="Sampling Trees"):
                    self.sampled_trees[entity_id] = sample_personalized_tree(entity_id, self.neighbors_dict_complete,
                                                                             self.degree_penalties, embs_for_sampling)
                print("Tree sampling finished.")

            self.model.train()
            total_loss, total_la, total_lmi, num_batches = 0, 0, 0, 0
            num_pairs = len(self.new_pair_np)
            shuffled_indices = np.random.permutation(num_pairs)

            for i in tqdm(range(0, num_pairs, self.args.batch_size), desc=f"Epoch {epoch}"):
                if i + self.args.batch_size > num_pairs: continue
                batch_indices = shuffled_indices[i: i + self.args.batch_size]
                pair_batch = torch.from_numpy(self.new_pair_np[batch_indices]).long().to(self.device)
                left_ids, right_ids = pair_batch[:, 0], pair_batch[:, 1]
                if not all(j.item() in self.sampled_trees for j in left_ids) or not all(
                    j.item() in self.sampled_trees for j in right_ids): continue

                tree_data_left = self.collate_trees([self.sampled_trees[j.item()] for j in left_ids])
                tree_data_right = self.collate_trees([self.sampled_trees[j.item()] for j in right_ids])
                if not tree_data_left[0].numel() or not tree_data_right[0].numel(): continue

                self.optimizer.zero_grad()
                emb_left = self.model(left_ids, tree_data_left)
                emb_right = self.model(right_ids, tree_data_right)
                loss_align = self.model.alignment_loss(emb_left, emb_right)

                batch_ids = torch.cat([left_ids, right_ids])
                loss_mi_E = self.model.mi_loss(torch.cat([emb_left, emb_right]),
                                               self.initial_entity_embs_torch[batch_ids])
                loss_mi_R = self.model.mi_loss(self.model.relation_embs.weight, self.initial_rel_embs_torch)
                loss_mi_T = bce_loss_fn(torch.matmul(emb_left, emb_right.T),
                                        torch.eye(emb_left.shape[0], device=self.device))
                loss_mi = loss_mi_E + loss_mi_R + loss_mi_T

                loss = self.args.lambda_val * loss_align + (1 - self.args.lambda_val) * loss_mi
                loss.backward()
                self.optimizer.step()
                self._momentum_update_key_encoder()

                total_loss += loss.item();
                total_la += loss_align.item();
                total_lmi += loss_mi.item();
                num_batches += 1

            if num_batches > 0:
                avg_loss, avg_la, avg_lmi = total_loss / num_batches, total_la / num_batches, total_lmi / num_batches
                print(
                    f'Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} (LA: {avg_la:.4f}, LMI: {avg_lmi:.4f}), Time: {time.time() - self.start_time:.2f}s')

        self.evaluate("final_test", is_test=True)
