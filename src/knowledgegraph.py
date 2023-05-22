import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os



# knowledge_graph
class KnowledgeGraph(nn.Module):
    def __init__(self, lang, kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations, device, n_neg_pos=1):
        super().__init__()
        self.lang = lang
        
        self.train_data = kg_train_data
        self.val_data = kg_val_data
        self.test_data = kg_test_data
        
        
        self.num_relations = num_relations
        self.num_entities = num_entities
        
        self.device = device

        self.k_subgraph_list = None
        self.computed_entity_embedding_kg = None
        
        self.h_train, self.r_train, self.t_train = self.train_data[:, 0], self.train_data[:, 1], self.train_data[:, 2]
        self.h_val, self.r_val, self.t_val = self.val_data[:, 0], self.val_data[:, 1], self.val_data[:, 2]
        self.h_test, self.r_test, self.t_test = self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2]

    
    def get_global_h_t(self, seeds, pre_langs, all_entity2kgs, all_entity_global_index):
        self._get_entity_global_index(seeds, pre_langs, all_entity2kgs, all_entity_global_index)
        
        self.h_train_global = self._get_global_index(self.h_train, all_entity_global_index)
        self.t_train_global = self._get_global_index(self.t_train, all_entity_global_index)
        self.h_val_global = self._get_global_index(self.h_val, all_entity_global_index)
        self.t_val_global = self._get_global_index(self.t_val, all_entity_global_index)
        self.h_test_global = self._get_global_index(self.h_test, all_entity_global_index)
        self.t_test_global = self._get_global_index(self.t_test, all_entity_global_index)
        self.entity_global_index = all_entity_global_index[self.lang]
    
    def _get_entity_global_index(self, seeds, pre_langs, all_entity2kgs, all_entity_global_index):
        all_entity_global_index[self.lang] = []
        if len(all_entity2kgs) == 0:
            for i in range(self.num_entities):
                all_entity_global_index[self.lang].append(i)
                all_entity2kgs[i] = set([self.lang]) # 
        else:
            for i in range(self.num_entities):
                index = None
                for pre_l in pre_langs:
                    if (self.lang, pre_l) in seeds:
                        index = torch.nonzero(seeds[(self.lang, pre_l)][:, 0] == i)
                        if index.shape[0] != 0:
                            align_entity = seeds[(self.lang, pre_l)][index[0].item()][1]
                            i_global = all_entity_global_index[pre_l][align_entity]
                            all_entity_global_index[self.lang].append(i_global)
                            all_entity2kgs[i_global].add(self.lang)
                            break
                    elif (pre_l, self.lang) in seeds:
                        index = torch.nonzero(seeds[(pre_l, self.lang)][:, 1] == i)
                        if index.shape[0] != 0:
                            align_entity = seeds[(pre_l, self.lang)][index[0].item()][0]
                            i_global = all_entity_global_index[pre_l][align_entity]
                            all_entity_global_index[self.lang].append(i_global)
                            all_entity2kgs[i_global].add(self.lang)
                            break
                    else:
                        assert False, f"no seeds for {self.lang}-{pre_l} or {pre_l}-{self.lang}"
                if index.shape[0] == 0:
                    i_global = len(all_entity2kgs)
                    all_entity_global_index[self.lang].append(i_global)
                    all_entity2kgs[i_global] = set([self.lang])
        pre_langs.append(self.lang)
        
    
    def _get_global_index(self, entity_ids, all_entity_global_index):
        
        entity_id_global = []
        for e in entity_ids:
            entity_id_global.append(all_entity_global_index[self.lang][e])
        return torch.LongTensor(entity_id_global)
    

    def generate_batch_data(self, h_all, r_all, t_all, batch_size, shuffle=True):
        h_all = torch.unsqueeze(h_all, dim=1) # [N, 1]
        r_all = torch.unsqueeze(r_all, dim=1) # [N, 1]
        t_all = torch.unsqueeze(t_all, dim=1)# [N, 1]
        
        triple_all = torch.cat([h_all, r_all, t_all], dim=-1) # [N, 3]
        triple_dataloader = DataLoader(triple_all, batch_size=batch_size, shuffle=shuffle)
        
        return triple_dataloader

