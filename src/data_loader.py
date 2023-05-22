import numpy as np
import pandas as pd
import torch

from src.knowledgegraph import KnowledgeGraph
from src.utils import get_language_list, get_all_edges, create_subgraph_list

import copy
import os



# data_loader
class MKGDataset:
    def __init__(self, args):
        self.data_dir = args.data_path + args.dataset
        self.entity_dir = self.data_dir + '/entity'
        self.kg_dir = self.data_dir + '/kg'
        self.align_dir = self.data_dir + '/seed_alignlinks'
        self.args = args
        
        self.kg_names = args.langs # ['el', 'en', ...]
        self.num_kgs = len(self.kg_names)
    
    def load_data(self):

        kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index = self.create_KG_objects_and_subgraph()
        

        return kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index
    
    def create_KG_objects_and_subgraph(self):

        kg_objects_dict = {}
        seeds = self.load_align_links() # {(lang1, lang2): torch.LongTensor}
        pre_langs = []
        all_entity2kgs = {}
        all_entity_global_index = {}
        
        
        for lang in self.kg_names:

            kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations = self.load_kg_data(lang)
            
            kg_object = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, num_entities, num_relations, self.args.device)
            kg_object.get_global_h_t(seeds, pre_langs, all_entity2kgs, all_entity_global_index) # 处理pre_langs, all_entity2kgs, all_entity_global_index, 以及头尾实体的global id
            kg_objects_dict[lang] = kg_object
            
        self.num_entities = len(all_entity2kgs)
        self.num_relations = num_relations

        edge_index, edge_type = get_all_edges(self.kg_dir, kg_objects_dict, all_entity_global_index)
        graph_dir = '_'.join(self.kg_names)
        subgraph_list_path = os.path.join(self.data_dir, graph_dir, 'subgraph_list.graph')
        if not os.path.exists(subgraph_list_path):
            if not os.path.exists(os.path.join(self.data_dir, graph_dir)):
                os.mkdir(os.path.join(self.data_dir, graph_dir))
            subgraph_list = create_subgraph_list(edge_index, edge_type, self.num_entities, self.args.num_hops, self.args.k)
            torch.save(subgraph_list, subgraph_list_path)
        else:
            # subgraph_list = torch.load(subgraph_list_path)
            subgraph_list = []
        
        return kg_objects_dict, subgraph_list, all_entity2kgs, all_entity_global_index
    
    def load_align_links(self):
        seeds = {}
        for f in os.listdir(self.align_dir):
            lang1 = f[:2]
            lang2 = f[3:5]
            links = pd.read_csv(os.path.join(self.align_dir, f), sep='\t',
                                header=None).values.astype(int) # [N_align, 2]
            
            links = torch.LongTensor(links)
            links = torch.unique(links, dim=0)
            seeds[(lang1, lang2)] = torch.LongTensor(links) # [N_align, 2]
            
        return seeds
    

    
    def load_kg_data(self, language):

        train_df = pd.read_csv(os.path.join(self.kg_dir, language + '-train.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
        val_df = pd.read_csv(os.path.join(self.kg_dir, language + '-val.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
        test_df = pd.read_csv(os.path.join(self.kg_dir, language + '-test.tsv'), sep='\t', header=None, names=['head', 'relation', 'tail'])
        
        entity_file = open(os.path.join(self.entity_dir, language + '.tsv'), encoding='utf-8')
        num_entities = len(entity_file.readlines())
        entity_file.close()
        
        relation_file = open(os.path.join(self.data_dir, 'relations.txt'))
        num_relations = len(relation_file.readlines())
        relation_file.close()
        
        triples_train = train_df.values.astype(np.int)
        triples_val = val_df.values.astype(np.int)
        triples_test = test_df.values.astype(np.int)
        
        return torch.LongTensor(triples_train), torch.LongTensor(triples_val), torch.LongTensor(triples_test), num_entities, num_relations
