import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.utils import k_hop_subgraph

import os
from tqdm import tqdm
import copy
import time


class MyData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_kg_index':
            return 0
        else:
            return super().__inc__(key, value, *args, **kwargs)


def save_model(model, output_dir, filename, args):
    """
    Save the trained knowledge model under output_dir. Filename: 'language.h5'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the weights for the whole model
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'args': args,
    }, ckpt_path)


def get_negative_samples_graph(batch_size_each, num_entity):

    rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,1]

    return rand_negs



def ranking_all_batch(predicted_t, embedding_matrix, k = None):

    num_total_entities = embedding_matrix.shape[0]
    predicted_t = torch.unsqueeze(predicted_t, dim=1)  # [b,1,d]
    predicted_t = predicted_t.repeat(1, num_total_entities, 1)  # [b,n,d]

    distance = torch.norm(predicted_t - embedding_matrix, dim=2)  # [b,n]

    if k==None:
        k = num_total_entities

    top_k_scores, top_k_indexes = torch.topk(-distance, k=k) # [b, k], [b, k]
    return top_k_indexes, top_k_scores




def get_language_list(entity_dir):
    entity_files = os.listdir(entity_dir)
    entity_files = sorted(entity_files)
    
    kg_names = [e[:2] for e in entity_files]
    
    return kg_names


def get_kg_edges_for_each(kg_dir, language):
    """
    """
    train_df = pd.read_csv(os.path.join(kg_dir, language + '-train.tsv'), sep='\t',
                          header=None, names=['head', 'relation', 'tail'])
    val_df = pd.read_csv(os.path.join(kg_dir, language + '-val.tsv'), sep='\t',
                         header=None, names=['head', 'relation', 'tail'])
    
    #training data graph construction
    sender_node_list = train_df['head'].values.astype(np.int).tolist()
    sender_node_list += train_df['tail'].values.astype(np.int).tolist()
    
    receiver_node_list = train_df['tail'].values.astype(np.int).tolist() 
    receiver_node_list += train_df['head'].values.astype(np.int).tolist()

    
    edge_relation_list = train_df['relation'].values.astype(np.int).tolist()
    edge_relation_list += train_df['relation'].values.astype(np.int).tolist()

    edge_index = torch.LongTensor(np.vstack((sender_node_list, receiver_node_list))) # [2, num_edges]
    edge_relation = torch.LongTensor(np.asarray(edge_relation_list)) # [num_edges]
    
    return edge_index, edge_relation


def get_all_edges(kg_dir, kg_objects_dict, all_entity_global_index):
    """
    return: edge_index, edge_relation
    """
    edge_index_list = []
    edge_relation_list = []
    def get_global(x, language):
        return all_entity_global_index[language][x]
    
    for language in kg_objects_dict:
        edge_index, edge_relation = get_kg_edges_for_each(kg_dir, language)
        
        edge_index_list.append(edge_index.apply_(lambda x: get_global(x, language)))
        edge_relation_list.append(edge_relation)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_relation = torch.cat(edge_relation_list, dim=0)
    return edge_index, edge_relation
    
def create_subgraph_list(edge_index, edge_type, total_num_nodes, num_hops, k):
    subgraph_list = []
    num_edges = []
    node_list = list(range(total_num_nodes))
    for i in tqdm(node_list):
        [subgraph_node_ids, edge_index_each, node_position, edge_masks] = k_hop_subgraph([i], num_hops, edge_index, num_nodes=total_num_nodes, relabel_nodes=True)
        x = subgraph_node_ids
        edge_index_each = edge_index_each[:, :k]
        edge_type_masked = (edge_type + 1) * edge_masks
        edge_attr = edge_type_masked[edge_type_masked.nonzero(as_tuple=True)] - 1
        edge_attr = edge_attr[:k]
        
        assert edge_attr.shape[0] == edge_index_each.shape[1]
        
        node_position = torch.LongTensor([node_position])
        num_size = torch.LongTensor([len(subgraph_node_ids)])
        subgraph_each = Data(x=x, edge_index=edge_index_each, edge_attr=edge_attr, y=node_position, num_size=num_size)
        subgraph_list.append(subgraph_each)
        num_edges.append(edge_index_each.shape[1])
    
    print('Average subgraph edges %.2f' % np.mean(num_edges))
    
    return subgraph_list

            
def nodes_to_graph(subgraph_list, node_index, batch_size=-1):

    one_batch = False
    if batch_size == -1:

        batch_size = node_index.shape[0]
        one_batch = True
    
    graphs = [subgraph_list[i.item()] for i in node_index]
    graph_loader = GraphLoader(graphs, batch_size=batch_size, shuffle=False)
    
    if one_batch:
        for batch in graph_loader:
            assert batch.edge_index.shape[1] == batch.edge_attr.shape[0]
            return batch
    else:
        return graph_loader



def get_k_subgraph_list(subgraph_list, node_index, kg_index, num_kgs, entity2kgidx, total_num_nodes, data_dir):

    k_subgraph_list = []
    def get_kg_index(x, subset, node2kgidx):
        return node2kgidx[subset[x].item()]
    print('get_k_subgraph_list: kg_index: ', kg_index)
    k_subgraph_list_path = os.path.join(data_dir, 'kg' + str(kg_index) + '_k_subgraph_list.graph')
    if os.path.exists(k_subgraph_list_path):
        k_subgraph_list = torch.load(k_subgraph_list_path)
        return k_subgraph_list
    for i in tqdm(node_index):
        graphs = []
        subgraph = subgraph_list[i]
        nodes = subgraph.x
        row, col = subgraph.edge_index
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr
        y = subgraph.y

        for j in range(num_kgs):
            subset_j = [i]
            tmp_node2kgidx = {i: kg_index}
            edge_mask = edge_index.new_empty((2, edge_index.shape[1]), dtype=torch.bool) # [2, num_edges]
            edge_mask.fill_(False)
            for r in range(2):
                for c in range(edge_index.shape[1]):
                    node_rc = nodes[edge_index[r][c]]
                    if j in entity2kgidx[node_rc.item()] or node_rc == i:
                        subset_j.append(node_rc.item())
                        edge_mask[r][c]= True
                        if node_rc != i:
                            tmp_node2kgidx[node_rc.item()] = j
            subset_j, inv = torch.tensor(subset_j).unique(return_inverse=True)
            inv = inv[:1]
            edge_mask = edge_mask[0] & edge_mask[1]
            edge_index_j = edge_index[:, edge_mask]
            edge_index_j = nodes[edge_index_j]
            # relable_nodes
            tmp_index = edge_index_j.new_full((total_num_nodes, ), -1)
            tmp_index[subset_j] = torch.arange(subset_j.size(0))
            edge_index_j = tmp_index[edge_index_j]
            edge_attr_j = edge_attr[edge_mask]
            num_size_j = torch.LongTensor([len(subset_j)])
            
            edge_kg_index = copy.deepcopy(edge_index_j)
            edge_kg_index.apply_(lambda x: get_kg_index(x, subset_j, tmp_node2kgidx)) # [2, num_edges]
            graphs.append(MyData(x=subset_j, edge_index=edge_index_j, edge_kg_index=edge_kg_index, edge_attr=edge_attr_j, y=inv, num_size=num_size_j))
                
        k_subgraph_list.append(graphs)
    # k_subgraph_list: [num_nodes, num_kgs]
    torch.save(k_subgraph_list, k_subgraph_list_path)
    return k_subgraph_list


def nodes_to_k_graph(k_subgraph_list, node_index, device, shuffle=False):
    batch_size = node_index.shape[0]

    graph_batches = []
    for i in range(len(k_subgraph_list[0])):
        graphs = [k_subgraph_list[j.item()][i] for j in node_index]
        graph_loader = GraphLoader(graphs, batch_size=batch_size, shuffle=shuffle)
        for batch in graph_loader:
            graph_batches.append(batch.to(device))
    return graph_batches
