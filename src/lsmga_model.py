import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn import GNN
from src.utils import nodes_to_graph

import math

class LSMGA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_kgs = args.num_kgs
        self.num_entities = args.num_entities
        self.num_relations = args.num_relations
        
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.device = args.device
        self.criterion = nn.MarginRankingLoss(margin=args.margin, reduction='mean')
        
        # embedding init
        self.entity_embedding_layer = nn.Embedding(self.num_entities, self.entity_dim)
        nn.init.xavier_uniform_(self.entity_embedding_layer.weight)
        
        self.rel_embedding_layer = nn.Embedding(self.num_relations, self.relation_dim)
        nn.init.xavier_uniform_(self.rel_embedding_layer.weight)
        
        self.relation_prior = nn.Embedding(self.num_relations, 1)
        nn.init.xavier_uniform_(self.relation_prior.weight)

        self.kg_embedding_layer = nn.Embedding(self.num_kgs, self.entity_dim)
        nn.init.xavier_uniform_(self.kg_embedding_layer.weight)
        
        # create GNN encoder
        self.encoder_KG = GNN(num_kgs=args.num_kgs, in_dim=args.entity_dim, in_edge_dim=args.relation_dim, n_hid=args.encoder_hdim_gnn,
                             out_dim=args.entity_dim, n_heads=args.n_heads, n_layers=args.n_layers_gnn, dropout=args.dropout)


        self.n_heads = 1
        self.dropout = nn.Dropout(args.dropout)
        
        self.d_input = args.entity_dim + args.entity_dim
        self.d_q = args.entity_dim // args.n_heads
        self.d_k = args.entity_dim // args.n_heads
        self.d_v = args.entity_dim // args.n_heads
        self.d_sqrt = math.sqrt(self.d_input // args.n_heads)
        
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for i in range(self.n_heads)])
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(args.entity_dim, self.d_v, bias=True) for i in range(self.n_heads)])
        self.fc = nn.Linear(self.d_v * self.n_heads, self.entity_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.entity_dim)
    
    
    def cross_graph_attention(self, x_gnn_output_all, kg_index):
        """
        x_gnn_output_all: [num_kgs, batch_size, out_dim]
        """
        x_gnn_output_all = x_gnn_output_all.permute(1, 0, 2) # [batch_size, num_kgs, out_dim]
        batch_size = x_gnn_output_all.shape[0]
        kgs = torch.arange(self.num_kgs).to(self.device)
        kg_embeddings = self.kg_embedding_layer(kgs) # [num_kgs, kg_dim]
        kg_expand = kg_embeddings.expand(batch_size, kg_embeddings.shape[0], kg_embeddings.shape[1]) # [batch_size, num_kgs, kg_dim]
        x_out = []
        for i in range(self.n_heads):
            q_linear = self.w_q_list[i]
            k_linear = self.w_k_list[i]
            v_linear = self.w_v_list[i]
            
            q_transfer = q_linear(torch.cat([x_gnn_output_all[:, kg_index:(kg_index + 1), :], 
                                             kg_expand[:, kg_index:(kg_index + 1), :]], dim=2)) # [batch_size, 1, d_q]
            k_transfer = k_linear(torch.cat([x_gnn_output_all, kg_expand], dim=2)) # [batch_size, num_kgs, d_k]
            v_transfer = v_linear(x_gnn_output_all) # [batch_size, num_kgs, d_v]
            
            attention = torch.bmm(q_transfer, k_transfer.permute(0, 2, 1)) # [batch_size, 1, num_kgs]
            attention = torch.div(attention, self.d_sqrt) # [batch_size, 1, num_kgs]
            attention_norm = F.softmax(attention, dim=2) # [batch_size, 1, num_kgs]
            x_out_head_i = torch.bmm(attention_norm, v_transfer) # [batch_size, 1, d_v]
            x_out.append(x_out_head_i.squeeze(1)) # [batch_size, d_v]
        x_out = torch.cat(x_out, dim=1) # [batch_size, d_out]
        
        residual = x_gnn_output_all[:, kg_index, :]
        x_out = self.layer_norm(residual + F.gelu(x_out))
        
        return x_out
    
    
    def forward_GNN_embedding(self, graph_input_list, kg_index):

        x_gnn_output_all = []
        
        for graph_input in graph_input_list:

            x_features = self.entity_embedding_layer(graph_input.x) # [num_nodes, entity_dim]
            edge_index = graph_input.edge_index # [2, num_edges]
            edge_kg_index = graph_input.edge_kg_index # [2, num_edges]
            edge_beta_r = self.relation_prior(graph_input.edge_attr) # [num_edges, 1]

            edge_relation_embedding = self.rel_embedding_layer(graph_input.edge_attr) # [num_edges, relation_dim]
            x_gnn_output = self.encoder_KG(x_features, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding,
                                  graph_input.y, graph_input.num_size) # [batch_size, out_dim]
            x_gnn_output_all.append(x_gnn_output)
        x_gnn_output_all = torch.stack(x_gnn_output_all) # [num_kgs, batch_size, out_dim]
        
        x_out = self.cross_graph_attention(x_gnn_output_all, kg_index)

        return x_out
        
    
    def forward_kg(self, h_graph, sample, t_graph, t_neg_graph, kg_index):

        h = self.forward_GNN_embedding(h_graph, kg_index).unsqueeze(1) # [batch_size, 1, d]
        r = self.rel_embedding_layer(sample[:, 1]).unsqueeze(1) # [batch_size, 1, d]
        t = self.forward_GNN_embedding(t_graph, kg_index).unsqueeze(1) # [batch_size, 1, d]
        t_neg = self.forward_GNN_embedding(t_neg_graph, kg_index).unsqueeze(1) # [batch_size, 1, d]
        
        projected_t = self.project_t([h, r]) # h + r
        pos_loss = self.define_loss([t, projected_t]) # norm(h + r - t) [batch_size, 1]
        neg_losses = self.define_loss([t, t_neg]) # norm(t_neg - t) [batch_size, num_neg]

        
        # TransE
        neg_loss = torch.mean(neg_losses, dim=-1) # [batch_size, 1]
        
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        total_loss = self.criterion(pos_loss, neg_loss, target)
        
        return total_loss
    
    def project_t(self, hr):
        return hr[0] + hr[1]
    
    def define_loss(self, t_true_pred):
        t_true = t_true_pred[0]
        t_pred = t_true_pred[1]
        return torch.norm(t_true - t_pred + 1e-8, dim=2)
    
    def predict(self, h_emb, r):
        
        entity_dim = h_emb.shape[1]
        h = h_emb.view(-1, entity_dim).unsqueeze(1) # [batch_size, 1, d]
        r = self.rel_embedding_layer(r).unsqueeze(1) # [batch_size, 1, d]
        projected_t = self.project_t([h, r]) # h + r, [batch_size, 1, d]
        
        return projected_t

