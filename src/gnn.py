import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import math
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_max, scatter_add


def softmax(src, index, num_nodes=None):
    """
    src: [n_heads, num_edges, 1]
    index: [num_edges,]
    """
    out = []
    for s in src:
        # s: [num_edges, 1]
        out_s = s - scatter_max(s, index, dim=0, dim_size=num_nodes)[0][index]

        out_s = out_s.exp() # [num_edges, 1]
        out_s = out_s / (scatter_add(out_s, index, dim=0, dim_size=num_nodes)[index] + 1e-16) # [num_edges, 1]
        out.append(out_s)
    out = torch.stack(out)
    return out


class MGA(MessagePassing):
    def __init__(self, num_kgs=1, n_heads=2, d_input=32, d_input_edge=32, d_out=32, dropout=0.1):

        super().__init__(aggr='add')
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        
        self.d_input = d_input
        self.d_q = d_out // n_heads
        self.d_k = d_out // n_heads
        self.d_v = d_out // n_heads
        # d_k = d_q = d_v
        self.d_sqrt = math.sqrt(d_out // n_heads)
        
        # Attention Layer
        self.w_q_list = nn.ModuleList()
        self.w_k_list = nn.ModuleList()
        self.w_v_list = nn.ModuleList()
        self.w_transfer_list = nn.ModuleList()
        for _ in range(num_kgs):
            self.w_q_list.append(nn.Linear(self.d_input, self.d_q * n_heads, bias=True))
            self.w_k_list.append(nn.Linear(self.d_input, self.d_k * n_heads, bias=True))
            self.w_v_list.append(nn.Linear(self.d_input, self.d_v * n_heads, bias=True))
            self.w_transfer_list.append(nn.Linear(self.d_input + d_input_edge, self.d_input, bias=True))
        
        # Normalization
        self.layer_norm = nn.LayerNorm(d_input)
    
    def forward(self, x, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding):

        num_nodes = x.shape[0]
        residual = x # [num_nodes, d_input]
        x = self.layer_norm(x) # [num_nodes, d_input]
        return self.propagate(edge_index, x=x, edge_kg_index=edge_kg_index, edge_beta_r=edge_beta_r,
                              edge_relation_embedding=edge_relation_embedding, residual=residual, num_nodes=num_nodes)
    
    def message(self, x_j, x_i, edge_index_i, edge_kg_index, edge_beta_r, edge_relation_embedding, num_nodes):

        edge_kg_index_j = edge_kg_index[0] # [num_edges,]
        edge_kg_index_i = edge_kg_index[1] # [num_edges,]
        messages = []
        edge_value = edge_beta_r.view(-1, 1) # beta_r, [num_edges, 1]
        x_j_transfer = F.gelu(self.compute_transfer(torch.cat([x_j, edge_relation_embedding], dim=1), self.w_transfer_list, self.d_input,
                                                    edge_kg_index_j)) # [num_edges, d_input]
        attention = self.multi_head_cross_attention(x_i, x_j_transfer, edge_value, edge_kg_index_i, edge_kg_index_j) # [n_heads, num_edges, 1]
        attention = torch.div(attention, self.d_sqrt) # [n_heads, num_edges, 1]
        attention_norm = softmax(attention, edge_index_i, num_nodes) # [n_heads, num_edges, 1]

        sender = x_j_transfer # [num_edges, d_v * n_heads]

        sender = sender.view(-1, self.n_heads, self.d_v).transpose(0, 1) # [n_heads, num_edges, d_v]
        
        message = attention_norm * sender # [n_heads, num_edges, d_v]
        message_all_head = message.transpose(0, 1).reshape(-1, self.d_v * self.n_heads) # [num_edges, d_v * n_heads]
        
        return message_all_head

    
    def compute_transfer(self, input_x, linear_transfer, output_dim, input_edge_kg_index):

        out_x = input_x.new_empty((input_x.shape[0], output_dim))
        for i in range(len(linear_transfer)):
            idx_i = torch.nonzero(input_edge_kg_index == i).view(-1)
            input_x_i = input_x[idx_i] # 取出第i个图谱的输入特征
            out_i = linear_transfer[i](input_x_i)
            out_x[idx_i] = out_i
        return out_x
    
    def multi_head_cross_attention(self, x_i, x_j_transfer, edge_value, edge_kg_index_i, edge_kg_index_j):

        x_i = self.compute_transfer(x_i, self.w_q_list, self.d_q * self.n_heads, edge_kg_index_i) # [num_edges, d_q * n_heads]
        x_i = x_i.view(-1, self.n_heads, self.d_q).transpose(0, 1) # [n_heads, num_edges, d_q]
        x_j = self.compute_transfer(x_j_transfer, self.w_k_list, self.d_k * self.n_heads, edge_kg_index_j) # [num_edges, d_k * n_heads]
        x_j = x_j.view(-1, self.n_heads, self.d_k).transpose(0, 1) # [n_heads, num_edges, d_k]
        
        attention = torch.matmul(torch.unsqueeze(x_j, dim=2), torch.unsqueeze(x_i, dim=3)) # [n_heads, num_edges, 1, 1]
        edge_value = torch.unsqueeze(edge_value, dim=2) # [num_edges, 1, 1]
        
        attention = attention * edge_value # [n_heads, num_edges, 1, 1]
        return torch.squeeze(attention, dim=2) # [n_heads, num_edges, 1]
    
    def update(self, aggr_out, residual):
        x_new = residual + F.gelu(aggr_out)
        
        return self.dropout(x_new)
    
    def __repr__(self):
        return '{}'.format(self.__class__.__name__)



class GNN(nn.Module):
    def __init__(self, num_kgs, in_dim, in_edge_dim, n_hid, out_dim, n_heads, n_layers, dropout=0.1):

        super().__init__()
        self.gnn_layers = nn.ModuleList()
        self.in_dim = in_dim 
        self.in_edge_dim = in_edge_dim
        self.n_hid = n_hid
        self.dropout = nn.Dropout(dropout)
        self.num_kgs = num_kgs
        
        
        for l in range(n_layers):
            self.gnn_layers.append(MGA(num_kgs=num_kgs, n_heads=n_heads, d_input=in_dim, d_input_edge=in_edge_dim, d_out=out_dim, dropout=dropout))

    def forward(self, x, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding, y=None, s=None):

        h_t = self.dropout(x)

        for layer in self.gnn_layers:
            h_t = layer(h_t, edge_index, edge_kg_index, edge_beta_r, edge_relation_embedding) # [num_nodes, out_dim]

        if y != None:
            true_indexes = self.get_real_index(y, s) # [num_graphs,]
            h_t = torch.index_select(h_t, 0, true_indexes) # [num_graphs, out_dim]
        
        return h_t
    
    def get_real_index(self, y, s):

        num_graphs = y.shape[0]
        node_base = torch.LongTensor([0]).to(y.device)
        true_index = []
        
        for i in range(num_graphs):
            true_index_each = node_base + y[i]
            node_base += s[i]
            true_index.append(true_index_each.view(-1, 1))
        true_index = torch.cat(true_index).to(y.device)
        true_index = true_index.view(-1)
        
        return true_index
            
        
