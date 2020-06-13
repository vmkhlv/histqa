import torch
import numpy as np


"""
pytorch implementation of graph-based question answering model architecture proposed in https://arxiv.org/abs/1808.09920
"""


def tile(tensor, shape, dim):
    init_dim = tensor.size(dim)
    repeat_idx = [1] * tensor.dim()
    repeat_idx[dim] = shape
    tensor = tensor.repeat(*(repeat_idx))
    order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(shape) + i for i in range(init_dim)]))
    return torch.index_select(tensor, dim, order_index)


def flatten(x, max_len, elmo_num_layers=3, elmo_layer_dim=1024, shape=-1):
    """
    :input: x (either nodes or query) (batch_size, max_x_len, elmo_num_layers, dim)
    :returns: flatten x (batch_size, max_x_size, elmo_num_layers * dim)
    """
    x = torch.tensor(x, dtype=torch.float32)
    return torch.reshape(x, (shape, max_len, elmo_num_layers * elmo_layer_dim))


class QueryDecoder(torch.nn.Module):
    def __init__(self, input_dim=3072, hidden_dim1=256, hidden_dim2=128, num_layers=2, max_query_size=25,
                 dropout_rate=0.35):
        super(QueryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_layers = num_layers
        self.max_query_size = max_query_size
        self.dropout_rate = dropout_rate
        self.lstm1 = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim1,
                                   num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=self.hidden_dim1 * self.num_layers, hidden_size=self.hidden_dim2,
                                   num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, query):
        x = flatten(query, self.max_query_size)
        x1, (h_n1, c_n1) = self.lstm1(x)
        x2, (h_n2, c_n2) = self.lstm2(x1)
        x2 = self.dropout(x2)
        return x2


class CandidateDecoder(torch.nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256, max_nodes=250, dropout_rate=0.35):
        super(CandidateDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.dropout_rate = dropout_rate
        self.fc = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, n):
        x = flatten(n, self.max_nodes)
        x = self.fc(x)
        x = self.tanh(x)
        x = self.dropout(x)
        return x


class RGCN(torch.nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, dropout_rate=0.35):
        super(RGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.input_dim * 2, self.hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

    def hop_layer(self, hidden_tensor, hidden_mask, adj):
        h_mu = torch.unsqueeze(hidden_mask, -1)
        h_t = torch.stack([self.fc1(hidden_tensor) for n in range(adj.shape[1])], 1)
        h_m = torch.unsqueeze(h_mu, 1)
        h_t = h_t * h_m
        u = torch.sum(torch.matmul(adj, h_t), 1) + self.fc1(hidden_tensor) * h_mu
        att = self.sigmoid(self.fc2(torch.cat((u, hidden_tensor), -1))) * h_mu
        x = att * self.tanh(u) + (1 - att) * hidden_tensor
        x = self.dropout(x)
        return x


class NodeClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=256, hidden_dim2=128, output_dim=1,
                 batch_size=32, max_candidates=25, max_nodes=250):
        super(NodeClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.max_nodes = max_nodes
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = torch.nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, bmask):
        x1 = self.tanh(self.fc1(x))
        x1 = self.tanh(self.fc2(x1))
        x1 = torch.squeeze(self.fc3(x1), -1)
        x2 = bmask * torch.unsqueeze(x1, 1)
        x2 = torch.where(
            x2.eq(torch.tensor(0, dtype=torch.float32)),
            torch.from_numpy(
                (np.ones((self.batch_size, self.max_candidates, self.max_nodes)) * -np.inf).astype(np.float32)
            ),
            x2
        )
        x2 = torch.max(x2, -1).values
        return x2


class HistQANet(torch.nn.Module):
    def __init__(self, max_nodes=250, max_query_size=25, max_candidates=25, batch_size=32):
        super(HistQANet, self).__init__()
        self.max_nodes = max_nodes
        self.max_query_size = max_query_size
        self.max_candidates = max_candidates
        self.query_decoder = QueryDecoder(max_query_size=self.max_query_size)
        self.candidate_decoder = CandidateDecoder(max_nodes=self.max_nodes)
        self.rgcn = RGCN()
        self.clf = NodeClassifier(max_nodes=self.max_nodes, max_candidates=self.max_candidates, batch_size=batch_size)

    def read_batch(self, x):
        self.nodes = torch.tensor(x["nodes"])
        self.nodes_length = torch.tensor(x["nodes_length"])
        self.query = torch.tensor(x["query"])
        self.query_length = torch.tensor(x["query_length"])
        self.adj = torch.tensor(x["adjacency"])
        self.bmask = torch.tensor(x["bmask"])

    def create_mask(self):
        nodes_mask = tile(
            torch.unsqueeze(torch.arange(self.max_nodes, dtype=torch.int32), 0),
            self.nodes_length.size()[0], 0
        ) < torch.unsqueeze(self.nodes_length, -1)
        return nodes_mask.float()

    def create_nodes(self):
        decoded_query = self.query_decoder.forward(self.query)
        decoded_cand = self.candidate_decoder.forward(self.nodes)
        nodes_mask = self.create_mask()
        nodes = torch.cat(
            ((decoded_cand), tile(decoded_query, self.max_nodes // self.max_query_size, 1)),
            -1) * tile(torch.unsqueeze(nodes_mask, -1), 512, -1)
        return nodes, nodes_mask, decoded_query

    def forward(self, x, hop_num=1):
        nodes, nodes_mask, decoded_query = self.create_nodes()
        last_hop = nodes
        for hop in range(hop_num):
            last_hop = self.rgcn.hop_layer(nodes, nodes_mask, self.adj)
        x = torch.cat(
            (last_hop, tile(decoded_query, self.max_nodes // self.max_query_size, 1)), -1
        ) * torch.unsqueeze(nodes_mask, -1)
        prediction = self.clf(x, self.bmask)
        return prediction
