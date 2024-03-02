import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Linear, Dropout
import numpy as np
import scipy.sparse as sp

def dense_to_sparse(adj_mat):
    sparse = sp.coo_matrix(adj_mat)
    row = tlx.convert_to_tensor(sparse.row)
    col = tlx.convert_to_tensor(sparse.col)
    return tlx.stack((row, col))

class Generator(tlx.nn.Module):
    def __init__(self, conv_dims, num_vertices, num_features, dropout):
        super(Generator, self).__init__()

        self.num_vertices = num_vertices
        self.num_features = num_features

        layer_list = []
        for out_channels in conv_dims:
            layer_list.append(Linear(out_features=out_channels, act=tlx.ReLU))
            layer_list.append(Dropout(dropout))

        self.layer_list = tlx.nn.Sequential(layer_list)
        self.nodes_layer = Linear(out_features=num_vertices * num_features)

    def forward(self, x):
        output = self.layer_list(x)
        nodes_logits = self.nodes_layer(output)
        nodes_logits = tlx.reshape(nodes_logits, [-1, self.num_vertices, self.num_features])

        inner_product = tlx.ops.matmul(nodes_logits, tlx.ops.transpose(nodes_logits, [0, 2, 1]))
        adj = tlx.nn.Sigmoid()(inner_product)
        adj = adj > 0.5

        sparse_adj = [dense_to_sparse(adj_mat) for adj_mat in adj]

        return sparse_adj, nodes_logits

'''
z_dim = 8
generator = Generator([32, 64], z_dim, 10, 6, 0.2)
batch_size = 2
z = tlx.random_normal((batch_size, 8))
print(generator(z)) '''
