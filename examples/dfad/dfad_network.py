import gammagl.data
from gammagl.datasets import TUDataset
import tensorlayerx as tlx
import argparse
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.loader import DataLoader
from gan import Generator
from gammagl.loader import DataLoader
from gammagl.models import GINModel, GCNModel, GATModel, GraphSAGE_Full_Model
from gammagl.layers.pool.glob import global_sum_pool
from gammagl.models.mlp import MLP

class DFADNetwork(tlx.nn.Module):
    def __init__(self, model_name, feature_dim, hidden_dim, num_classes, num_layers, drop_rate):
        super(DFADNetwork, self).__init__()
        if model_name == "gcn":
            self.gnn = GCNModel(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_class=hidden_dim, # 此处先让模型输出hidden_dim大小的logits。然后经过池化得到分类结果
                num_layers=num_layers
            )
        elif model_name == "gin":
            self.gnn = GINModel(
                in_channels=feature_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers
            )
        elif model_name == "gat":
            self.gnn = GATModel(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_class=hidden_dim,
                heads=3,
                drop_rate=drop_rate,
                num_layers=num_layers
            )
        elif model_name == "graphsage":
            self.gnn = GraphSAGE_Full_Model(
                in_feats=feature_dim,
                n_hidden=hidden_dim,
                n_classes=hidden_dim,
                n_layers=num_layers,
                activation=tlx.nn.ReLU(),
                dropout=drop_rate,
                aggregator_type="mean"
            )
        else:
            raise NameError("model name error")

        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mlp = MLP([hidden_dim, hidden_dim, num_classes])



    def forward(self, x, edge_index, num_nodes, batch):
        print(x.shape)
        if self.model_name == "gcn":
            logits = self.gnn(x, edge_index, None, num_nodes)
        elif self.model_name == "gin":
            logits = self.gnn(x, edge_index, batch)
        elif self.model_name == "gat":
            print("TAT:", num_nodes)
            logits = self.gnn(x, edge_index, num_nodes)
        elif self.model_name == "graphsage":
            logits = self.gnn(x, edge_index)
        else:
            raise NameError("model name error")

        if self.model_name != "gin":
            print("Before global sum pool: ", logits.shape, batch.shape)
            logits = global_sum_pool(logits, batch)
            return self.mlp(logits)
        else:
            return logits


