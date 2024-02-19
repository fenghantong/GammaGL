import gammagl.data
from gammagl.data import Graph
import numpy as np
from gammagl.datasets import TUDataset
import tensorlayerx as tlx
from gammagl.loader import DataLoader
from gammagl.transforms import BaseTransform
from gammagl.utils import degree
from my_utils import one_hot

# 对节点出度进行独热编码作为节点特征
class OneHotDegree(BaseTransform):
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def __call__(self, graph: Graph):
        # assert graph.edge_index is not None
        deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes, dtype=tlx.int32)
        one_hog_deg = one_hot(deg, self.max_degree + 1)
        one_hog_deg = tlx.cast(one_hog_deg, np.float32)
        graph.x = one_hog_deg
        return graph

# 对节点出度规范化作为节点特征
class NormalizedDegree(BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, graph: Graph):
        assert graph.edge_index is not None
        deg = degree(graph.edge_index[0], num_nodes = graph.num_nodes)
        normalized_deg = (deg - self.mean) / self.std
        graph.x = tlx.reshape(normalized_deg, (-1, 1))
        return graph

def get_dataset(dataset_name):
    path = "./dataset"
    dataset = TUDataset(path, dataset_name)
    transform = None

    if dataset.data.x is None:
        degree_list = []
        loader = DataLoader(dataset, batch_size=1)
        for graph in loader:
            deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
            degree_list.append(deg)

        degrees = tlx.convert_to_tensor(np.concatenate(degree_list))
        max_deg = tlx.reduce_max(degrees)

        if max_deg <= 500:
            transform = OneHotDegree(max_deg)
        else:
            mean_deg = tlx.reduce_mean(degrees)
            std = tlx.reduce_std(degrees)
            print(mean_deg)
            transform = NormalizedDegree(mean_deg, std)

    return TUDataset(path, dataset_name, transform=transform)

#dataset = get_dataset("COLLAB")
#print(dataset[0].x)