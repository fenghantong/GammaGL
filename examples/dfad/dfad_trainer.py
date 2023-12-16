import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'tensorflow'
import sys
from pathlib import Path

import gammagl.data

current_file_path = Path(__file__)

parent_dir = current_file_path.parent.parent.parent

sys.path.append(str(parent_dir))

from gammagl.datasets import TUDataset
import tensorlayerx as tlx
import argparse
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.loader import DataLoader
from gammagl.models.gan import Generator
from gammagl.loader import DataLoader


class GeneratorLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(GeneratorLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data)
        return -self._loss_fn(logits, label)

def data_construct(args, nodes_logits, adj):
    data_list = []
    for i in range(nodes_logits.shape[0]):
        x = nodes_logits[i]
        edge = adj[i]
        graph = gammagl.data.Graph(x = x, edge_index = edge)
        data_list.append(graph)
    return DataLoader(data_list, batch_size = args.batch_size)

# generator生成图，并利用生成的图训练model
def generate_and_train(model, train_one_step, teacher):
    z = tlx.random_normal((args.batch_size, args.nz))  # 随机噪声
    model.set_train()
    generated_graph = generator(z)
    nodes_logits, adj = generated_graph
    loader = data_construct(args, nodes_logits, adj)
    for data in loader:
        t_logits = teacher(data)
        train_one_step(data, t_logits)


def train(args, teacher, student, generator, optimizer, epochs, student_epochs):
    student_trainable_weight = student.trainable_weights
    generator_trainable_weights = generator.trainable_weights
    loss_fun = tlx.losses.absolute_difference_error
    s_loss = WithLoss(student, loss_fun) # student的损失函数
    g_loss = GeneratorLoss(generator, loss_fun) # generator的损失函数
    s_train_one_step = TrainOneStep(s_loss, optimizer, student_trainable_weight)
    g_train_one_step = TrainOneStep(g_loss, optimizer, generator_trainable_weights)
    metrics = tlx.metrics.Accuracy()

    for epoch in range(epochs):
        for _ in range(student_epochs):
            # 训练student模型
            generate_and_train(student, s_train_one_step, teacher)

        # 训练generator模型
        generate_and_train(generator, g_train_one_step, teacher)




def test_dataset(dataset_name):
    path = "./dataset" + dataset_name
    dataset = TUDataset(path, dataset_name)
    print(dataset.num_features, dataset.num_classes, dataset.num_node_features)
    print(dataset.data)
    dataloader = DataLoader(dataset, batch_size=32)
    print("length of dataloader: ", len(dataloader))
    for data in dataloader:
        print(data)


test_dataset("PTC-MR")