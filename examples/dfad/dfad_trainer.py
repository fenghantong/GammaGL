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
import numpy as np


class GeneratorLoss(WithLoss):
    def __init__(self, net, loss_fn, student, teacher):
        super(GeneratorLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.student = student
        self.teacher = teacher

    def forward(self, z, label_no_use):
        generated_graph = self.backbone_network(z)
        nodes_logits, adj = generated_graph
        loader = data_construct(z.shape[0], nodes_logits, adj)
        x, edge_index, num_nodes = loader[0].x, loader[0].edge_index, loader[0].num_nodes
        student_logits = self.student(x, edge_index, None, num_nodes)
        teacher_logits = self.teacher(x, edge_index, None, num_nodes)
        return -self._loss_fn(student_logits, teacher_logits)

def data_construct(batch_size, nodes_logits, adj):
    data_list = []
    for i in range(nodes_logits.shape[0]):
        x = nodes_logits[i]
        edge = adj[i]
        graph = gammagl.data.Graph(x = x, edge_index = edge)
        data_list.append(graph)
    return DataLoader(data_list, batch_size = batch_size)


# generator生成图，并利用生成的图训练model
def generate_graph(args):
    z = tlx.random_normal((args.batch_size, args.nz))  # 随机噪声
    generated_graph = generator(z) # z通过生成器，生成一个图
    nodes_logits, adj = generated_graph
    loader = data_construct(args.batch_size, nodes_logits, adj)
    return loader



def train(args, teacher, student, generator, optimizer_s, optimizer_g, epochs, student_epochs):
    teacher.set_eval()
    student.set_train()
    generator.set_train()

    student_trainable_weight = student.trainable_weights
    generator_trainable_weights = generator.trainable_weights
    loss_fun = tlx.losses.absolute_difference_error
    s_with_loss = WithLoss(student, loss_fun) # student的损失函数
    g_with_loss = GeneratorLoss(generator, loss_fun, student, teacher) # generator的损失函数
    s_train_one_step = TrainOneStep(s_with_loss, optimizer_s, student_trainable_weight)
    g_train_one_step = TrainOneStep(g_with_loss, optimizer_g, generator_trainable_weights)
    metrics = tlx.metrics.Accuracy()

    for epoch in range(epochs):
        for _ in range(student_epochs):
            # 训练student模型
            loader = generate_graph(args)
            for batch in loader:
                t_logits = teacher(batch.x, batch.edge_index, None, batch.num_nodes)
                s_loss = s_train_one_step(batch, t_logits)

        # 训练generator模型
        z = tlx.random_normal((args.batch_size, args.nz))  # 随机噪声
        g_loss = g_train_one_step(z, None)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   s_loss: {:.4f}".format(s_loss.item())
              + "   g_loss: {:.4f}".format(g_loss.item()))


def test(student, test_loader, epoch):
    student.set_eval()

    all_preds = []
    all_labels = []

    for batch in test_loader:
        pred = student(batch.x, batch.edge_index, None, batch.num_nodes)
        all_preds.append(pred)
        all_labels.append(batch.y)

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = tlx.metrics.acc(all_preds, all_labels)
    print("Epoch {0}, Test: acc = {1}".format(epoch, acc))
    return acc


def test_dataset(dataset_name):
    path = "./dataset" + dataset_name
    dataset = TUDataset(path, dataset_name)
    print(dataset.num_features, dataset.num_classes, dataset.num_node_features)
    print(dataset.data)
    dataloader = DataLoader(dataset, batch_size=32)
    print("length of dataloader: ", len(dataloader))
    for data in dataloader:
        print(data)


if __name__ == '__main__':
    # 1. 处理args

    # 2. 为student准备checkpoint文件

    # 3. 加载dataset和loader

    # 4. 加载teacher

    # 5. 评估teacher准确度

    # 6. load学生模型，生成器

    # 7. 计算总epochs数 ???

    # 8. scheduler ???

    # 9. 训练，并记录最优结果