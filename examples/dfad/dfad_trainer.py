import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
import sys
import json

import gammagl.data
from gammagl.datasets import TUDataset
import tensorlayerx as tlx
import argparse
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.loader import DataLoader
from gan import Generator
from gammagl.loader import DataLoader
from gammagl.models import GINModel, GCNModel, GATModel, GraphSAGE_Full_Model

import numpy as np
from datasets import get_dataset, load_dataloader
import csv
from gammagl.layers.pool.glob import global_sum_pool
from gammagl.models.mlp import MLP

from dfad_network import DFADNetwork



class GeneratorLoss(WithLoss):
    def __init__(self, net, loss_fn, student, teacher):
        super(GeneratorLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.student = student
        self.teacher = teacher

    def forward(self, z, label_no_use):
        generated_graph = self.backbone_network(z)
        nodes_logits, adj = generated_graph
        loader = data_construct(z.shape[0], nodes_logits, adj)
        for data in loader:
            x, edge_index, num_nodes, batch = data.x, data.edge_index, data.num_nodes, data.batch
            student_logits = self.student(x, edge_index, num_nodes, batch)
            teacher_logits = self.teacher(x, edge_index, num_nodes)
            student_logits = tlx.nn.Softmax()(student_logits)
            teacher_logits = tlx.nn.Softmax()(teacher_logits)
        return -self._loss_fn(student_logits, teacher_logits)

class StudentLoss(WithLoss):
    def __init__(self, net, loss_fn, batch_size):
        super(StudentLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def forward(self, data, label):
        print(data['x'].shape)
        num_nodes = data['x'].shape[0] / self.batch_size
        print(num_nodes)
        logits = self.backbone_network(data['x'], data['edge_index'], data['x'].shape[0], data['batch'])
        loss = self._loss_fn(logits, label)
        return loss

def data_construct(batch_size, nodes_logits, adj):
    data_list = []
    for i in range(len(nodes_logits)):
        x = nodes_logits[i]
        print("feature shape of generated graph:", x.shape)
        edge = adj[i]
        graph = gammagl.data.Graph(x = x, edge_index = edge)
        data_list.append(graph)

    return DataLoader(data_list, batch_size = batch_size)


# generator生成图，并利用生成的图训练model
def generate_graph(args):
    z = tlx.random_normal((args.batch_size, args.nz))  # 随机噪声
    generated_graph = generator(z) # z通过生成器，生成一个图
    adj, nodes_logits = generated_graph
    loader = data_construct(args.batch_size, nodes_logits, adj)
    return loader


def train(args, teacher, student, generator, optimizer_s, optimizer_g, test_loader):
    os.makedirs("./dfad_result/{0}".format(args.dataset), exist_ok=True)
    os.makedirs("./student_model/{0}".format(args.dataset), exist_ok=True)
    f = open("./dfad_result/{0}/{0}_{1}.csv".format(args.dataset, fold_number), "w", newline="")
    csv_writer = csv.writer(f)

    teacher.set_eval()

    student_trainable_weight = student.trainable_weights
    generator_trainable_weights = generator.trainable_weights
    loss_fun = tlx.losses.absolute_difference_error
    # s_with_loss = WithLoss(student, loss_fun) # student的损失函数

    s_with_loss = StudentLoss(student, loss_fun, args.batch_size)

    g_with_loss = GeneratorLoss(generator, loss_fun, student, teacher) # generator的损失函数
    s_train_one_step = TrainOneStep(s_with_loss, optimizer_s, student_trainable_weight)
    g_train_one_step = TrainOneStep(g_with_loss, optimizer_g, generator_trainable_weights)

    epochs = args.n_epochs
    student_epochs = args.student_epochs

    best_acc = 0
    for epoch in range(epochs):
        for _ in range(student_epochs):
            # 训练student模型
            student.set_train()
            loader = generate_graph(args)
            for batch in loader:
                t_logits = teacher(batch.x, batch.edge_index, batch.batch)
                print("t_logits.shape:", t_logits.shape)
                s_loss = s_train_one_step(batch, t_logits)
            student.set_eval()

        # 训练generator模型
        generator.set_train()
        z = tlx.random_normal((args.batch_size, args.nz))  # 随机噪声
        g_loss = g_train_one_step(z, None)
        generator.set_eval()

        acc = test(student, test_loader)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   acc: {:.4f}".format(acc))

        if epoch == 0:
            csv_writer.writerow(["epoch", "acc"])
        csv_writer.writerow([epoch, round(acc, 4)])

        if acc > best_acc:
            best_acc = acc
            student.save_weights("./student_model/{0}/{0}_{1}.npz".format(args.dataset, fold_number), format="npz_dict")

    f.close()
    return best_acc



def test(net, test_loader):
    net.set_eval()

    all_preds = []
    all_labels = []

    for batch in test_loader:
        if type(net).__name__ == "GINModel":
            pred = net(batch.x, batch.edge_index, batch.batch).detach()
        else:
            pred = net(batch.x, batch.edge_index, batch.x.shape[1], batch.batch).detach()
        all_preds.append(pred)
        all_labels.append(batch.y)

    all_preds = tlx.convert_to_tensor(np.vstack(all_preds))
    all_labels = tlx.convert_to_tensor(np.concatenate(all_labels))
    all_labels = tlx.reshape(all_labels, (-1, 1))

    all_preds = tlx.cast(all_preds, tlx.int32)
    all_labels = tlx.cast(all_labels, tlx.int32)

    acc = tlx.metrics.acc(all_preds, all_labels)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default='gcn', help="student model")
    parser.add_argument("--student_lr", type=float, default=0.0005, help="learning rate of student model")
    parser.add_argument("--generator_lr", type=float, default=0.0005, help="learning rate of generator")
    parser.add_argument("--n_epochs", type=int, default=80, help="number of epoch")
    parser.add_argument("--student_epochs", type=int, default=5)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_units", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--student_l2_coef", type=float, default=5e-4, help="l2 loss coeficient for student")
    parser.add_argument("--generator_l2_coef", type=float, default=5e-4, help="l2 loss coeficient for generator")
    parser.add_argument('--dataset', type=str, default='COLLAB', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    parser.add_argument("--generator_dropout", type=float, default=0.5)
    parser.add_argument("--student_dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nz", type=int, default=32)
    args = parser.parse_args()

    # get dataset & loader
    dataset_name = args.dataset
    dataset = get_dataset(dataset_name)

    os.makedirs("./dfad_result", exist_ok=True)
    os.makedirs("./student_model", exist_ok=True)

    # 10-fold
    for fold_number in range(1, 11):
        train_loader, test_loader, train_set, test_set = load_dataloader(dataset_name, dataset, args.batch_size, fold_number)
        assert train_set[0].x != None

        # load teacher
        with open("./result/best-acc/{0}/overall_result.txt".format(dataset_name), "r") as f:
            teacher_info = eval(f.read())

        teacher = GINModel(
            in_channels=train_set[0].x.shape[1],
            hidden_channels=teacher_info['hidden_units'],
            out_channels=dataset.num_classes,
            num_layers=teacher_info['num_layers'],
            name="GIN"
        )

        print("teacher:")
        print(train_set[0].x.shape[1], teacher_info['hidden_units'], teacher_info['num_layers'])

        teacher_folder = "./teacher_model/{0}/".format(dataset_name)
        f_name = "{0}_{1}.npz".format(dataset_name, fold_number)
        file_path = os.path.join(teacher_folder, f_name)
        


        # teacher.load_weights(file_path, format="npz_dict", skip=True)

        test_acc = test(teacher, test_loader)
        teacher.load_weights(file_path, format="npz_dict", skip=True)
        '''test_acc = test(teacher, test_loader)
        print(test_acc)'''

        # initialize student
        student = DFADNetwork(
            model_name=args.student,
            feature_dim=train_set[0].x.shape[1],
            hidden_dim=args.hidden_units,
            num_classes=dataset.num_classes,
            num_layers=args.num_layers,
            drop_rate=args.student_dropout
        )


        #initialize generator
        x_example = train_set[0].x
        generator = Generator([64, 128, 256], x_example.shape[0], x_example.shape[1], args.generator_dropout)


        optimizer_s = tlx.optimizers.Adam(lr=args.student_lr, weight_decay=args.student_l2_coef)
        optimizer_g = tlx.optimizers.Adam(lr=args.generator_lr, weight_decay=args.generator_l2_coef)

        best_acc = train(args, teacher, student, generator, optimizer_s, optimizer_g, test_loader)

        os.makedirs("./dfad_result/best-acc/{0}".format(dataset_name), exist_ok=True)
        with open("./dfad_result/best-acc/{0}/{0}_{1}_best-acc.txt".format(dataset_name, fold_number), "w") as f:
            f.write(str(best_acc))

        print("best_acc:", best_acc)
        best_acc_list.append(best_acc)

        best_acc_mean = '{:.4f}'.format(np.array(best_acc_list).mean())
        best_acc_std = '{:.4f}'.format(np.array(best_acc_list).std())
        overall_result = {
            "student": args.student,
            "student_lr": args.student_lr,
            "generator_lr": args.generator_lr,
            "n_epochs": args.n_epochs,
            "student_epochs": args.student_epochs,
            "num_layers": args.num_layers,
            "hidden_units": args.hidden_units,
            "student_l2_coef": args.student_l2_coef,
            "generator_l2_coef": args.generator_l2_coef,
            "dataset": args.dataset,
            "best_acc_mean": best_acc_mean,
            "best_acc_std": best_acc_std
        }
        with open("./dfad_result/best-acc/{0}/overall_result.txt".format(args.dataset), "w") as f:
            f.write(str(overall_result))
