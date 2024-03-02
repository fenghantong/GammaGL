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



class GeneratorLoss(WithLoss):
    def __init__(self, net, loss_fn, student, teacher):
        super(GeneratorLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.student = student
        self.teacher = teacher

    def student_forward(self, x, edge_index, num_nodes, batch):
        model_name = type(self.student).__name__
        if model_name == "GCNModel":
            return self.student(x, edge_index, None, num_nodes)
        elif model_name == "GINModel":
            return self.student(x, edge_index, batch)
        elif model_name == "GATModel":
            return self.student(x, edge_index, num_nodes)
        elif model_name == "GraphSAGE_Full_Model":
            return self.student(x, edge_index)
        else:
            raise NameError("Model name error")

    def forward(self, z, label_no_use):
        generated_graph = self.backbone_network(z)
        nodes_logits, adj = generated_graph
        loader = data_construct(z.shape[0], nodes_logits, adj)
        x, edge_index, num_nodes, batch = loader[0].x, loader[0].edge_index, loader[0].num_nodes, loader[0].batch
        student_logits = self.student_forward(x, edge_index, num_nodes, batch)
        teacher_logits = self.teacher(x, edge_index, num_nodes)
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


def train(args, teacher, student, generator, optimizer_s, optimizer_g, test_loader):
    os.makedirs("./dfad_result/{0}".format(args.dataset), exist_ok=True)
    os.makedirs("./student_model/{0}".format(args.dataset), exist_ok=True)
    f = open("./dfad_result/{0}/{0}_{1}.csv".format(args.dataset, fold_number), "w", newline="")
    csv_writer = csv.writer(f)

    teacher.set_eval()

    student_trainable_weight = student.trainable_weights
    generator_trainable_weights = generator.trainable_weights
    loss_fun = tlx.losses.absolute_difference_error
    s_with_loss = WithLoss(student, loss_fun) # student的损失函数
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
                t_logits = teacher(batch.x, batch.edge_index, None, batch.num_nodes)
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
        pred = net(batch.x, batch.edge_index, batch.batch)
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
    parser.add_argument("--n_epochs", type=int, default=120, help="number of epoch")
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_units", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--student_l2_coef", type=float, default=5e-4, help="l2 loss coeficient for student")
    parser.add_argument("--generator_l2_coef", type=float, default=5e-4, help="l2 loss coeficient for generator")
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')
    args = parser.parse_args()

    # get dataset & loader
    dataset_name = args.dataset
    dataset = get_dataset(dataset_name)

    os.makedirs("./dfad_result", exist_ok=True)
    os.makedirs("./student_model", exist_ok=True)

    # 10-fold
    for fold_number in range(9, 11):
        train_loader, test_loader, train_set, test_set = load_dataloader(dataset_name, dataset, 32, fold_number)
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
        #weights = tlx.files.load_npz(path=teacher_folder, name=f_name)
        

        '''data = np.load(file_path, allow_pickle=True)
        weights = [data[str(data.files[i])].T for i in range(len(data.files)) if "weights" in str(data.files[i]) or "biases" in str(data.files[i])]
        print(data.files)
        print(len(data.files))

        s1 = [t.shape for t in weights]
        s2 = [tuple(t.shape) for t in teacher.trainable_weights]
        print(s1)
        print("--------")
        print(s2)'''

        teacher.load_weights(file_path, format="npz_dict", skip=True)
        print("success")

        test_acc = test(teacher, test_loader)
        print(test_acc)

        # initialize student
        if args.student == "gcn":
            student = GCNModel(
                feature_dim=train_set[0].x.shape[1],
                hidden_dim=teacher_info['hidden_units'],
                num_classes=dataset.num_classes,
                num_layers=teacher_info['num_layers']
            )
        elif args.student == 'gin':
            student = GINModel(
                in_channels=train_set[0].x.shape[1],
                hidden_channels=teacher_info['hidden_units'],
                out_channels=dataset.num_classes,
                num_layers=teacher_info['num_layers'],
                name="GIN"
            )
        elif args.student == 'gat':
            student = GATModel(
                featurn_dim=train_set[0].x.shape[1],
                hidden_dim=teacher_info['hidden_units'],
                num_classes=dataset.num_classes,
                num_layers=teacher_info['num_layers']
            )
        elif args.student == 'graphsage':
            student = GraphSAGE_Full_Model(
                in_feats=train_set[0].x.shape[1],
                n_hidden=teacher_info['hidden_units'],
                n_classes=dataset.num_classes,
                n_layers=teacher_info['num_layers']
            )
        else:
            raise NameError("Incorrect model name.")

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
