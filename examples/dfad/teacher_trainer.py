import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TL_BACKEND'] = 'torch'
import sys

import gammagl.data
import numpy as np
from gammagl.datasets import TUDataset
import tensorlayerx as tlx
from gammagl.loader import DataLoader
from tensorlayerx.model import TrainOneStep, WithLoss
from gammagl.models import GINModel

from datasets import get_dataset

import argparse
import csv


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, label):
        logits = self.backbone_network(data.x, data.edge_index, data.batch)
        loss = self._loss_fn(logits, data.y)
        return loss


def load_dataloader(dataset_name, dataset, batch_size, fold_number):
    train_idx = np.loadtxt('./dataset/{0}/10fold_idx/train_idx-{1}.txt'.format(dataset_name, fold_number), dtype=np.int64)
    test_idx = np.loadtxt('./dataset/{0}/10fold_idx/test_idx-{1}.txt'.format(dataset_name, fold_number), dtype=np.int64)

    print(len(test_idx))
    assert len(train_idx) + len(test_idx) == len(dataset)

    train_set, test_set = dataset[train_idx], dataset[test_idx]

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader, train_set, test_set

def test(net, test_loader, epoch):
    net.set_eval()

    all_preds = []
    all_labels = []

    for batch in test_loader:
        pred = net(batch.x, batch.edge_index, batch.batch).detach()
        all_preds.append(pred)
        all_labels.append(batch.y)

    all_preds = tlx.convert_to_tensor(np.vstack(all_preds))
    all_labels = tlx.convert_to_tensor(np.concatenate(all_labels))
    all_labels = tlx.reshape(all_labels, (-1, 1))

    all_preds = tlx.cast(all_preds, tlx.int32)
    all_labels = tlx.cast(all_labels, tlx.int32)

    acc = tlx.metrics.acc(all_preds, all_labels)
    print("Epoch {0}, Test: acc = {1}".format(epoch, acc))
    return acc

def train(args, gin_net, train_loader, test_loader, fold_number):
    os.makedirs("./result/{0}".format(args.dataset), exist_ok=True)
    os.makedirs("./teacher_model/{0}".format(args.dataset), exist_ok=True)
    f = open("./result/{0}/{0}_{1}.csv".format(args.dataset, fold_number), "w", newline="")
    csv_writer = csv.writer(f)

    optimizer = tlx.optimizers.Adam(lr=args.lr, weight_decay=args.l2_coef)
    train_weights = gin_net.trainable_weights
    loss_fn = SemiSpvzLoss(gin_net, tlx.losses.softmax_cross_entropy_with_logits)
    train_one_step = TrainOneStep(loss_fn, optimizer, train_weights)

    best_acc = 0
    acc_list = []
    for epoch in range(args.n_epochs):
        # training process
        gin_net.set_train()
        for batch in train_loader:
            train_loss = train_one_step(batch, batch.y)

        # test
        acc = test(gin_net, test_loader, epoch)
        acc_list.append(acc)
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  acc: {:.4f}".format(acc))
        if epoch == 0:
            csv_writer.writerow(["epoch", "acc"])
        csv_writer.writerow([epoch, round(acc, 3)])

        if acc > best_acc:
            best_acc = acc
            gin_net.save_weights("./teacher_model/{0}/{0}_{1}.npz".format(args.dataset, fold_number), format="npz_dict")

    f.close()

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=120, help="number of epoch")
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--hidden_units", type=int, default=128, help="dimention of hidden layers")
    parser.add_argument("--l2_coef", type=float, default=5e-4, help="l2 loss coeficient")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset(MUTAG/IMDB-BINARY/REDDIT-BINARY)')

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = get_dataset(dataset_name)

    os.makedirs("./result", exist_ok=True)
    os.makedirs("./teacher_model", exist_ok=True)

    best_acc_list = []
    for fold_number in range(1, 11):
        train_loader, test_loader, train_set, test_set = load_dataloader(dataset_name, dataset, 32, fold_number)

        assert train_set[0].x != None
        
        print(train_set[0].x)
        gin_net = GINModel(
            in_channels = train_set[0].x.shape[1],
            hidden_channels = args.hidden_units,
            out_channels = dataset.num_classes,
            num_layers = args.num_layers,
            name = "GIN"
        )

        best_acc = train(args, gin_net, train_loader, test_loader, fold_number)

        os.makedirs("./result/best-acc/{0}".format(dataset_name), exist_ok=True)
        with open("./result/best-acc/{0}/{0}_{1}_best-acc.txt".format(dataset_name, fold_number), "w") as f:
            f.write(str(best_acc))

        print("best_acc:", best_acc)
        best_acc_list.append(best_acc)


    best_acc_mean = '{:.3f}'.format(np.array(best_acc_list).mean())
    best_acc_std = '{:.3f}'.format(np.array(best_acc_list).std())
    overall_result = {
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "num_layers": args.num_layers,
        "hidden_units": args.hidden_units,
        "l2_coef": args.l2_coef,
        "dataset": args.dataset,
        "best_acc_mean": best_acc_mean,
        "best_acc_std": best_acc_std
    }
    with open("./result/best-acc/{0}/overall_result.txt".format(args.dataset), "w") as f:
        f.write(str(overall_result))




