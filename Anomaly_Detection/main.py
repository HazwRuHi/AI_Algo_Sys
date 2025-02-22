import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GATConv, GCNConv, SAGEConv
import numpy as np
from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection with GIN')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., "cuda:0" or "cpu")')
    return parser.parse_args()


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, batchnorm=False):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, mlp_layers=1, dropout=0.5, train_eps=True):
        super(GIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.train_eps = train_eps

        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(GINConv(MLP(in_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))
        else:
            self.convs.append(GINConv(MLP(in_channels, hidden_channels, hidden_channels, mlp_layers, dropout), train_eps=train_eps))
            for layer in range(self.n_layers - 2):
                self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, mlp_layers, dropout), train_eps=train_eps))
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))

    def forward(self, x, adj):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, adj)
            x = F.relu(x)
        x = self.convs[-1](x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, n_heads=1, dropout=0.5):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=n_heads, dropout=dropout))
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_channels * n_heads, hidden_channels, heads=n_heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * n_heads, out_channels, heads=n_heads, dropout=dropout, concat=False))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, dropout=0.5, batchnorm=False):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.batchnorm = batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batchnorm else None

        if n_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                if batchnorm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, dropout=0.5, batchnorm=False):
        super(GraphSAGE, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.dropout = dropout
        self.batchnorm = batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batchnorm else None

        if n_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if batchnorm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


def tes(model, data, split_idx, evaluator):
    with torch.no_grad():
        model.eval()
        losses, eval_results = dict(), dict()
        for key in ['train', 'valid']:
            node_id = split_idx[key]
            out = model(data.x, data.adj_t)
            y_pred = out.exp()  # (N, num_classes)
            losses[key] = F.cross_entropy(out[node_id], data.y[node_id]).item()
            eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
    return eval_results, losses, y_pred


def predict(data):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.adj_t)
        y_pred = out.exp()  # (N, num_classes)
    return y_pred

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    # 数据保存路径
    path = './datasets/632d74d4e2843a53167ee9a1-momodel/'
    # 模型保存路径
    save_dir = './results/'
    # 数据集名称
    dataset_name = 'DGraph'

    # 加载数据集并转换为稀疏张量
    dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor()).to(device)

    # 获取数据集的类别数量
    nlabels = dataset.num_classes
    if dataset_name == 'DGraph':
        nlabels = 2  # 本实验中仅需预测类0和类1

    # 获取数据集中的第一个图数据
    data = dataset[0]
    # 将有向图转化为无向图
    data.adj_t = data.adj_t.to_symmetric()

    # 数据标准化处理
    if dataset_name == 'DGraph':
        x = data.x
        x = (x - x.mean(0)) / x.std(0)
        data.x = x

    # 如果标签维度为2，则压缩为1维
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    # 划分训练集、验证集和测试集
    split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}
    train_idx = split_idx['train']

    # 准备结果保存文件夹
    result_dir = prepare_folder(dataset_name, 'mlp')

    # 打印数据特征和标签的形状
    print(data)
    print(data.x.shape)  # feature
    print(data.y.shape)  # label

    # 训练轮数
    epochs = 400
    # 日志记录周期
    log_steps = 1

    # 初始化模型
    model = GraphSAGE(in_channels=data.x.size(-1), hidden_channels=64, out_channels=nlabels, n_layers=2, dropout=0.5, batchnorm=False).to(device)
    # model = GAT(in_channels=data.x.size(-1), hidden_channels=32, out_channels=nlabels, n_layers=2, n_heads=1, dropout=0.5).to(device)
    # model = GCN(in_channels=data.x.size(-1), hidden_channels=64, out_channels=nlabels, n_layers=2, dropout=0.5, batchnorm=False).to(device)
    # model = GIN(in_channels=data.x.size(-1), hidden_channels=64, out_channels=nlabels, n_layers=2, mlp_layers=1, dropout=0.1).to(device)

    # 评估指标
    eval_metric = 'auc'
    evaluator = Evaluator(eval_metric)

    # 打印模型总参数量
    print(sum(p.numel() for p in model.parameters()))

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # 训练和验证模型
    best_valid = 0
    min_valid_loss = 1e8
    train_loss_record = []
    train_eval_record = []
    valid_loss_record = []
    valid_eval_record = []
    for epoch in range(1, epochs + 1):
        loss = train(model, data, train_idx, optimizer)
        eval_results, losses, out = tes(model, data, split_idx, evaluator)
        train_eval, valid_eval = eval_results['train'], eval_results['valid']
        train_loss, valid_loss = losses['train'], losses['valid']

        train_loss_record.append(train_loss)
        train_eval_record.append(train_eval)
        valid_loss_record.append(valid_loss)
        valid_eval_record.append(valid_eval)

        # 保存验证集上表现最好的模型
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

        # 打印日志
        if epoch % log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_eval:.3f}, '  # 我们将AUC值乘上100，使其在0-100的区间内
                  f'Valid: {100 * valid_eval:.3f} ')

    # 载入验证集上表现最好的模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pt')))
    out = model(data.x, data.adj_t)
    torch.save(out, os.path.join(save_dir, 'out.pt'))

    # 预测并打印结果
    dic = {0: "正常用户", 1: "欺诈用户"}
    out = torch.load(os.path.join(save_dir, 'out.pt'))
    for node_idx in [0, 1]:
        y_pred = out[node_idx]
        print(y_pred)
        print(f'节点 {node_idx} 预测对应的标签为:{torch.argmax(y_pred)}, 为{dic[torch.argmax(y_pred).item()]}')

    import matplotlib.pyplot as plt

    # Plot training and validation loss and evaluation metrics separately
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_record, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot validation loss
    plt.subplot(2, 2, 2)
    plt.plot(valid_loss_record, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    # Plot training evaluation metric
    plt.subplot(2, 2, 3)
    plt.plot(train_eval_record, label='Train Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Metric')
    plt.title('Training Evaluation Metric')
    plt.legend()

    # Plot validation evaluation metric
    plt.subplot(2, 2, 4)
    plt.plot(valid_eval_record, label='Valid Eval')
    plt.xlabel('Epochs')
    plt.ylabel('Evaluation Metric')
    plt.title('Validation Evaluation Metric')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('loss_eval_separate.png'), dpi=300)
    plt.show()
