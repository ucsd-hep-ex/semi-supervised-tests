import math
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MaskLabel, TransformerConv
from torch_geometric.utils import index_to_mask

from src.data.jetnet_graph import JetNetGraph

root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "JetNet")
dataset = JetNetGraph(root, max_jets=1_000, n_files=1)  # just use one file, 1k jets for fast testing


class UniMP(torch.nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels, num_layers, heads, dropout=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.label_emb = MaskLabel(self.num_classes, self.in_channels)
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_channels = self.hidden_channels // heads
                concat = True
            else:
                out_channels = self.num_classes
                concat = False
            conv = TransformerConv(in_channels, out_channels, heads, concat=concat, beta=True, dropout=dropout)
            self.convs.append(conv)
            in_channels = self.hidden_channels

            if i < num_layers:
                self.norms.append(torch.nn.LayerNorm(self.hidden_channels))

    def forward(self, x, y, edge_index, label_mask):
        x = self.label_emb(x, y, label_mask)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index)).relu()
        return self.convs[-1](x, edge_index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = dataset.data.to(device)

model = UniMP(in_channels=dataset.num_features, num_classes=5, hidden_channels=64, num_layers=3, heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

tv_frac = 0.10
tv_num = math.ceil(data.num_nodes * tv_frac)
splits = np.cumsum([data.num_nodes - 2 * tv_num, tv_num, tv_num])

train_index = torch.tensor(np.arange(start=0, stop=splits[0]), dtype=torch.long)
val_index = torch.tensor(np.arange(start=splits[0], stop=splits[1]), dtype=torch.long)
test_index = torch.tensor(np.arange(start=splits[1], stop=data.num_nodes), dtype=torch.long)

train_mask = index_to_mask(train_index, size=data.num_nodes)
val_mask = index_to_mask(val_index, size=data.num_nodes)
test_mask = index_to_mask(test_index, size=data.num_nodes)


def train(label_rate=0.75):  # How many labels to use for propagation
    model.train()

    propagation_mask = MaskLabel.ratio_mask(train_mask, ratio=label_rate)
    supervision_mask = train_mask ^ propagation_mask

    optimizer.zero_grad()
    out = model(data.x, data.y, data.edge_index, propagation_mask)
    loss = F.cross_entropy(out[supervision_mask], data.y[supervision_mask])
    loss.backward()
    optimizer.step()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()

    propagation_mask = train_mask
    out = model(data.x, data.y, data.edge_index, propagation_mask)
    pred = out[val_mask].argmax(dim=-1)
    val_acc = int((pred == data.y[val_mask]).sum()) / pred.size(0)

    propagation_mask = train_mask | val_mask
    out = model(data.x, data.y, data.edge_index, propagation_mask)
    pred = out[test_mask].argmax(dim=-1)
    test_acc = int((pred == data.y[test_mask]).sum()) / pred.size(0)

    return val_acc, test_acc


for epoch in range(1, 101):
    loss = train()
    val_acc, test_acc = test()
    print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
