import logging
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import MaskLabel

from src.data.jetnet_graph import JetNetGraph
from src.models.unimp_model import UniMP

logging.basicConfig(level=logging.INFO)


def train(model, loader, optimizer, label_rate=0.85, loss_fcn=F.cross_entropy):
    model.train()

    sum_loss = 0
    sum_true = 0
    sum_all = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        train_mask = torch.ones_like(data.x[:, 0], dtype=torch.bool)
        propagation_mask = MaskLabel.ratio_mask(train_mask, ratio=label_rate)
        supervision_mask = train_mask ^ propagation_mask

        out = model(data.x, data.y, data.edge_index, propagation_mask)
        loss = loss_fcn(out[supervision_mask], data.y[supervision_mask])
        loss.backward()
        sum_loss += float(loss)
        optimizer.step()

        pred = out[supervision_mask].argmax(dim=-1)
        sum_true += int((pred == data.y[supervision_mask]).sum())
        sum_all += pred.size(0)
        logging.info(f"Batch: {i + 1:03d}, Train Loss: {sum_loss:.4f}")

    return float(sum_loss) / (i + 1), float(sum_true) / sum_all


@torch.no_grad()
def test(model, loader, label_rate=0.85):
    model.eval()

    sum_true = 0
    sum_all = 0
    for data in loader:

        test_mask = torch.ones_like(data.x[:, 0], dtype=torch.bool)
        propagation_mask = MaskLabel.ratio_mask(test_mask, ratio=label_rate)
        supervision_mask = test_mask ^ propagation_mask

        out = model(data.x, data.y, data.edge_index, propagation_mask)
        pred = out[supervision_mask].argmax(dim=-1)
        sum_true += int((pred == data.y[supervision_mask]).sum())
        sum_all += pred.size(0)

    return float(sum_true) / sum_all


def collate_fn(items):
    sum_list = sum(items, [])
    return Batch.from_data_list(sum_list)


def main():
    train_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", "train")
    val_root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", "val")
    train_dataset = JetNetGraph(train_root, max_jets=10_000, file_start=0, file_stop=1)
    val_dataset = JetNetGraph(val_root, max_jets=10_000, file_start=1, file_stop=2)
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate_fn
    val_loader = DataListLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    val_loader.collate_fn = collate_fn

    model = UniMP(
        in_channels=train_dataset.num_features,
        num_classes=train_dataset.num_classes,
        hidden_channels=64,
        num_layers=3,
        heads=2,
    ).to(device)

    logging.info("Model summary")
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    for epoch in range(1, 101):
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_acc = test(model, val_loader)
        logging.info(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


if __name__ == "__main__":
    main()
