import itertools
import os.path as osp

import numpy as np
import torch
from jetnet.datasets import QuarkGluon
from torch_geometric.data import Data, Dataset

# electron
PDG_CLASSES = ["electron", "muon", "photon", "charged_hadron", "neutral_hadron"]
N_JETS_PER_FILE = 100_000


def pdg_map(pdg_id):
    if abs(pdg_id) == 11:
        return 0
    if abs(pdg_id) == 13:
        return 1
    if abs(pdg_id) == 22:
        return 2
    if abs(pdg_id) == 211:
        return 3
    if abs(pdg_id) == 2212:
        return 3
    if abs(pdg_id) == 321:
        return 3
    if abs(pdg_id) == 130:
        return 4
    if abs(pdg_id) == 2112:
        return 4
    return None


class JetNetGraph(Dataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, max_jets=None, n_files=20, n_jets_merge=1_000
    ):
        self.raw_data = None
        self.max_jets = max_jets
        self.n_files = n_files
        self.n_jets_merge = n_jets_merge
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "QG_jets_withbc_0.npz",
            "QG_jets_withbc_1.npz",
            "QG_jets_withbc_2.npz",
            "QG_jets_withbc_3.npz",
            "QG_jets_withbc_3.npz",
            "QG_jets_withbc_4.npz",
            "QG_jets_withbc_5.npz",
            "QG_jets_withbc_6.npz",
            "QG_jets_withbc_7.npz",
            "QG_jets_withbc_8.npz",
            "QG_jets_withbc_9.npz",
            "QG_jets_withbc_10.npz",
            "QG_jets_withbc_11.npz",
            "QG_jets_withbc_12.npz",
            "QG_jets_withbc_13.npz",
            "QG_jets_withbc_14.npz",
            "QG_jets_withbc_15.npz",
            "QG_jets_withbc_16.npz",
            "QG_jets_withbc_17.npz",
            "QG_jets_withbc_18.npz",
            "QG_jets_withbc_19.npz",
        ][: self.n_files]

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        if self.max_jets:
            n_files = int(self.max_jets / self.n_jets_merge)
        else:
            n_files = self.n_files * N_JETS_PER_FILE

        return [f"qg_graph_{i}.pt" for i in range(0, n_files)]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"qg_graph_{idx}.pt"))
        return data

    def download(self):
        # Download to `self.raw_dir`.
        self.raw_data = QuarkGluon(jet_type="all", file_list=self.raw_file_names)

    def transform_labels(self, y):
        return torch.tensor([pdg_map(pdg_id) for pdg_id in list(y.numpy())], dtype=torch.long)

    def process(self):
        # Read data into huge `Data` list.
        if self.raw_data is None:
            self.raw_data = QuarkGluon(jet_type="all", file_list=self.raw_file_names)

        for i, (x, _) in enumerate(self.raw_data):
            if i % self.n_jets_merge == 0:
                data_list = []
            if self.max_jets is not None and i > self.max_jets:
                break
            # mask away particles that are zero-padded
            mask = torch.logical_and(x[:, 0] != 0, x[:, 1] != 0)
            mask = torch.logical_and(mask, x[:, 2] != 0)
            mask = torch.logical_and(mask, x[:, 3] != 0)
            n_particles = len(x[mask])
            pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles), range(n_particles)) if m != n])
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            y = x[mask][:, 3].to(torch.int32)
            y = self.transform_labels(y)
            data = Data(x=x[mask][:, :3], edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # append to data list
            data_list.append(data)

            if i % self.n_events_merge == self.n_events_merge - 1:
                data_list = sum(data_list, [])
                torch.save(data_list, osp.join(self.processed_dir, f"qg_graph_{i}.pt"))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
