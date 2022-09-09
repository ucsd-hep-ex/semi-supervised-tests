import logging
import os.path as osp
from glob import glob

import torch
from jetnet.datasets import QuarkGluon
from torch_geometric.data import Data, Dataset

logging.basicConfig(level=logging.INFO)

# electron
PDG_CLASSES = ["electron", "muon", "photon", "charged_hadron", "neutral_hadron"]
FEATURES = ["pt", "eta", "phi"]
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
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_jets=None,
        file_start=0,
        file_stop=1,
        n_jets_merge=1_000,
    ):
        if file_stop < file_start:
            raise RuntimeError(f"Expect file_start={file_start} <= file_stop={file_stop}")
        elif file_stop == file_start:
            file_stop = file_start + 1  # for easier indexing

        self.file_start = file_start
        self.file_stop = file_stop
        self.raw_data = None
        self.max_jets = max_jets
        self.n_files = file_stop - file_start
        self.n_jets_merge = n_jets_merge
        super().__init__(root, transform, pre_transform, pre_filter)

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
        ][self.file_start : self.file_stop]

    @property
    def processed_file_names(self):
        """
        Returns a list of all the files in the processed files directory
        """
        proc_list = glob(osp.join(self.processed_dir, "qg_graph_*.pt"))
        return_list = list(map(osp.basename, proc_list))
        return return_list

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        p = osp.join(self.processed_dir, self.processed_file_names[idx])
        data = torch.load(p)
        return data

    @property
    def num_features(self):
        return len(FEATURES)

    @property
    def num_classes(self):
        return len(PDG_CLASSES)

    @property
    def num_nodes(self):
        return sum([data.num_nodes for data_list in self for data in data_list])

    def download(self):
        # Download to `self.raw_dir`.
        self.raw_data = QuarkGluon(jet_type="all", file_list=self.raw_file_names, data_dir=self.raw_dir)

    def transform_labels(self, y):
        return torch.tensor([pdg_map(pdg_id) for pdg_id in list(y.numpy())], dtype=torch.long)

    def process(self):
        # Read data into huge `Data` list.
        if self.raw_data is None:
            self.raw_data = QuarkGluon(jet_type="all", file_list=self.raw_file_names)

        data_list = []
        for i, (x, _) in enumerate(self.raw_data):
            if self.max_jets is not None and i >= self.max_jets:
                break
            # mask away particles that are zero-padded
            mask = torch.logical_and(x[:, 0] != 0, x[:, 1] != 0)
            mask = torch.logical_and(mask, x[:, 2] != 0)
            mask = torch.logical_and(mask, x[:, 3] != 0)
            # get all pairs of particles for edges (both directions)
            n_particles = len(x[mask])
            particle_idx = torch.arange(n_particles, dtype=torch.long)
            pairs = torch.cartesian_prod(particle_idx, particle_idx)
            # remove self loops
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            edge_index = pairs.t().contiguous()
            y = x[mask][:, 3].to(torch.int32)
            y = self.transform_labels(y)
            data = Data(x=x[mask][:, :3], edge_index=edge_index, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # append to data list
            data_list.append([data])

            if i % self.n_jets_merge == self.n_jets_merge - 1:
                # sum into big list
                datas = sum(data_list, [])
                logging.info(f"Saving qg_graph_{i}.pt")
                torch.save(datas, osp.join(self.processed_dir, f"qg_graph_{i}.pt"))
                # reset to empty data list
                data_list = []

        # check if there are any leftovers, and save
        if data_list:
            datas = sum(data_list, [])
            torch.save(datas, osp.join(self.processed_dir, f"qg_graph_{i - 1}.pt"))


if __name__ == "__main__":
    root = osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", "data", "tmp")
    dataset = JetNetGraph(root, max_jets=10_000, file_start=0, file_stop=1)
