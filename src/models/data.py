from jetnet.datasets import JetNet
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
import itertools
import numpy as np

class JetNetGraph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.raw_data = None
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['t.pt']

    @property
    def processed_file_names(self):
        return ['t_graph.pt']

    def download(self):
        # Download to `self.raw_dir`.
        self.raw_data = JetNet(jet_type='t', use_num_particles_jet_feature=False, use_mask=False, data_dir=self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        if self.raw_data is None:
            self.raw_data = JetNet(jet_type='t', use_num_particles_jet_feature=False, use_mask=False, data_dir=self.raw_dir)
        data_list = []
        for i, (x, _) in enumerate(self.raw_data):
            n_particles = len(x) # can use mask in the future
            pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles), range(n_particles)) if m != n])
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            data_list.append(Data(x=x, edge_index=edge_index))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    dataset = JetNetGraph('../data/JetNet')
    for data in dataset:
        print(data)
