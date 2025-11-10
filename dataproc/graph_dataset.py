# -*- coding: utf-8 -*-
# @Time: 2025/3/4
# @File: graph_dataset.py
# @Author: fwb
import os.path as osp
import glob2
import h5py
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


class GraphDataset:
    def __init__(self, args, save_graph_dir, mode: str):
        self.args = args
        self.save_graph_dir = save_graph_dir
        self.save_mode_graph_dir = osp.join(save_graph_dir, mode)
        self.graph_path_list = glob2.glob(osp.join(self.save_mode_graph_dir, '*.hdf5'))

    def __len__(self):
        return len(self.graph_path_list)

    def __getitem__(self, item):
        graph_data_dict = {}
        with h5py.File(self.graph_path_list[item], 'r') as f:
            for key in f.keys():
                value = f[key][()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')  # decode
                graph_data_dict[key] = value
        feat = torch.from_numpy(np.array(graph_data_dict['feat']).astype(np.float32))
        coord = torch.from_numpy(np.array(graph_data_dict['coord']).astype(np.float32))
        label = torch.from_numpy(np.array(graph_data_dict['label']).astype(np.int64))
        graph_data = Data(
            x=feat,
            pos=coord,
            y=label
        )
        if self.args.transform:
            transform = T.Compose([
                T.RandomScale([0.95, 0.99]),
                T.RandomJitter(0.001)
            ])
            graph_data = transform(graph_data)
        return graph_data
