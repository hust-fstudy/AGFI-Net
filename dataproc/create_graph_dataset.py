# -*- coding: utf-8 -*-
# @Time: 2025/3/4
# @File: create_graph_dataset.py
# @Author: fwb
import os
import os.path as osp
import random
import glob2
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from utils.read_file import ReadFile
from utils.uniform_event import Event
from dataproc.proc_utils import min_max_normalization
from dataproc.proc_utils import create_split, create_path, remove_files_in_dir
from e2g.events_filtering import adaptive
from e2g.cal_graph import CalGraph


def generate(loader):
    for _, _ in enumerate(loader):
        pass


def get_label_dict(args, data_dir):
    if args.is_split:
        train_dir = osp.join(data_dir, 'total')
    else:
        train_dir = osp.join(data_dir, 'train')
    class_list = os.listdir(train_dir)
    label_dict = {class_name: label for label, class_name in enumerate(class_list)}
    return label_dict


def batch_create(args, data_path, save_graph_dir, seed=1):
    label_dict = get_label_dict(args, data_path)
    batch_train_graph_dataset = CreateGraphDataset(args, data_path, save_graph_dir, label_dict, 'train', seed=seed)
    batch_test_graph_dataset = CreateGraphDataset(args, data_path, save_graph_dir, label_dict, 'test', seed=seed)
    batch_train_loader = DataLoader(batch_train_graph_dataset, batch_size=args.gen_batch, num_workers=args.num_workers)
    batch_test_loader = DataLoader(batch_test_graph_dataset, batch_size=args.gen_batch, num_workers=args.num_workers)
    generate(batch_train_loader)
    generate(batch_test_loader)


def create_graph_data(args, events_dict, graph_index_path):
    # Unify the temporal scale of events to the spatial scale.
    x_max = events_dict['x'].max().astype(np.float32)
    y_max = events_dict['y'].max().astype(np.float32)
    t_scale = (x_max + y_max) // 2
    events_dict['t'] = min_max_normalization(events_dict['t'], mx=t_scale)
    # Event filtering.
    events_dict = adaptive(args, events_dict)
    # Calculate graph.
    CG = CalGraph(args, t_scale)
    events_dict = CG(events_dict)
    # Graph data info.
    graph_data_dict = {
        'feat': np.array(events_dict['feat']),
        'coord': np.array(events_dict['coord']),
        'label': np.array(events_dict['label'])
    }
    with h5py.File(graph_index_path, 'w') as f:
        for key, value in graph_data_dict.items():
            if isinstance(value, str):
                value = np.string_(value)
            f.create_dataset(key, data=value)
    return graph_data_dict


class CreateGraphDataset(Dataset):
    def __init__(self, args, data_path, save_graph_dir, label_dict, mode: str, seed=1, train_ratio=0.8):
        super(CreateGraphDataset, self).__init__()
        random.seed(seed)
        self.args = args
        self.data_path = data_path
        self.save_graph_dir = save_graph_dir
        self.label_dict = label_dict
        self.mode = mode
        self.dataset_name = Path(data_path).parent.name
        self.RF = ReadFile()
        if args.is_split:
            remove_files_in_dir(osp.join(data_path, 'train'))
            remove_files_in_dir(osp.join(data_path, 'test'))
            create_split(data_path, train_ratio)
        self.mode_dir = osp.join(data_path, mode)
        self.save_graph_dir = osp.join(save_graph_dir, mode)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_list = os.listdir(self.mode_dir)
        self.sample_path_list = []
        self.sample_label_list = []
        for each_class in tqdm(self.class_list, desc="Read each class"):  # iterate over all classes
            each_class_path = osp.join(self.mode_dir, each_class)
            file_path_list = glob2.glob(osp.join(each_class_path, '*'))
            file_label_list = np.full(len(file_path_list), self.label_dict[each_class])
            self.sample_path_list.extend(file_path_list)
            self.sample_label_list.extend(file_label_list)
        self.save_graph_names = [f"{mode}_data_{i}.hdf5" for i in range(len(self.sample_path_list))]
        create_path(self.save_graph_dir)

    def __len__(self):
        return len(self.sample_path_list)

    def __getitem__(self, index):
        graph_index_path = osp.join(self.save_graph_dir, self.save_graph_names[index])
        sample_path = self.sample_path_list[index]
        if self.dataset_name in ['dvsgesture']:
            [x, y, t, p] = self.RF.npz_file_reader(sample_path)
        elif self.dataset_name in ['thu']:
            [x, y, t, p] = self.RF.npy_file_reader(sample_path)
        elif self.dataset_name in ['paf']:
            [x, y, t, p] = self.RF.aedat_file_reader(sample_path)
        elif self.dataset_name in ['kth']:
            [x, y, t, p] = self.RF.mat_file_reader(sample_path)
        elif self.dataset_name in ['bully']:
            [x, y, t, p] = self.RF.bully_data_reader(sample_path)
        elif self.dataset_name in ['daily']:
            [x, y, t, p] = self.RF.daily_data_reader(sample_path)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        sample_label = np.array(self.sample_label_list[index])
        events_dict = Event(x, y, t, p, sample_label).to_uniform_format()
        graph_data = create_graph_data(self.args, events_dict, graph_index_path)
        graph_data = list(graph_data)
        return graph_data
