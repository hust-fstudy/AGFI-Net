import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path as osp
import random
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from dataproc.create_graph_dataset import batch_create
from dataproc.proc_utils import remove_files_in_dir
from dataproc.graph_dataset import GraphDataset
from model.net import Net
from opt.criterion import build_criterion
from opt.optimizer import build_optimizer
from opt.lr_scheduler import build_scheduler
from utils.parse_yaml import parse_yaml
from utils.model_performance import model_train, model_test
from utils.save_results import SaveResults


def initial_model(args):
    model = Net(
        in_chs=args.in_chs,
        grid_size=tuple(args.grid_size),
        num_classes=args.num_classes,
        order=tuple(args.order),
        stride=tuple(args.stride),
        enc_depths=tuple(args.enc_depths),
        enc_chs=tuple(args.enc_chs),
        enc_heads=tuple(args.enc_heads),
        enc_patches=tuple(args.enc_patches),
        mlp_ratio=args.mlp_ratio,
        shuffle_orders=args.shuffle_orders,
        enable_rpe=args.enable_rpe,
        enable_flash=args.enable_flash,
        upcast_attention=args.upcast_attention,
        upcast_softmax=args.upcast_softmax,
    )
    return model


class RecognitionMain:
    def __init__(self, cfgs_dir: str, dataset_name: str):
        # Parse configs info.
        self.dataset_name = dataset_name.lower()
        com_cfgs_path = osp.join(cfgs_dir, 'com_params' + '.yaml')
        self.com_cfgs = parse_yaml(com_cfgs_path)
        dataset_cfgs_path = osp.join(cfgs_dir, self.dataset_name + '.yaml')
        self.args = parse_yaml(dataset_cfgs_path)
        self.random_seed = self.com_cfgs.random_seed
        # Load the corresponding dataset.
        print(f"The current dataset used is {dataset_name}")
        self.data_root_dir = self.com_cfgs.data_root_dir
        self.data_path = osp.join(self.data_root_dir, self.dataset_name, 'raw')
        # Graph data storage path.
        self.save_graph_dir = osp.join(self.data_root_dir, 'to_graph', dataset_name,
                                       f'be{self.args.base_events/1e+4}w'
                                       f'_ek{self.args.every_k_points}'
                                       f'_ne{self.args.nb_neighbors}'
                                       f'_sr{self.args.std_ratio}'
                                       f'_ic{self.args.in_chs}')
        # Fixed batch creation graph dataset.
        if self.args.is_remove_graph:
            remove_files_in_dir(self.save_graph_dir)
            batch_create(self.args, self.data_path, self.save_graph_dir, seed=self.random_seed)

    def run(self):
        # Set the random seed and initialize it.
        print(f"Random seed ID is: {self.random_seed}")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # Load graph dataset.
        train_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'train')
        test_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'test')
        print(f"train len: {len(train_graph_dataset)}, test len: {len(test_graph_dataset)}")
        train_loader = DataLoader(train_graph_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=self.args.shuffle)
        test_loader = DataLoader(test_graph_dataset,
                                 batch_size=self.args.batch_size)
        # Initialization Model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initial_model(self.args)
        model.to(device)
        criterion = build_criterion(self.args)
        optimizer = build_optimizer(self.args, model)
        lr_scheduler = None
        if self.args.is_lr_scheduler:
            lr_scheduler = build_scheduler(self.args, optimizer, len(train_loader))
        # Model training.
        SR = SaveResults(self.args, self.random_seed)
        keys = ['train_loss', 'train_acc', 'test_acc', 'best_acc', 'test_epoch_acc']
        results_dict = dict((key, []) for key in keys)
        for epoch in range(self.args.epochs):
            with tqdm(desc=f"epoch {epoch + 1}/{self.args.epochs}"):
                train_temp_loss, train_temp_acc = model_train(self.args, model, train_loader, criterion,
                                                              optimizer, lr_scheduler, epoch, device)
                results_dict['train_loss'].append(train_temp_loss)
                results_dict['train_acc'].append(train_temp_acc)
                print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss}, acc: {train_temp_acc})")
                test_temp_acc = model_test(model, test_loader, device)
                results_dict['test_epoch_acc'].append(test_temp_acc)
                best_acc = max(results_dict['test_epoch_acc'])
                results_dict['best_acc'] = best_acc
                best_epoch = results_dict['test_epoch_acc'].index(best_acc) + 1
                results_dict['best_epoch'] = best_epoch
                print(f"test info for epoch {epoch + 1}: (acc: {test_temp_acc},"
                      f" best epoch: {best_epoch}, best acc: {best_acc})")
                # Save results.
                SR.save_results(results_dict)


if __name__ == '__main__':
    cfgs_dir = r'./configs'
    dataset_dict = {
        1: 'DVSGesture',
        2: 'PAF',
        3: 'THU',
        4: 'KTH',
        5: 'Bully',
        6: 'Daily'
    }
    RM = RecognitionMain(cfgs_dir=cfgs_dir, dataset_name=dataset_dict[1])
    RM.run()
