# -*- coding: utf-8 -*-
# @Time: 2025/6/25
# @File: run_recognition.py
# @Author: fwb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
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
from utils.model_performance import model_train, model_val, model_test
from utils.save_results import SaveResults


def initial_model(args):
    if args.is_cls:
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
            is_cls=args.is_cls,
        )
    else:
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
            dec_depths=tuple(args.dec_depths),
            dec_chs=tuple(args.dec_chs),
            dec_heads=tuple(args.dec_heads),
            dec_patches=tuple(args.dec_patches),
            mlp_ratio=args.mlp_ratio,
            shuffle_orders=args.shuffle_orders,
            enable_rpe=args.enable_rpe,
            enable_flash=args.enable_flash,
            upcast_attention=args.upcast_attention,
            upcast_softmax=args.upcast_softmax,
            is_cls=args.is_cls,
        )
    return model


class TaskMain:
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
                                       f'be{self.args.base_events / 1e+4}w'
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
        if not self.args.is_cls:
            val_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'val')
            print(f"val len: {len(val_graph_dataset)}")
            val_loader = DataLoader(val_graph_dataset,
                                    batch_size=self.args.batch_size)
        else:
            val_loader = None
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
        if self.args.is_cls:
            keys = ['train_loss', 'train_acc',
                    'test_acc', 'best_acc', 'test_epoch_acc']
        else:
            keys = ['train_epoch_loss', 'val_epoch_iou',
                    'test_epoch_iou', 'test_epoch_acc', 'test_epoch_pd', 'test_epoch_fa',
                    'test_best_loss', 'test_best_iou']
        results_dict = dict((key, []) for key in keys)
        model_weights = []
        for epoch in range(self.args.epochs):
            with tqdm(desc=f"epoch {epoch + 1}/{self.args.epochs}"):
                if self.args.is_cls:
                    train_temp_loss, train_temp_acc = model_train(self.args, model, train_loader, criterion,
                                                                  optimizer, lr_scheduler, epoch, device)
                    results_dict['train_loss'].append(train_temp_loss)
                    results_dict['train_acc'].append(train_temp_acc)
                    print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss}, acc: {train_temp_acc})")
                    test_temp_acc = model_test(self.args, model, test_loader, device)
                    results_dict['test_epoch_acc'].append(test_temp_acc)
                    best_acc = max(results_dict['test_epoch_acc'])
                    results_dict['best_acc'] = best_acc
                    best_epoch = results_dict['test_epoch_acc'].index(best_acc) + 1
                    results_dict['best_epoch'] = best_epoch
                    print(f"test info for epoch {epoch + 1}: (acc: {test_temp_acc}, "
                          f"best epoch: {best_epoch}, best acc: {best_acc})")
                else:
                    train_temp_loss = model_train(self.args, model, train_loader, criterion,
                                                  optimizer, lr_scheduler, epoch, device)
                    results_dict['train_epoch_loss'].append(train_temp_loss)
                    model_weights.append(model.state_dict())
                    print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss})")
                    val_temp_iou = model_val(self.args, model, val_loader, device)
                    results_dict['val_epoch_iou'].append(val_temp_iou)
                    test_temp_iou, test_temp_seg_acc, test_temp_pd, test_temp_fa = model_test(self.args, model,
                                                                                              test_loader, device)
                    results_dict['test_epoch_iou'].append(test_temp_iou)
                    results_dict['test_epoch_acc'].append(test_temp_seg_acc)
                    results_dict['test_epoch_pd'].append(test_temp_pd)
                    results_dict['test_epoch_fa'].append(test_temp_fa)
                    best_iou = max(results_dict['test_epoch_iou'])
                    best_seg_acc = max(results_dict['test_epoch_acc'])
                    best_pd = max(results_dict['test_epoch_pd'])
                    best_fa = min(results_dict['test_epoch_fa'])
                    best_epoch = results_dict['test_epoch_iou'].index(best_iou) + 1
                    print(f"test info for epoch {epoch + 1}: (iou: {test_temp_iou}, seg_acc: {test_temp_seg_acc}, "
                          f"pd: {test_temp_pd}, fa: {test_temp_fa}e-4)")
                    print(f"statistics: (best_iou: {best_iou}, best_seg_acc: {best_seg_acc}, "
                          f"best_pd: {best_pd}, best_fa: {best_fa}e-4)")
                    print(f"best iou for epoch {best_epoch}: (best_iou: {best_iou}, "
                          f"seg_acc: {results_dict['test_epoch_acc'][best_epoch-1]}, "
                          f"pd: {results_dict['test_epoch_pd'][best_epoch-1]}, "
                          f"fa: {results_dict['test_epoch_fa'][best_epoch-1]}e-4)")
                torch.cuda.empty_cache()
        if not self.args.is_cls:
            best_loss_epoch = results_dict['train_epoch_loss'].index(min(results_dict['train_epoch_loss'])) + 1
            best_iou_epoch = results_dict['val_epoch_iou'].index(max(results_dict['val_epoch_iou'])) + 1
            best_loss_model_weights = model_weights[best_loss_epoch-1]
            best_iou_model_weights = model_weights[best_iou_epoch-1]
            # Test.
            model.load_state_dict(best_loss_model_weights)
            best_loss_iou, best_loss_seg_acc, best_loss_pd, best_loss_fa = model_test(self.args,
                                                                                      model,
                                                                                      test_loader,
                                                                                      device)
            results_dict['test_best_loss'].append(
                {'best_loss_iou': best_loss_iou,
                 'best_loss_seg_acc': best_loss_seg_acc,
                 'best_loss_pd': best_loss_pd,
                 'best_loss_fa': best_loss_fa}
            )
            print(f"test for best loss in epoch {best_loss_epoch}: (best_loss_iou: {best_loss_iou}, "
                  f"best_loss_seg_acc: {best_loss_seg_acc}, "
                  f"best_loss_pd: {best_loss_pd}, best_loss_fa: {best_loss_fa}e-4)")
            model.load_state_dict(best_iou_model_weights)
            best_iou_iou, best_iou_seg_acc, best_iou_pd, best_iou_fa = model_test(self.args,
                                                                                  model,
                                                                                  test_loader,
                                                                                  device)
            results_dict['test_best_iou'].append(
                {'best_iou_iou': best_iou_iou,
                 'best_iou_seg_acc': best_iou_seg_acc,
                 'best_iou_pd': best_iou_pd,
                 'best_iou_fa': best_iou_fa}
            )
            print(f"test for best iou in epoch {best_iou_epoch}: (best_iou_iou: {best_iou_iou}, "
                  f"best_iou_seg_acc: {best_iou_seg_acc}, "
                  f"best_iou_pd: {best_iou_pd}, best_iou_fa: {best_iou_fa}e-4)")
            # Log.
        #     SR.save_model(best_loss_model_weights, 'best_loss_model')
        #     SR.save_model(best_iou_model_weights, 'best_iou_model')
        # # Save results.
        # SR.save_results(results_dict)


if __name__ == '__main__':
    cfgs_dir = r'./configs'
    dataset_dict = {
        1: 'DVSGesture',
        2: 'PAF',
        3: 'THU',
        4: 'KTH',
        5: 'Bully',
        6: 'Daily',
        7: 'EV-UAV'
    }
    RM = TaskMain(cfgs_dir=cfgs_dir, dataset_name=dataset_dict[1])
    RM.run()
