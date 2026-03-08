# -*- coding: utf-8 -*-
# @Time: 2024/12/10
# @File: model_performance.py
# @Author: fwb
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from utils.cal_metrics import CalMetrics
from serialization import offset2batch, batch2offset


def model_train(args, model, train_loader, criterion, optimizer, lr_scheduler, epoch, device):
    # Define variables.
    train_output = []
    train_pred = []
    train_label = []
    train_loss = []
    # cal_metrics = CalMetrics(args)
    num_steps = len(train_loader)
    # Start training.
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Calculate loss.
        if args.is_cls:
            loss = criterion(output, data.y)
        else:
            loss = criterion(output, data.y.to(dtype=torch.float32))
        # Backward propagation.
        loss.backward()
        optimizer.step()
        if args.is_lr_scheduler:
            if args.lr_scheduler.lower() in ['cosine', 'linear', 'step', 'multistep']:
                lr_scheduler.step_update(epoch * num_steps + i)
            elif args.lr_scheduler.lower() in ['cycle']:
                lr_scheduler.step()
        if args.is_cls:
            # Output.
            prediction = torch.argmax(output, dim=1)
            out = output.detach()
            pred = prediction.detach()
            label = data.y.detach()
            # Statistics.
            train_output.append(out)
            train_pred.append(pred)
            train_label.append(label)
        else:
            train_loss.append(loss.item())
    if args.is_cls:
        train_loss = criterion(torch.vstack(train_output), torch.hstack(train_label)).item()
        train_acc = accuracy_score(torch.hstack(train_label).cpu(), torch.hstack(train_pred).cpu())
        return train_loss, train_acc
    else:
        train_loss = round(np.mean(train_loss), 8)
        return train_loss


def model_val(args, model, val_loader, device):
    # Define variables.
    val_pred = []
    val_label = []
    cal_metrics = None
    if not args.is_cls:
        cal_metrics = CalMetrics(args)
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            output = model(data)
            if args.is_cls:
                # Output.
                prediction = torch.argmax(output, dim=1)
                pred = prediction.detach()
                label = data.y.detach()
                # Statistics.
                val_pred.append(pred)
                val_label.append(label)
            else:
                cal_metrics.matches[str(i)] = {}
                cal_metrics.matches[str(i)]['seg_pred'] = output
                cal_metrics.matches[str(i)]['seg_gt'] = data.y
                if args.roc:
                    data.batch = offset2batch(data.ptr) - 1
                    ev_locs = torch.cat([data.batch.reshape(-1, 1), data.coord], dim=1)
                    cal_metrics.roc_update(ev_locs[:, 3].cpu(), output.cpu(),
                                           data.idx.cpu(), data.y.cpu(), ev_locs.cpu())
        if args.is_cls:
            val_pred = torch.hstack(val_pred).cpu().numpy()
            val_label = torch.hstack(val_label).cpu().numpy()
            # Score.
            test_acc = round(accuracy_score(val_label, val_pred), 4)
            return test_acc
        else:
            iou = cal_metrics.evaluate_semantic_segmentation_iou()
            iou = round(iou.item(), 4)
            return iou


def model_test(args, model, test_loader, device):
    # Define variables.
    test_pred = []
    test_label = []
    cal_metrics = None
    if not args.is_cls:
        cal_metrics = CalMetrics(args)
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            if args.is_cls:
                # Output.
                prediction = torch.argmax(output, dim=1)
                pred = prediction.detach()
                label = data.y.detach()
                # Statistics.
                test_pred.append(pred)
                test_label.append(label)
            else:
                cal_metrics.matches[str(i)] = {}
                cal_metrics.matches[str(i)]['seg_pred'] = output
                cal_metrics.matches[str(i)]['seg_gt'] = data.y
                if args.roc:
                    data.batch = offset2batch(data.ptr) - 1
                    ev_locs = torch.cat([data.batch.reshape(-1, 1), data.coord], dim=1)
                    cal_metrics.roc_update(ev_locs[:, 3].cpu(), output.cpu(),
                                           data.idx.cpu(), data.y.cpu(), ev_locs.cpu())
        if args.is_cls:
            test_pred = torch.hstack(test_pred).cpu().numpy()
            test_label = torch.hstack(test_label).cpu().numpy()
            # Score.
            test_acc = round(accuracy_score(test_label, test_pred), 4)
            return test_acc
        else:
            iou = cal_metrics.evaluate_semantic_segmentation_iou()
            seg_acc = cal_metrics.evaluate_semantic_segmentation_accuracy()
            if args.roc:
                pd, fa = cal_metrics.cal_roc()
            iou = round(iou.item(), 4)
            seg_acc = round(seg_acc.item(), 4)
            pd = round(pd, 4)
            fa = round(fa * 1e4, 4)
            return iou, seg_acc, pd, fa
