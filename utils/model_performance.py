import torch
from sklearn.metrics import accuracy_score


def model_train(args, model, train_loader, criterion, optimizer, lr_scheduler, epoch, device):
    # Define variables.
    train_output = []
    train_pred = []
    train_label = []
    num_steps = len(train_loader)
    # Start training.
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Calculate loss.
        loss = criterion(output, data.y)
        # Backward propagation.
        loss.backward()
        optimizer.step()
        if args.is_lr_scheduler:
            if args.lr_scheduler.lower() in ['cosine', 'linear', 'step', 'multistep']:
                lr_scheduler.step_update(epoch * num_steps + i)
            elif args.lr_scheduler.lower() in ['cycle']:
                lr_scheduler.step()
        # Output.
        prediction = torch.argmax(output, dim=1)
        out = output.detach()
        pred = prediction.detach()
        label = data.y.detach()
        # Statistics.
        train_output.append(out)
        train_pred.append(pred)
        train_label.append(label)
    train_loss = criterion(torch.vstack(train_output), torch.hstack(train_label)).item()
    train_acc = accuracy_score(torch.hstack(train_label).cpu(), torch.hstack(train_pred).cpu())

    return train_loss, train_acc


def model_test(model, test_loader, device):
    # Define variables.
    test_pred = []
    test_label = []
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            # Output.
            prediction = torch.argmax(output, dim=1)
            pred = prediction.detach()
            label = data.y.detach()
            # Statistics.
            test_pred.append(pred)
            test_label.append(label)
        test_pred = torch.hstack(test_pred).cpu().numpy()
        test_label = torch.hstack(test_label).cpu().numpy()
        # Score.
        test_acc = round(accuracy_score(test_label, test_pred), 4)

    return test_acc
