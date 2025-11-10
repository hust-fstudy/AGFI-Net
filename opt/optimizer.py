import torch.optim as optim


def build_optimizer(args, model):
    """
    Build optimizer.
    """
    parameters = model.parameters()
    opt_name = args.opt_name.lower()
    match opt_name:
        case 'adam':
            optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        case 'adamw':
            optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        case 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay, nesterov=True)
        case _:
            optimizer = None
            print(f"The {opt_name} optimizer does not exist!")
    return optimizer
