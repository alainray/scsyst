from collections import defaultdict
import torch
import torch.nn as nn
from utils import AverageMeter, add_new_metrics, pretty_print_metrics, set_deterministic, save_features
from models import get_model
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import create_dataloaders
from scsyst import TRANSFORMATIONS
def run_class_epoch(args, model, data_loader, optimizer = None, train=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    all_feats = []
    all_reps = []
    with torch.set_grad_enabled(train):
        for i, (main_x, _, main_y, _ , shape_y, color_y) in enumerate(data_loader):
            
            # Send everything to GPU
            main_x = main_x.cuda()
            main_y = main_y.cuda()
            shape_y = shape_y.cuda()
            color_y = color_y.cuda()
            #print("main",main_x.shape, shape_y.shape)
            # Fix labels depending on training method
            y = shape_y if args.main_task=="shape" else color_y
            y = y.view(-1, 1)
            output, feats, reps = model(main_x)
            output = output.squeeze()
            # Calculate metrics
            preds = output.argmax(dim=1)
            #print("preds",preds.shape)
            correct = (preds == y.squeeze()).float()
            #print("correct",correct.shape)
            acc = correct.view(-1, 1)
            #print("acc", acc.shape)
            n_samples = acc.shape[0]
            acc = acc.sum(dim=0)
            #print(acc, n_samples)
            acc = acc/n_samples
            #print(acc)
            # handle per task losses
            print(output.shape, y.shape)
            print(train)
            loss = criterion(output, y.squeeze())        # main task
            #loss = loss.mean()
            all_feats.append(feats.detach().cpu())
            all_reps.append(reps.detach().cpu())
            # Backward pass and optimization
            if train:                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    metrics = { 'loss': loss.detach().cpu().item()}
    for i, a in enumerate(acc):
        metrics[f'acc_{i}'] = 100*a.detach().cpu().item()

    return  model, optimizer, metrics, {'feats': torch.stack(all_feats).squeeze(), 'reps': torch.stack(all_reps).squeeze()}


def run_epoch(args, model, data_loader, optimizer = None, train=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    criterion = nn.MSELoss(reduction="none")
    all_feats = []
    all_reps = []
    with torch.set_grad_enabled(train):
        for i, (main_x, aux_x, main_y, aux_y, *_) in enumerate(data_loader):
            
            # Send everything to GPU
            main_x = main_x.cuda()
            aux_x = aux_x.cuda()
            main_y = main_y.cuda()
            aux_y = aux_y.cuda()

            # Fix labels depending on training method
            y = main_y
            if args.train_method in ["aux_tasks", "tasks", "super_reps"]:
                y = torch.cat((main_y.unsqueeze(1), aux_y), dim = 1)
            y = y.view(-1, 1)
            output, feats, reps = model(main_x)
            output = output.view(-1,1)
            #print("output", output.shape, y.shape)

            # Calculate metrics
            # handle per task accuracy
            preds = output.round()
            correct = (preds == y).float()
            acc = correct.view(-1, args.n_tasks)
            n_samples = acc.shape[0]
            acc = acc.sum(dim=0)
            #print(acc, n_samples)
            acc = acc/n_samples
            # handle per task losses
            loss = criterion(output, y)        # main task
            loss = loss.view(args.n_tasks,-1)
            loss = loss.mean(dim=1)
            loss = loss * args.task_importance # adjust loss per task if required
            loss = loss.mean()
            
            # add representation loss if required
            if args.train_method == "super_reps":
                # add representation losses
                _, feats_y, _ = model(aux_x)      # get target reps
                feats_y = feats_y.view(-1,1)
                reps = reps[:,1:,:].reshape(-1,1) # use predicted losses
                rep_loss = criterion(reps, feats_y).view(args.n_tasks-1, -1).mean(dim=1)
                loss += rep_loss.mean()
            all_feats.append(feats.detach().cpu())
            all_reps.append(reps.detach().cpu())
            # Backward pass and optimization
            if train:                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             

    metrics = { 'loss': loss.detach().cpu().item()}
    for i, a in enumerate(acc):
        metrics[f'acc_{TRANSFORMATIONS[i]}'] = 100*a.detach().cpu().item()

    return  model, optimizer, metrics, {'feats': torch.stack(all_feats).squeeze(), 'reps': torch.stack(all_reps).squeeze()}

epoch_method = {'shape': run_class_epoch,
                'color': run_class_epoch,
                'sum': run_epoch}
def save_features_best_model(args, model):
    epoch_method = {'shape': run_class_epoch,
                    'color': run_class_epoch,
                    'sum': run_epoch}
    loaders = create_dataloaders(args)
    for split in args.dataset_parameters.splits:
        _, _, epoch_metrics, features = epoch_method[args.main_task](args,model, loaders[split], None,
                                                train=False)
        save_features(args, features, split)

def run_experiment(args): # runs experiment based on args, returns information to be logged and best model
    print_every=args.print_every
    best_acc = -10.0
    best_epoch = 0
    set_deterministic(seed=args.seed)
    loaders = create_dataloaders(args)
    model = get_model(args).cuda()
    best_model = model
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=20,factor=0.5)
    full_metrics = {split: defaultdict(list) for split in args.dataset_parameters.splits}
    current_metrics = {split: dict() for split in args.dataset_parameters.splits}
    for epoch in range(1,args.epochs+1):
        # Train first
        model, optimizer, _, _ = epoch_method[args.main_task](args,model, loaders['train'], optimizer,
                                                        train=True)
        # Evaluate on all splits

        for split in args.dataset_parameters.splits:
            _, _, epoch_metrics, features = epoch_method[args.main_task](args,model, loaders[split], optimizer,
                                                        train=False)
            epoch_metrics['epoch'] = epoch
            current_metrics[split] = epoch_metrics
            full_metrics[split] = add_new_metrics(full_metrics[split], epoch_metrics)
        
        test_acc = current_metrics['test_out_dist']['acc_main'] 
        train_loss = current_metrics['train']['loss'] 
        scheduler.step(train_loss)
        if test_acc> best_acc:
            best_model = model
            best_epoch = epoch
            best_acc = test_acc

        # Print results

        if epoch % print_every == 0:
            pretty_print_metrics(current_metrics)
            print(f"Best model achieved! With Acc: {best_acc:.2f}%")
            print(f"Current Learning Rate: {scheduler.get_last_lr()}")
    return best_model, best_epoch, full_metrics, features