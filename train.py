from collections import defaultdict
import torch
import torch.nn as nn
from utils import AverageMeter, add_new_metrics, pretty_print_metrics, set_deterministic, save_features
from models import get_model
import torch.optim as optim
from datasets import create_dataloaders


def run_epoch(args, model, data_loader, optimizer = None, train=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    criterion = nn.MSELoss(reduction="none")
    all_feats = []
    all_reps = []
    with torch.set_grad_enabled(train):
        for i, (main_x, aux_x, main_y, aux_y) in enumerate(data_loader):
            
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
        metrics[f'acc_{i}'] = 100*a.detach().cpu().item()

    return  model, optimizer, metrics, {'feats': torch.stack(all_feats).squeeze(), 'reps': torch.stack(all_reps).squeeze()}


'''def run_epoch(args, model, data_loader, optimizer = None, train=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    criterion = nn.MSELoss()
    all_feats = []
    all_reps = []
    with torch.set_grad_enabled(train):
        for i, (images, ys, *_) in enumerate(data_loader):
              # depending on the train method is how we deal with input and labels
              # erm: nothing special
              # augment: nothing special
              # multitask: we need to concatenate all ys from all tasks for the criterion, then 
              # aux_tasks: we need to concatenate all reps, all ys and then apply criterion
              # supervised aux_tasks: same as aux_tasks but calculate loss from aux_inputs
              # Forward pass
              images = images.cuda()
              ys = ys.cuda()
              output, feats, reps = model(images)
              output = output.view(-1, 1) # Doing regression
              preds = output.round()
              correct = (preds == ys).float().sum()
              loss = criterion(output,ys)
              acc = correct/ys.numel()
              # if auxiliary tasks add auxiliary losses
              if args.train_method in ['aux_tasks']:
                  loss += 0.1
                  reps
              all_feats.append(feats.detach().cpu())
              all_reps.append(reps.detach().cpu())
              # Backward pass and optimization
              if train:                 
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
    
    return  model, optimizer, {
                'loss': loss.detach().cpu().item(),
                'acc': 100*acc.cpu().item()
            }, {'feats': torch.stack(all_feats).squeeze(), 'reps': torch.stack(all_reps).squeeze()}
'''
def save_features_best_model(args, model):
    loaders = create_dataloaders(args)
    for split in args.dataset_parameters.splits:
        _, _, epoch_metrics, features = run_epoch(args,model, loaders[split], None,
                                                train=False)
        save_features(args, features, split)

def run_experiment(args, print_every=100): # runs experiment based on args, returns information to be logged and best model

    best_acc = -10.0
    set_deterministic(seed=args.seed)
    loaders = create_dataloaders(args)
    model = get_model(args).cuda()
    best_model = model
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    full_metrics = {split: defaultdict(list) for split in args.dataset_parameters.splits}
    current_metrics = {split: dict() for split in args.dataset_parameters.splits}
    for epoch in range(1,args.epochs+1):
        # Train first
        model, optimizer, _, _ = run_epoch(args,model, loaders['train'], optimizer,
                                                        train=True)
        # Evaluate on all splits

        for split in args.dataset_parameters.splits:
            _, _, epoch_metrics, features = run_epoch(args,model, loaders[split], optimizer,
                                                        train=False)
            epoch_metrics['epoch'] = epoch
            current_metrics[split] = epoch_metrics
            full_metrics[split] = add_new_metrics(full_metrics[split], epoch_metrics)
        
        test_acc = current_metrics['test']['acc_1'] 
        if test_acc> best_acc:
            best_model = model
            best_acc = test_acc
            print(f"Best model achieved! With Acc: {best_acc:.2f}%")
        
        # Print results
        for split in args.dataset_parameters.splits:
            if epoch % print_every == 0:
                pretty_print_metrics(current_metrics)

    return best_model, full_metrics, features