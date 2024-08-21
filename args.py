from easydict import EasyDict as edict
import torch
args = edict()



def update_args(args):
    args.n_tasks = 1 if args.train_method in ["erm", "augment"] else args.n_tasks
    args.n_heads = args.n_tasks if args.train_method in ["tasks"] else 1
    args.task_importance = torch.ones(args.n_tasks).float().cuda() # all tasks have the same importance
    return args

# Presentation
args.print_every = 1
args.save_best = True
args.log_exp_results = True

# Model

args.hidden_dim = 100 # Dimension of final layer of feature extractor.
args.feature_extractor = "cnn" # cnn / resnet

# Training
args.epochs = 20000
args.filters = 1024 # powers of 2: (16, 32, 64, 128, 256, 512, 1024)
args.n_seeds = 1
args.train_method = "tasks" # erm/augmented/tasks/super_reps/aux_tasks
args.n_tasks = 10 # 1 main + 9 auxiliary
args = update_args(args)
args.main_task = "sum"
# "erm"   --> standard MSE loss for predicting main task
# "reps"  --> representation loss, 
# "tasks" --> loss associated to each auxiliary task
args.losses = ["erm", "reps", "tasks"] 

# Dataset 
args.root_path = "../datasets"
args.dataset = "scsyst" 
args.dataset_parameters = {'height': 3,
                           'width': 3,
                           'n_shapes': 36, 
                           'n_colors': 12,
                           'color_splits': 4,
                           'num_repeats': 5,
                           'splits': ['train','test_in_dist', 'test_out_dist'],
                           'batch_size': 5000}

args.tasks = ['rotation', 'flipY', 'flipZ']

# Optimizer
args.optimizer = "adamw" # "sgd", "adam", "adamw"
args.lr = 0.001
args.weight_decay = 0.1 