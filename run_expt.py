from args import args, update_args
from train import run_experiment,save_features_best_model
from utils import log_results, generate_random_seeds, save_model
import time
import argparse

def run(args):
    seeds = generate_random_seeds(100)
    seed = seeds[args.n_seed] #

    start_time = time.time()
    end_time = start_time
    
   
    print(f"Running experiment for seed {seed}. Args:")
    print(args)
    seed_start_time  = time.time()
    args.seed = seed
    best_model, best_epoch, metrics, feats = run_experiment(args)
    # save metrics        
    for split in args.dataset_parameters.splits:
        log_results(args, metrics[split], split)
    
    save_features_best_model(args, best_model)

    seed_end_time  = time.time()
    # Calculate the elapsed time
    seed_elapsed_time = seed_end_time - seed_start_time
    # save best model
    save_model(args, best_model, best_epoch)
    print(f"Elapsed time for seed {seed}: {seed_elapsed_time:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add an argument
    parser.add_argument(
        "--n_filters", 
        type=int, 
        help="Number of filters in conv layer", default=256
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        help="Number of dimensions of representation", default=100
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        help="weight decay factor", default=0
    )

    parser.add_argument(
        "--n_seed", 
        type=int, 
        help="index for seed to use", default=0
    )

    parser.add_argument(
        "--train_method", 
        type=str, 
        help="training method (erm/tasks/super_reps/aux_tasks)", default="erm"
    )  


    # Parse the arguments
    input_args = parser.parse_args()
    args.filters = input_args.n_filters
    args.hidden_dim = input_args.hidden_dim 
    args.weight_decay = input_args.weight_decay
    args.n_seed = input_args.n_seed
    args.train_method = input_args.train_method

    args = update_args(args)
    run(args)