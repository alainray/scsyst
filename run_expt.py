from args import args
from train import run_experiment,save_features_best_model
from utils import log_results, generate_random_seeds, save_model
import time
import argparse

def run(args):
    seeds = generate_random_seeds(args.n_seeds)


    start_time = time.time()
    end_time = start_time
    
    for i, seed in enumerate(seeds):
   
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
        print(f"[{i+1}] Elapsed time for seed {seed}: {seed_elapsed_time:.1f} seconds")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed total time {seed}: {elapsed_time:.1f} seconds")


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


    # Parse the arguments
    input_args = parser.parse_args()
    args.filters = input_args.n_filters
    args.hidden_dim = input_args.hidden_dim
    run(args)