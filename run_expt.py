from args import args
from train import run_experiment,save_features_best_model
from utils import log_results, generate_random_seeds
import time


def run(args):
    seeds = generate_random_seeds(args.n_seeds)


    start_time = time.time()
    end_time = start_time
    
    for i, seed in enumerate(seeds):
   
        print(f"Running experiment for seed {seed}. Args:")
        print(args)
        seed_start_time  = time.time()
        args.seed = seed
        best_model, metrics, feats = run_experiment(args)
        # save metrics        
        for split in args.dataset_parameters.splits:
            log_results(args, metrics[split], split)
        
        save_features_best_model(args, best_model)

        seed_end_time  = time.time()
        # Calculate the elapsed time
        seed_elapsed_time = seed_end_time - seed_start_time
        print(f"[{i+1}] Elapsed time for seed {seed}: {seed_elapsed_time:.1f} seconds")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed total time {seed}: {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    run(args)