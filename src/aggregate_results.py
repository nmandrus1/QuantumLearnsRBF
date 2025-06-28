"""
Result Aggregator.

This script reads all the raw trial data for a given experiment,
computes the mean and standard deviation for all metrics across trials,
and generates the final summary plots.
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd

# Set backend for matplotlib before other imports
import matplotlib
matplotlib.use('Agg')

from lib.plotting import plot_results

def aggregate_results(exp_name: str, results_dir: str, plots_dir: str):
    """
    Aggregates results for a given experiment and generates plots.
    """
    exp_results_dir = os.path.join(results_dir, exp_name)
    if not os.path.isdir(exp_results_dir):
        print(f"Error: No results directory found for experiment '{exp_name}'.")
        print(f"Looked in: {exp_results_dir}")
        return

    trial_files = [f for f in os.listdir(exp_results_dir) if f.endswith('.pkl')]
    if not trial_files:
        print(f"No trial files (.pkl) found for experiment '{exp_name}'.")
        return

    print(f"Found {len(trial_files)} trial files for '{exp_name}'. Loading data...")

    # --- Data Loading and Consolidation ---
    all_trials_data = []
    for trial_file in trial_files:
        with open(os.path.join(exp_results_dir, trial_file), 'rb') as f:
            trial_data = pickle.load(f)
            # Convert to a flat structure for pandas DataFrame
            for model_name, model_results in trial_data.items():
                for i, size in enumerate(model_results['train_sizes']):
                    all_trials_data.append({
                        'model': model_name,
                        'train_size': size,
                        'train_accuracy': model_results['train_accuracies'][i],
                        'test_accuracy': model_results['test_accuracies'][i],
                        'training_time': model_results['training_times'][i],
                    })
    
    df = pd.DataFrame(all_trials_data)

    # --- Aggregation (Mean and Std Dev) ---
    aggregated_data = df.groupby(['model', 'train_size']).agg(
        train_accuracies_mean=('train_accuracy', 'mean'),
        train_accuracies_std=('train_accuracy', 'std'),
        test_accuracies_mean=('test_accuracy', 'mean'),
        test_accuracies_std=('test_accuracy', 'std'),
        training_times_mean=('training_time', 'mean'),
        training_times_std=('training_time', 'std')
    ).reset_index()

    # Fill NaNs in std dev for single-trial cases
    aggregated_data.fillna(0, inplace=True)

    # --- Reformat for Plotting --- 
    final_plot_data = {}
    for model in aggregated_data['model'].unique():
        model_df = aggregated_data[aggregated_data['model'] == model]
        final_plot_data[model] = {
            'train_sizes': model_df['train_size'].values,
            'train_accuracies_mean': model_df['train_accuracies_mean'].values,
            'train_accuracies_std': model_df['train_accuracies_std'].values,
            'test_accuracies_mean': model_df['test_accuracies_mean'].values,
            'test_accuracies_std': model_df['test_accuracies_std'].values,
            'training_times_mean': model_df['training_times_mean'].values,
            'training_times_std': model_df['training_times_std'].values,
        }

    # --- Plotting ---
    os.makedirs(plots_dir, exist_ok=True)
    plot_results(final_plot_data, exp_name, plots_dir)

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate experiment results and generate plots.")
    parser.add_argument("--exp", type=str, help="Name of the experiment to aggregate.")
    parser.add_argument("--config", type=str, help="Path to a configuration JSON file containing experiment definitions.")
    
    # Define absolute paths for directories
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(base_dir, 'results')
    plots_dir = os.path.join(base_dir, 'plots')

    args = parser.parse_args()

    if args.config:
        if args.exp:
            parser.error("Cannot specify both --exp and --config. Choose one.")
        
        config_path = os.path.abspath(os.path.join(base_dir, args.config))
        if not os.path.exists(config_path):
            parser.error(f"Configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        for exp_config in configs:
            if exp_config.get("enabled", False): # Only process enabled experiments
                exp_name = exp_config.get("experiment_name")
                if exp_name:
                    print(f"Aggregating and plotting results for experiment: {exp_name}")
                    aggregate_results(exp_name, results_dir, plots_dir)
                else:
                    print(f"Warning: Experiment configuration missing 'experiment_name': {exp_config}")
    elif args.exp:
        aggregate_results(args.exp, results_dir, plots_dir)
    else:
        parser.error("Either --exp or --config must be specified.")
