"""
Experiment Worker: Executes a single trial of an experiment.

This script is designed to be called by the main launcher. It takes a JSON
configuration string and a seed, runs the specified experiment trial, and
-saves the results to a file.
"""

import argparse
import json
import os
import pickle

# Set backend for matplotlib before other imports
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.svm import SVC

matplotlib.use("Agg")

from lib.data_generators import (
    add_label_noise,
    generate_quantum_data,
    generate_rbf_data,
)
from lib.model_runners import run_quantum_svm, run_rbf_svm, _tune_fixed_quantum_kernel


def run_single_trial(config: dict, seed: int, results_dir: str):
    """
    Executes one full trial of an experiment configuration.
    """
    exp_name = config["experiment_name"]
    trial_id = f"trial_seed_{seed}"
    output_path = os.path.join(results_dir, exp_name, f"{trial_id}.pkl")

    if os.path.exists(output_path):
        print(f"Skipping already completed trial: {trial_id}", flush=True)
        return

    print(f"--- Starting Trial: {exp_name} (Seed: {seed}) ---", flush=True)
    np.random.seed(seed)

    # 1. Data Generation
    if config["experiment_type"] == "quantum_learns_rbf":
        X, y = generate_rbf_data(config["rbf_data_params"], seed)
        test_size = config["rbf_data_params"]["test_size"]
    elif config["experiment_type"] in [
        "rbf_learns_quantum",
        "trained_quantum_learns_quantum",
    ]:
        # This requires a feature map object to be created first
        from lib.kernel_factories import create_fixed_kernel

        # Note: This is a bit circular, but necessary as the data depends on the kernel config
        temp_kernel = create_fixed_kernel(config, config["feature_map"]["n_qubits"])
        X, y = generate_quantum_data(
            temp_kernel.feature_map, config["data_generation_params"], seed
        )
        test_size = config["data_generation_params"]["test_size"]
    else:
        raise ValueError(f"Unknown experiment type: {config['experiment_type']}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Optional: Add noise to the entire training set (including kernel training data)
    if config.get("add_noise_percent", 0) > 0:
        y_train_full = add_label_noise(
            y_train_full, config["add_noise_percent"] / 100, seed
        )

    # 3-way split for tuning kernels
    X_kernel_train, y_kernel_train = None, None
    if config["experiment_type"] in [
        "quantum_learns_rbf",
        "trained_quantum_learns_quantum",
    ]:
        if config.get("kernel_type") == "trainable":
            kernel_train_size = config["trainable_kernel_params"]["trainer_config"][
                "training_subset_size"
            ]
        else:
            kernel_train_size = config["qsvc_tuning_params"]["tuning_samples"]

        print(f"Training kernel with {kernel_train_size} datapoints", flush=True)

        X_kernel_train, X_train_full, y_kernel_train, y_train_full = train_test_split(
            X_train_full,
            y_train_full,
            test_size=(len(X_train_full) - kernel_train_size) / len(X_train_full),
            random_state=seed,
            stratify=y_train_full,
        )
        # Train the kernel once here
        if config.get("kernel_type") == "trainable":
            from lib.kernel_factories import create_trainable_kernel

            print("Training trainable quantum kernel...", flush=True)
            trained_kernel = create_trainable_kernel(
                config, X_kernel_train, y_kernel_train, seed
            )
        else:
            # random hyperparameter search
            trained_kernel = _tune_fixed_quantum_kernel(
                config, X_kernel_train, y_kernel_train, seed
            )

    else:
        # Do hyperparameter tuning for RBF SVM
        if (
            "rbf_tuning_params" in config
            and config["experiment_type"] == "rbf_learns_quantum"
        ):
            print("Tuning RBF SVM hyperparameters...", flush=True)
            tuning_config = config["rbf_tuning_params"]
            tuning_samples = tuning_config["tuning_samples"]

            # Stratified split to get a subset for tuning. The rest of the data is kept for the main training loop.
            X_tune, X_train_full, y_tune, y_train_full = train_test_split(
                X_train_full,
                y_train_full,
                train_size=tuning_samples,
                random_state=seed,
                stratify=y_train_full,
            )

            param_grid = {"C": tuning_config["C"], "gamma": tuning_config["gamma"]}

            # Using GridSearchCV to find the best hyperparameters
            grid_search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_tune, y_tune)

            best_params = grid_search.best_params_
            print(f"Best RBF params found: {best_params}", flush=True)

            # Store the best params to be used by the RBF SVM runner
            config["tuned_rbf_params"] = best_params

        trained_kernel = None

    # 2. Run models over different training sizes
    trial_results = {}
    training_sizes = config["comparison_train_sizes"]

    for size in training_sizes:
        if size > len(X_train_full):
            continue

        X_train_step = X_train_full[:size]
        y_train_step = y_train_full[:size]

        # Define models for this trial
        if config["experiment_type"] == "quantum_learns_rbf":
            models_to_run = {"Quantum_SVM": run_quantum_svm, "RBF_SVM": run_rbf_svm}
        else:  # rbf_learns_quantum
            models_to_run = {"RBF_SVM": run_rbf_svm, "Quantum_SVM": run_quantum_svm}

        for model_name, runner_func in models_to_run.items():
            if model_name not in trial_results:
                trial_results[model_name] = {
                    "train_sizes": [],
                    "train_accuracies": [],
                    "test_accuracies": [],
                    "training_times": [],
                }

            print(f"  Training {model_name} on {size} samples...", flush=True)
            result = runner_func(
                config,
                X_train_step,
                y_train_step,
                X_test,
                y_test,
                seed,
                trained_kernel=trained_kernel,  # Pass the trained kernel
            )

            trial_results[model_name]["train_sizes"].append(size)
            trial_results[model_name]["train_accuracies"].append(
                result["train_accuracy"]
            )
            trial_results[model_name]["test_accuracies"].append(result["test_accuracy"])
            trial_results[model_name]["training_times"].append(result["training_time"])

    # 3. Save trial results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trial_results, f)

    print(f"--- Trial Complete. Results saved to {output_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment trial.")
    parser.add_argument(
        "--config_json",
        type=str,
        required=True,
        help="JSON string of the experiment config.",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for this trial."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory to save trial results.",
    )

    args = parser.parse_args()

    # The config is passed as a JSON string to avoid file I/O issues in parallel execution
    config_dict = json.loads(args.config_json)

    run_single_trial(config_dict, args.seed, args.results_dir)
