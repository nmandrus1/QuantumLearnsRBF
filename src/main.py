"""
Main Experiment Launcher.

This script reads a JSON configuration file, iterates through the defined
experiments, and launches parallel worker processes to execute the trials.
"""

import argparse
import json
import os
import subprocess
from multiprocessing import Pool, cpu_count


def launch_worker(args):
    """
    Helper function to launch a single experiment_worker process.
    We pass the config as a JSON string to avoid file I/O race conditions.
    """
    config_json_string, seed, results_dir = args
    command = [
        "python",
        "experiment_worker.py",
        "--config_json",
        config_json_string,
        "--seed",
        str(seed),
        "--results_dir",
        results_dir,
    ]

    print(command)

    # We run the subprocess from the src directory to ensure correct module resolution
    # By removing capture_output=True, the worker's stdout/stderr will be streamed in real-time.
    process = subprocess.run(
        command, cwd=os.path.dirname(__file__), text=True
    )

    if process.returncode != 0:
        print(f"--- ERROR in worker (seed {seed}) ---")
        # The worker's output has already been streamed.
        print("-------------------------------------")
        return False

    # A final message is useful to confirm completion, but the worker's output is now streamed.
    print(f"Worker with seed {seed} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Launch Quantum Kernel experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config file (e.g., ../configs/your_config.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers to use.",
    )

    args = parser.parse_args()

    # Define absolute paths for directories
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(base_dir, "results")
    config_path = os.path.join(base_dir, os.path.basename(args.config))

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    with open(args.config, "r") as f:
        all_experiments = json.load(f)

    enabled_experiments = [exp for exp in all_experiments if exp.get("enabled", True)]

    print(f"Found {len(enabled_experiments)} enabled experiments.")
    print(f"Using {args.workers} parallel workers.")

    for exp_config in enabled_experiments:
        exp_name = exp_config["experiment_name"]
        print(f"\n{'=' * 60}\nLaunching Experiment: {exp_name}\n{'=' * 60}")

        exec_params = exp_config["execution_params"]
        num_trials = exec_params["num_trials"]
        base_seed = exec_params["base_seed"]

        # Create a list of tasks for the worker pool
        tasks = []
        config_str = json.dumps(exp_config)  # Serialize config once
        for i in range(num_trials):
            seed = base_seed + i
            tasks.append((config_str, seed, results_dir))

        # Run tasks in parallel
        with Pool(processes=args.workers) as pool:
            results = pool.map(launch_worker, tasks)

        if all(results):
            print(
                f"\nExperiment '{exp_name}' completed successfully for all {num_trials} trials."
            )
            print("To aggregate and plot the results, run:")
            print(f'  python aggregate_results.py --exp "{exp_name}"')
        else:
            print(
                f"\nExperiment '{exp_name}' had one or more failed trials. Check logs for details."
            )


if __name__ == "__main__":
    main()
