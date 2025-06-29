import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

# Set up logging to see progress and potential issues
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def flatten_quantum_kernel_trial_data(trial_path: Path, master_config: dict) -> list:
    """
    Loads a single trial pickle file and flattens its nested structure into a list
    of records, where each record represents one point on a learning curve.

    Args:
        trial_path (Path): Path to the 'trial_seed_*.pkl' file.
        master_config (dict): The master configuration for this experiment.

    Returns:
        list: A list of flat dictionaries, each being a row in the final DataFrame.
    """
    with open(trial_path, "rb") as f:
        trial_data = pickle.load(f)

    records = []

    # --- Extract Independent Variables from the master config ---
    # These are constant for every row generated from this file.
    exp_name = master_config.get("experiment_name", "N/A")
    kernel_type = master_config.get(
        "kernel_type", "Fixed"
    )  # Assume 'Fixed' if not specified
    gamma = master_config.get("rbf_data_params", {}).get("gamma", 0)
    noise = master_config.get("add_noise_percent", 0)
    seed = trial_data.get(
        "seed", int(trial_path.stem.split("_")[-1])
    )  # Get seed from data or filename

    # --- Extract Dependent Variables from the trial results ---
    # The 'results' dictionary contains the learning curve data.
    for model_name, model_results in trial_data.items():
        # Each 'model_results' dict has lists for train_sizes, accuracies, etc.
        for i, train_size in enumerate(model_results.get("train_sizes", [])):
            train_acc = model_results["train_accuracies"][i]
            test_acc = model_results["test_accuracies"][i]
            train_time = model_results["training_times"][i]

            # Create a single flat record for this specific data point
            record = {
                "experiment_name": exp_name,
                "kernel_type": kernel_type,
                "rbf_gamma": gamma,
                "noise_percent": noise,
                "trial_seed": seed,
                "train_size": train_size,
                "model_name": model_name,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "generalization_gap": train_acc - test_acc,
                "training_time_sec": train_time,
            }
            records.append(record)

    return records


def flatten_rbf_kernel_trial_data(trial_path: Path, master_config: dict) -> list:
    """
    Loads a single trial pickle file and flattens its nested structure into a list
    of records, where each record represents one point on a learning curve.

    Args:
        trial_path (Path): Path to the 'trial_seed_*.pkl' file.
        master_config (dict): The master configuration for this experiment.

    Returns:
        list: A list of flat dictionaries, each being a row in the final DataFrame.
    """
    with open(trial_path, "rb") as f:
        trial_data = pickle.load(f)

    records = []

    # --- Extract Independent Variables from the master config ---
    # These are constant for every row generated from this file.
    exp_name = master_config.get("experiment_name", "N/A")
    kernel_type = "rbf"
    entanglement = master_config["feature_map"]["entanglement"]
    noise = master_config.get("add_noise_percent", 0)

    seed = trial_data.get(
        "seed", int(trial_path.stem.split("_")[-1])
    )  # Get seed from data or filename

    # --- Extract Dependent Variables from the trial results ---
    # The 'results' dictionary contains the learning curve data.
    for model_name, model_results in trial_data.items():
        # Each 'model_results' dict has lists for train_sizes, accuracies, etc.
        for i, train_size in enumerate(model_results.get("train_sizes", [])):
            train_acc = model_results["train_accuracies"][i]
            test_acc = model_results["test_accuracies"][i]
            train_time = model_results["training_times"][i]

            # Create a single flat record for this specific data point
            record = {
                "experiment_name": exp_name,
                "kernel_type": kernel_type,
                "quantum_data_entanglment": entanglement,
                "noise_percent": noise,
                "trial_seed": seed,
                "train_size": train_size,
                "model_name": model_name,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "generalization_gap": train_acc - test_acc,
                "training_time_sec": train_time,
            }
            records.append(record)

    return records


def create_dataframe_from_config(
    config_path: Path, results_dir: Path, output_dir: Path
):
    """
    Generates a pandas DataFrame from a given JSON config by aggregating all trials.
    """
    try:
        with open(config_path, "r") as f:
            experiments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to read or parse config file {config_path}: {e}")
        return

    all_records = []

    for exp_config in experiments:
        if not exp_config.get("enabled", True):
            continue

        exp_name = exp_config["experiment_name"]
        exp_type = exp_config["experiment_type"]
        exp_dir = results_dir / exp_name
        logging.info(f"Processing experiment: {exp_name}")

        if not exp_dir.is_dir():
            logging.warning(f"Results directory not found for {exp_name}. Skipping.")
            continue

        # Find all trial result files
        trial_files = list(exp_dir.glob("trial_seed_*.pkl"))
        if not trial_files:
            logging.warning(f"No trial results found for {exp_name}.")
            continue

        for trial_file in trial_files:
            try:
                # Pass the specific experiment's config to the flattening function
                if exp_type in ["quantum_learns_rbf"]:
                    records = flatten_quantum_kernel_trial_data(trial_file, exp_config)
                elif exp_type in ["rbf_learns_quantum"]:
                    records = flatten_rbf_kernel_trial_data(trial_file, exp_config)

                if records is None:
                    logging.error(f"No records for experiment {trial_file}")

                all_records.extend(records)
            except Exception as e:
                logging.error(f"Failed to process {trial_file}: {e}")

    if not all_records:
        logging.info("No data processed. No DataFrame will be created.")
        return

    df = pd.DataFrame(all_records)

    # Save the final, clean CSV
    output_filename = config_path.stem + "_dataframe.csv"
    output_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Successfully created DataFrame with {len(df)} rows.")
        logging.info(f"DataFrame saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a consolidated DataFrame from experiment results."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration JSON file.",
    )

    args = parser.parse_args()

    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    config_path = Path(args.config)

    # Make sure we use an absolute path for the config if a relative one is given
    if not config_path.is_absolute():
        config_path = project_root / config_path

    results_dir = project_root / "results"
    output_dir = project_root / "analysis"

    if not config_path.exists():
        logging.error(f"Configuration file not found at: {config_path.resolve()}")
        return

    create_dataframe_from_config(config_path, results_dir, output_dir)


if __name__ == "__main__":
    main()
