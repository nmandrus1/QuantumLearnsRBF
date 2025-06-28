"""
Plotting functions for visualizing experiment results.
"""

import os
import matplotlib.pyplot as plt


def plot_results(results: dict, exp_name: str, output_dir: str):
    """
    Generates and saves a 2x2 grid of comparison plots.
    """
    print(f"Generating plots for experiment: {exp_name}")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Comparison for Experiment: {exp_name}", fontsize=18)

    model_names = list(results.keys())
    colors = ["red", "blue", "green", "purple"]  # Extend if more models
    markers = ["o", "s", ">", "<"]

    for i, name in enumerate(model_names):
        res = results[name]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Learning Curves
        ax1.plot(
            res["train_sizes"],
            res["train_accuracies_mean"],
            marker=marker,
            linestyle="-",
            color=color,
            label=f"{name} Train",
        )
        ax1.fill_between(
            res["train_sizes"],
            res["train_accuracies_mean"] - res["train_accuracies_std"],
            res["train_accuracies_mean"] + res["train_accuracies_std"],
            alpha=0.2,
            color=color,
        )
        ax1.plot(
            res["train_sizes"],
            res["test_accuracies_mean"],
            marker=marker,
            linestyle="--",
            color=color,
            label=f"{name} Test",
        )
        ax1.fill_between(
            res["train_sizes"],
            res["test_accuracies_mean"] - res["test_accuracies_std"],
            res["test_accuracies_mean"] + res["test_accuracies_std"],
            alpha=0.2,
            color=color,
        )

        # Generalization Gap
        gap_mean = res["train_accuracies_mean"] - res["test_accuracies_mean"]
        gap_std = (
            res["train_accuracies_std"] + res["test_accuracies_std"]
        )  # Error propagation
        ax2.plot(
            res["train_sizes"],
            gap_mean,
            marker=marker,
            linestyle="-",
            color=color,
            label=name,
        )
        ax2.fill_between(
            res["train_sizes"],
            gap_mean - gap_std,
            gap_mean + gap_std,
            alpha=0.2,
            color=color,
        )

        # Test Performance
        ax3.plot(
            res["train_sizes"],
            res["test_accuracies_mean"],
            marker=marker,
            linestyle="-",
            color=color,
            label=name,
        )
        ax3.fill_between(
            res["train_sizes"],
            res["test_accuracies_mean"] - res["test_accuracies_std"],
            res["test_accuracies_mean"] + res["test_accuracies_std"],
            alpha=0.2,
            color=color,
        )

        # Training Time
        ax4.plot(
            res["train_sizes"],
            res["training_times_mean"],
            marker=marker,
            linestyle="-",
            color=color,
            label=name,
        )
        ax4.fill_between(
            res["train_sizes"],
            res["training_times_mean"] - res["training_times_std"],
            res["training_times_mean"] + res["training_times_std"],
            alpha=0.2,
            color=color,
        )

    ax1.set_title("Learning Curves")
    ax1.set_xlabel("Training Set Size")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Generalization Gap (Train - Test)")
    ax2.set_xlabel("Training Set Size")
    ax2.set_ylabel("Gap")
    ax2.axhline(0, color="k", linestyle=":", alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Test Accuracy vs. Training Size")
    ax3.set_xlabel("Training Set Size")
    ax3.set_ylabel("Test Accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_title("Training Time")
    ax4.set_xlabel("Training Set Size")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_path = os.path.join(output_dir, f"{exp_name}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plots saved to {plot_path}")
