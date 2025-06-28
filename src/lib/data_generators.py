"""
Functions for generating datasets for the experiments.
"""

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.metrics.pairwise import rbf_kernel

# --- RBF Data Generation ---


def generate_rbf_data(params: dict, seed: int):
    """
    Generates a dataset based on a pseudo-RBF decision boundary.
    """
    rng = np.random.default_rng(seed)

    n_samples = params["n_samples"]
    n_support_vectors = params["n_support_vectors"]
    gamma = params["gamma"]
    feature_range = tuple(params.get("feature_range", (0, 2 * np.pi)))

    # Generate pseudo support vectors and their properties
    support_vectors = rng.uniform(
        feature_range[0], feature_range[1], (n_support_vectors, 2)
    )
    labels_sv = rng.choice([-1, 1], n_support_vectors)
    alphas = rng.uniform(0.1, 2.0, n_support_vectors)
    bias = rng.uniform(-1, 1)

    # Generate data points and classify them
    X = rng.uniform(feature_range[0], feature_range[1], (n_samples, 2))

    kernel_values = rbf_kernel(X, support_vectors, gamma=gamma)
    decision_values = np.sum(kernel_values * (alphas * labels_sv), axis=1) + bias
    y = np.sign(decision_values)

    # Handle points on the boundary
    y[y == 0] = rng.choice([-1, 1], np.sum(y == 0))

    return X, y.astype(int)


# --- Quantum Data Generation ---


def _exp_label(psi: np.ndarray, gap: float, mat_o: np.ndarray) -> np.ndarray:
    """Helper to compute labels from expectation values."""
    psi_dag_o = np.einsum("ij,jk->ik", psi.conj(), mat_o)
    exp_val = np.real(np.einsum("ik,ik->i", psi_dag_o, psi))
    return (np.abs(exp_val) > gap) * np.sign(exp_val)


def generate_quantum_data(feature_map, params: dict, seed: int):
    """
    Generates a dataset that is perfectly separable by a given quantum kernel.
    """
    algorithm_globals.random_seed = seed
    n_qubits = feature_map.num_qubits

    if n_qubits > 3:
        raise ValueError(
            "This data generation method is only recommended for n_qubits <= 3."
        )

    # 1. Define the separating observable
    rng = np.random.default_rng(seed)
    a = rng.random((2**n_qubits, 2**n_qubits)) + 1j * rng.random(
        (2**n_qubits, 2**n_qubits)
    )
    q, _ = np.linalg.qr(a)
    z_op = np.array([[1, 0], [0, -1]])
    zn_op = z_op
    for _ in range(1, n_qubits):
        zn_op = np.kron(zn_op, z_op)
    observable_matrix = q.conj().T @ zn_op @ q

    # 2. Create a grid of points to sample from
    grid_density = 200 if n_qubits == 2 else 64
    x_coords = [
        np.linspace(0, 2 * np.pi, grid_density, endpoint=False) for _ in range(n_qubits)
    ]
    grid_points = np.array(np.meshgrid(*x_coords)).T.reshape(-1, n_qubits)

    # 3. Compute statevectors and labels for the entire grid
    circuits = [feature_map.assign_parameters(x) for x in grid_points]
    statevectors = np.array([Statevector.from_instruction(c).data for c in circuits])
    labels = _exp_label(statevectors, params["gap"], observable_matrix)

    # 4. Sample from the labeled grid
    indices_a = np.where(labels == 1)[0]
    indices_b = np.where(labels == -1)[0]
    num_per_class = params["total_samples"] // 2

    if len(indices_a) < num_per_class or len(indices_b) < num_per_class:
        raise ValueError(
            "Cannot generate enough samples. Try lowering 'gap' or increasing grid density."
        )

    chosen_a = rng.choice(indices_a, size=num_per_class, replace=False)
    chosen_b = rng.choice(indices_b, size=num_per_class, replace=False)

    X = np.concatenate([grid_points[chosen_a], grid_points[chosen_b]])
    y = np.concatenate([np.ones(num_per_class), -np.ones(num_per_class)])

    # Shuffle the final dataset
    shuffle_idx = rng.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


# --- Noise Addition ---


def add_label_noise(y: np.ndarray, noise_rate: float, seed: int) -> np.ndarray:
    """
    Adds label noise to the dataset.

    Args:
        y (np.ndarray): The original labels (expected to be -1 or 1).
        noise_rate (float): The proportion of labels to flip (between 0.0 and 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: The labels with noise added.
    """
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError("noise_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    y_noisy = np.copy(y)
    n_samples = len(y)
    n_noise_samples = int(n_samples * noise_rate)

    if n_noise_samples == 0 and noise_rate > 0:
        print(
            f"Warning: noise_rate {noise_rate} is too small to flip any labels in {n_samples} samples."
        )
        return y_noisy

    # Randomly select indices to flip
    noise_indices = rng.choice(n_samples, n_noise_samples, replace=False)

    # Flip the labels at the selected indices
    y_noisy[noise_indices] *= -1

    return y_noisy
