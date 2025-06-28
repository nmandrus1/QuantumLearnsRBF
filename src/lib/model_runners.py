"""
Functions for running the machine learning models.
"""

import time
import numpy as np
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from .kernel_factories import create_fixed_kernel, create_trainable_kernel


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a model and evaluates its performance.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "training_time": training_time,
    }


def run_rbf_svm(config: dict, X_train, y_train, X_test, y_test, seed: int, **kwargs):
    """
    Configures, trains, and evaluates a classical RBF SVM.
    """
    # Check if hyperparameters have been tuned and passed in the config
    if "tuned_rbf_params" in config:
        params = config["tuned_rbf_params"]
        C = params.get("C", 1.0)
        gamma = params.get("gamma", "scale")
        print(f"  Running RBF SVM with tuned params: C={C}, gamma={gamma}", flush=True)
    else:
        # Fallback for experiments without tuning (e.g., quantum_learns_rbf)
        C = 1.0  # Default C, not tuned in this path
        gamma = config.get("rbf_data_params", {}).get("gamma", "scale")

    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)
    return train_and_evaluate_model(model, X_train, y_train, X_test, y_test)


def run_quantum_svm(
    config: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
    trained_kernel=None,
):
    """
    Configures, trains, and evaluates a Quantum SVM.
    """
    n_features = X_train.shape[1]

    if config["experiment_type"] == "rbf_learns_quantum":
        # For this case, the quantum kernel is fixed and defined by the data source
        kernel = create_fixed_kernel(config, n_features)
        model = QSVC(quantum_kernel=kernel)

    elif config["experiment_type"] in [
        "quantum_learns_rbf",
        "trained_quantum_learns_quantum",
    ]:
        # For this case, we might be using a fixed or a trainable kernel
        if config.get("kernel_type") == "trainable":
            if trained_kernel is None:
                raise ValueError("Trainable kernel requires a pre-trained kernel.")
            kernel = trained_kernel
        else:  # Default to a fixed kernel, which needs to be tuned
            # fixed kernel was tuned on subset of the data before this
            if trained_kernel is None:
                raise ValueError("Fixed kernel requires a pre-trained kernel.")
            kernel = trained_kernel

        model = QSVC(quantum_kernel=kernel)
    else:
        raise ValueError(
            f"Invalid experiment type for run_quantum_svm: {config['experiment_type']}"
        )

    return train_and_evaluate_model(model, X_train, y_train, X_test, y_test)


def _tune_fixed_quantum_kernel(
    config: dict, X_tune: np.ndarray, y_tune: np.ndarray, seed: int
):
    """
    Performs a simplified random search for the best fixed Pauli kernel.
    """
    print("Tuning fixed quantum kernel...")
    q_tune_params = config["qsvc_tuning_params"]
    param_space = q_tune_params["parameter_space"]
    n_features = X_tune.shape[1]
    search_results = []

    for i in range(q_tune_params["n_iter_random"]):
        C = np.random.choice(param_space["qsvc__C"])
        reps = np.random.choice(param_space["qsvc__reps"])
        entanglement = np.random.choice(param_space["qsvc__entanglement"])
        paulis = _generate_constrained_pauli_set(
            param_space["pauli_search_constraints"]
        )

        fm = PauliFeatureMap(
            feature_dimension=n_features,
            reps=reps,
            entanglement=entanglement,
            paulis=paulis,
        )
        kernel = FidelityStatevectorKernel(feature_map=fm)
        qsvc = QSVC(C=C, quantum_kernel=kernel)

        # Using cross-validation for robust scoring
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        score = np.mean(
            cross_val_score(qsvc, X_tune, y_tune, cv=cv, scoring="accuracy")
        )
        search_results.append({"score": score, "kernel": kernel, "C": C})

    best_result = max(search_results, key=lambda x: x["score"])
    print(f"Best fixed kernel found with CV score: {best_result['score']:.4f}")
    return best_result["kernel"]


def _generate_constrained_pauli_set(constraints: dict) -> list[str]:
    """Generates a single, valid set of Pauli strings for randomized search."""
    base_paulis = constraints.get("base_paulis", ["X", "Y", "Z"])
    max_order = constraints.get("max_pauli_order", 2)
    max_terms = constraints.get("max_num_terms_in_set", 1)
    min_terms = constraints.get("min_num_terms_in_set", 1)

    pauli_set = set()
    # Ensure at least one high-order term for entanglement
    order = np.random.randint(2, max_order + 1)
    pauli_set.add("".join(np.random.choice(base_paulis, size=order)))

    num_terms = np.random.randint(min_terms, max_terms + 1)
    while len(pauli_set) < num_terms:
        order = np.random.randint(1, max_order + 1)
        pauli_set.add("".join(np.random.choice(base_paulis, size=order)))

    return list(pauli_set)
