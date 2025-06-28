"""
Factory functions for creating quantum feature maps and kernels.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import (
    FidelityStatevectorKernel,
    TrainableFidelityStatevectorKernel,
)
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from sklearn.model_selection import train_test_split

# --- Fixed Kernel Factories ---


def create_fixed_kernel(config: dict, n_features: int):
    """
    Creates a fixed (non-trainable) quantum kernel.
    """
    fm_config = config["feature_map"]
    name = fm_config["name"]
    reps = fm_config["reps"]
    entanglement = fm_config["entanglement"]

    if name == "ZZFeatureMap":
        feature_map = ZZFeatureMap(
            feature_dimension=n_features, reps=reps, entanglement=entanglement
        )
    elif name == "PauliFeatureMap":
        feature_map = PauliFeatureMap(
            feature_dimension=n_features,
            reps=reps,
            entanglement=entanglement,
            paulis=fm_config["paulis"],
        )
    else:
        raise ValueError(f"Unsupported feature map: {name}")

    return FidelityStatevectorKernel(feature_map=feature_map)


# --- Trainable Kernel Factory ---


def create_reuploading_circuit(
    num_qubits: int,
    data_dimension: int,
    reps: int = 1,
    entanglement: str = "alternating",
    skip_final_entanglement: bool = True,
) -> QuantumCircuit:
    """
    Constructs a data re-uploading quantum classifier using the compact scheme.

    This circuit architecture is inspired by the paper "Data re-uploading
    for a universal quantum classifier" (arXiv:1907.02085, Fig. 2b).
    It uses a compact layer structure where data features (x) are combined
    with weights (w) and biases (θ) to form the angles for a single
    rotation block, i.e., U(θ + w * x).

    Args:
        num_qubits: The number of qubits for the circuit.
        data_dimension: The dimensionality of the classical input data vector.
        reps: The number of layers (repetitions) of the data re-uploading block.
        entanglement: The entanglement strategy to use. Can be 'linear', 'full',
                      'alternating' (for 4 qubits), or 'none'. Defaults to 'alternating'.
        skip_final_entanglement: If True, the entanglement block is omitted
                                 in the final repetition. Defaults to True.

    Returns:
        A QuantumCircuit object representing the data re-uploading classifier.
    """
    # --- Validate Inputs ---
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer.")
    if not isinstance(data_dimension, int) or data_dimension < 1:
        raise ValueError("data_dimension must be a positive integer.")
    if not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be a positive integer.")

    entanglement_options = ["linear", "full", "alternating", "none", False]
    if entanglement not in entanglement_options:
        raise ValueError(f"entanglement must be one of {entanglement_options}.")
    if entanglement in ["none", False]:
        entanglement_strategy = "none"
    else:
        entanglement_strategy = entanglement.lower()

    if entanglement_strategy == "alternating" and num_qubits != 4:
        raise ValueError(
            "The 'alternating' entanglement strategy is only implemented for 4 qubits."
        )

    # --- Initialize Parameters and Circuit ---
    # Parameters for the classical data vector
    x = ParameterVector("x", data_dimension)

    # Calculate how many chunks of 3 are needed to encode the data
    num_data_chunks = int(np.ceil(data_dimension / 3))

    # Parameters for trainable biases (θ) and weights (w)
    # For each qubit, for each layer, for each data chunk, we have 3 biases and 3 weights
    num_params_per_layer = num_qubits * num_data_chunks * 3
    thetas = ParameterVector("θ", reps * num_params_per_layer)
    weights = ParameterVector("w", reps * num_params_per_layer)

    qc = QuantumCircuit(num_qubits, name=f"compact_reupload_{num_qubits}q_{reps}r")

    # --- Build the Circuit Layers ---
    param_idx = 0
    for layer in range(reps):
        qc.barrier(label=f"Rep {layer + 1}")

        # 1. Compact Data & Processing Layer
        # Each qubit gets a sequence of U gates. The angles for each U gate are a
        # combination of data, weights, and biases.
        for q in range(num_qubits):
            for k in range(num_data_chunks):
                # For each data chunk, we construct the three angles for the U gate.
                # angle_i = theta_i + weight_i * x_i

                # Get indices for x, θ, and w
                x_idx_0 = k * 3
                x_idx_1 = k * 3 + 1
                x_idx_2 = k * 3 + 2

                # Use 0.0 for data feature if dimension is not a multiple of 3
                x0 = x[x_idx_0] if x_idx_0 < data_dimension else 0.0
                x1 = x[x_idx_1] if x_idx_1 < data_dimension else 0.0
                x2 = x[x_idx_2] if x_idx_2 < data_dimension else 0.0

                # Construct the three parameter expressions for the U gate
                angle0 = thetas[param_idx] + weights[param_idx] * x0
                angle1 = thetas[param_idx + 1] + weights[param_idx + 1] * x1
                angle2 = thetas[param_idx + 2] + weights[param_idx + 2] * x2

                # qiskit.circuit.QuantumCircuit.u(theta, phi, lambda, qubit)
                qc.u(angle0, angle1, angle2, q)
                param_idx += 3

        # 2. Entanglement Layer
        # Apply entangling gates between qubits.
        is_not_final_rep = layer < reps - 1
        apply_entanglement = (entanglement_strategy != "none" and num_qubits > 1) and (
            not skip_final_entanglement or is_not_final_rep
        )

        if apply_entanglement:
            qc.barrier(label="Entanglement")
            if entanglement_strategy == "linear":
                for q in range(num_qubits - 1):
                    qc.cz(q, q + 1)
            elif entanglement_strategy == "full":
                for q1 in range(num_qubits):
                    for q2 in range(q1 + 1, num_qubits):
                        qc.cz(q1, q2)
            elif entanglement_strategy == "alternating":
                # This pattern is for 4 qubits, as in Fig. 5 of the paper.
                if layer % 2 == 0:  # Even layers (0, 2, ...)
                    qc.cz(0, 1)
                    qc.cz(2, 3)
                else:  # Odd layers (1, 3, ...)
                    qc.cz(1, 2)
                    qc.cz(
                        0, 3
                    )  # As per paper, this is (1)-(4) which is 0-indexed (0)-(3)

    return qc


def create_trainable_kernel(
    config: dict, X_train: np.ndarray, y_train: np.ndarray, seed: int
):
    """
    Builds and trains a data re-uploading kernel.
    """
    kernel_params = config["trainable_kernel_params"]
    trainer_config = kernel_params["trainer_config"]

    n_features = X_train.shape[1]

    feature_map = create_reuploading_circuit(
        num_qubits=kernel_params["num_qubits"],
        data_dimension=n_features,
        reps=kernel_params["reps"],
        entanglement=kernel_params["entanglement"],
    )

    trainable_params = [param for param in feature_map.parameters if "θ" in param.name]
    weights = [param for param in feature_map.parameters if "w" in param.name]

    quant_kernel = TrainableFidelityStatevectorKernel(
        feature_map=feature_map,
        training_parameters=trainable_params + weights,
    )

    spsa_opt = SPSA(maxiter=trainer_config["max_iters"])
    svc_loss = SVCLoss(C=trainer_config["regularization_c"])

    # generate random numbers from 0-pi
    initial_points = np.random.default_rng(seed).uniform(
        0, np.pi, size=len(quant_kernel.training_parameters)
    )

    qkt = QuantumKernelTrainer(
        quantum_kernel=quant_kernel,
        loss=svc_loss,
        optimizer=spsa_opt,
        initial_point=initial_points,
    )

    print(f"Starting kernel training on {len(X_train)} samples...")
    qka_results = qkt.fit(X_train, y_train)
    print(f"Kernel training complete. Final loss: {qka_results.optimal_value:.4f}")

    return qka_results.quantum_kernel
