from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CriterionName = Literal[
    "random",
    "output_weight_norm",
    "mean_activation",
    "firing_rate",
    "activation_variance",
    "class_selectivity",
    "redundancy",
    "hebbian",
    "activation_fisher",
    "weight_fisher",
    "single_neuron_ablation",
]


LOSS_FREE_CRITERIA: tuple[str, ...] = (
    "output_weight_norm",
    "mean_activation",
    "firing_rate",
    "activation_variance",
    "class_selectivity",
    "redundancy",
    "hebbian",
)

LOSS_AWARE_CRITERIA: tuple[str, ...] = (
    "activation_fisher",
    "weight_fisher",
)

REFERENCE_CRITERIA: tuple[str, ...] = ("single_neuron_ablation",)

DEFAULT_CRITERIA: tuple[str, ...] = (
    "random",
    "output_weight_norm",
    "mean_activation",
    "firing_rate",
    "activation_variance",
    "class_selectivity",
    "redundancy",
    "hebbian",
    "activation_fisher",
    "weight_fisher",
)

ARCHITECTURE_HIDDEN_DIMS: dict[str, tuple[int, ...]] = {
    "small": (256, 128),
    "medium": (512, 256, 128, 64),
    "deep": (512, 512, 256, 256, 128, 64),
    "cifar_flat": (1024, 512, 512, 256, 128),
}


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    input_shape: tuple[int, ...]
    input_dim: int
    num_classes: int


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    calibration_loader: DataLoader
    metadata: DatasetMetadata
    train_size: int
    val_size: int
    test_size: int
    calibration_size: int


@dataclass(frozen=True)
class ScoreCost:
    scoring_time_seconds: float = 0.0
    scoring_memory_mb: float = 0.0
    num_forward_passes: int = 0
    num_backward_passes: int = 0
    calibration_size: int = 0


@dataclass
class ProtocolConfig:
    dataset_names: tuple[str, ...] = ("MNIST", "FashionMNIST", "KMNIST")
    architectures: tuple[str, ...] = ("small", "medium", "deep")
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 10
    batch_size: int = 128
    val_size: int = 10_000
    calibration_size: int = 2_048
    num_workers: int = 2
    optimizer_name: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    dropout: float = 0.2
    activation_threshold: float = 0.0
    pruning_percentages: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70)
    fine_tune_epochs: tuple[int, ...] = (0, 1, 3, 5)
    criteria: tuple[str, ...] = DEFAULT_CRITERIA
    random_trials: int = 5
    prune_scopes: tuple[str, ...] = ("all", "layerwise")
    scoring_epochs: tuple[int | str, ...] = ("last",)
    stability_topk_ratios: tuple[float, ...] = (0.05, 0.10, 0.20)
    calibration_sensitivity_sizes: tuple[int, ...] = ()
    calibration_sensitivity_criteria: tuple[str, ...] = ("hebbian", "activation_fisher")
    weight_fisher_mode: Literal["batch", "per_sample"] = "batch"
    weight_fisher_parts: Literal["incoming", "outgoing", "both"] = "both"
    ablation_reference_metric: Literal["loss", "accuracy"] = "loss"
    save_checkpoints: bool = True
    download: bool = True
    data_root: str = "data"
    run_root: str = "runs/structured_pruning"
    device: str | None = None
    measure_compact_inference: bool = True


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonical_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().replace("-", "_").replace(" ", "_").upper()


def dataset_metadata(dataset_name: str) -> DatasetMetadata:
    canonical_name = canonical_dataset_name(dataset_name)
    if canonical_name == "MNIST":
        return DatasetMetadata("MNIST", (1, 28, 28), 28 * 28, 10)
    if canonical_name in {"FASHIONMNIST", "FASHION_MNIST"}:
        return DatasetMetadata("FashionMNIST", (1, 28, 28), 28 * 28, 10)
    if canonical_name == "KMNIST":
        return DatasetMetadata("KMNIST", (1, 28, 28), 28 * 28, 10)
    if canonical_name in {"CIFAR10", "CIFAR_10"}:
        return DatasetMetadata("CIFAR10", (3, 32, 32), 3 * 32 * 32, 10)
    if canonical_name in {"EMNIST_DIGITS", "EMNIST:DIGITS"}:
        return DatasetMetadata("EMNIST_DIGITS", (1, 28, 28), 28 * 28, 10)
    if canonical_name in {"EMNIST_LETTERS", "EMNIST:LETTERS"}:
        return DatasetMetadata("EMNIST_LETTERS", (1, 28, 28), 28 * 28, 26)
    if canonical_name in {"EMNIST_BALANCED", "EMNIST:BALANCED"}:
        return DatasetMetadata("EMNIST_BALANCED", (1, 28, 28), 28 * 28, 47)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_transforms(metadata: DatasetMetadata) -> transforms.Compose:
    if metadata.name == "CIFAR10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )


def load_dataset_pair(
    dataset_name: str,
    data_root: str | Path,
    download: bool,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, DatasetMetadata]:
    metadata = dataset_metadata(dataset_name)
    transform = build_transforms(metadata)
    root = str(data_root)

    if metadata.name == "MNIST":
        train_set = datasets.MNIST(root=root, train=True, download=download, transform=transform)
        test_set = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    elif metadata.name == "FashionMNIST":
        train_set = datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        test_set = datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
    elif metadata.name == "KMNIST":
        train_set = datasets.KMNIST(root=root, train=True, download=download, transform=transform)
        test_set = datasets.KMNIST(root=root, train=False, download=download, transform=transform)
    elif metadata.name == "CIFAR10":
        train_set = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    elif metadata.name.startswith("EMNIST"):
        split = metadata.name.split("_", maxsplit=1)[1].lower()
        target_transform = (lambda target: target - 1) if split == "letters" else None
        train_set = datasets.EMNIST(
            root=root,
            split=split,
            train=True,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        test_set = datasets.EMNIST(
            root=root,
            split=split,
            train=False,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, metadata


def build_data_bundle(
    dataset_name: str,
    config: ProtocolConfig,
    seed: int,
) -> DataBundle:
    train_val_set, test_set, metadata = load_dataset_pair(
        dataset_name=dataset_name,
        data_root=config.data_root,
        download=config.download,
    )

    if config.val_size + config.calibration_size >= len(train_val_set):
        raise ValueError(
            "Validation plus calibration split must be smaller than the training split: "
            f"val_size={config.val_size}, calibration_size={config.calibration_size}, "
            f"train_split_size={len(train_val_set)}"
        )

    train_size = len(train_val_set) - config.val_size - config.calibration_size
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set, calibration_set = random_split(
        train_val_set,
        [train_size, config.val_size, config.calibration_size],
        generator=split_generator,
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1000),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    calibration_loader = DataLoader(calibration_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        calibration_loader=calibration_loader,
        metadata=metadata,
        train_size=train_size,
        val_size=config.val_size,
        test_size=len(test_set),
        calibration_size=config.calibration_size,
    )


def hidden_dims_for_architecture(architecture: str, metadata: DatasetMetadata) -> tuple[int, ...]:
    architecture_key = architecture.strip().lower()
    if architecture_key == "cifar_flat" and metadata.name != "CIFAR10":
        raise ValueError("The cifar_flat architecture is intended for CIFAR10.")
    if architecture_key not in ARCHITECTURE_HIDDEN_DIMS:
        known = ", ".join(sorted(ARCHITECTURE_HIDDEN_DIMS))
        raise ValueError(f"Unsupported architecture: {architecture}. Known values: {known}")
    return ARCHITECTURE_HIDDEN_DIMS[architecture_key]


class MaskedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.flatten = nn.Flatten()
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(previous_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout_p))
            previous_dim = hidden_dim

        self.output_layer = nn.Linear(previous_dim, self.num_classes)
        self.layer_names = tuple(f"a{layer_index + 1}" for layer_index in range(len(self.hidden_dims)))
        self.layer_dims = dict(zip(self.layer_names, self.hidden_dims, strict=True))

    def forward(
        self,
        inputs: torch.Tensor,
        return_activations: bool = False,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        masks = masks or {}
        activations: dict[str, torch.Tensor] = {}
        hidden = self.flatten(inputs)

        for layer_name, linear_layer, dropout_layer in zip(
            self.layer_names,
            self.hidden_layers,
            self.dropouts,
            strict=True,
        ):
            hidden = F.relu(linear_layer(hidden))
            if layer_name in masks:
                mask = masks[layer_name].to(hidden.device, dtype=hidden.dtype).view(1, -1)
                hidden = hidden * mask
            activations[layer_name] = hidden
            hidden = dropout_layer(hidden)

        logits = self.output_layer(hidden)
        if return_activations:
            return logits, activations
        return logits


def make_model(metadata: DatasetMetadata, architecture: str, config: ProtocolConfig) -> MaskedMLP:
    hidden_dims = hidden_dims_for_architecture(architecture, metadata)
    return MaskedMLP(
        input_dim=metadata.input_dim,
        hidden_dims=hidden_dims,
        num_classes=metadata.num_classes,
        dropout=config.dropout,
    )


def make_optimizer(
    model: nn.Module,
    config: ProtocolConfig,
    learning_rate: float | None = None,
) -> torch.optim.Optimizer:
    lr = config.learning_rate if learning_rate is None else learning_rate
    if config.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss / 1024.0)
    except Exception:
        return 0.0


def train_one_epoch(
    model: MaskedMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, masks=masks)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.size(0))
        running_loss += float(loss.item()) * batch_size
        correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: MaskedMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    masks: dict[str, torch.Tensor] | None = None,
    num_classes: int | None = None,
    per_class: bool = False,
    measure_time: bool = False,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    if measure_time:
        synchronize_if_needed(device)
        start_time = time.perf_counter()
    else:
        start_time = 0.0

    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = None
    class_total = None

    if per_class:
        if num_classes is None:
            raise ValueError("num_classes is required when per_class=True")
        class_correct = torch.zeros(num_classes, dtype=torch.long)
        class_total = torch.zeros(num_classes, dtype=torch.long)

    num_forward_passes = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs, masks=masks)
        loss = loss_fn(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = int(targets.size(0))
        running_loss += float(loss.item()) * batch_size
        correct += int(predictions.eq(targets).sum().item())
        total += batch_size
        num_forward_passes += 1

        if per_class and class_correct is not None and class_total is not None:
            for class_index in range(num_classes):
                class_mask = targets.eq(class_index)
                class_total[class_index] += int(class_mask.sum().item())
                class_correct[class_index] += int(predictions[class_mask].eq(targets[class_mask]).sum().item())

    if measure_time:
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start_time
    else:
        elapsed = 0.0

    result: dict[str, Any] = {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "num_forward_passes": num_forward_passes,
        "inference_time_seconds": elapsed,
    }

    if per_class and class_correct is not None and class_total is not None:
        per_class_accuracy = []
        for class_index in range(num_classes):
            denominator = int(class_total[class_index].item())
            value = float(class_correct[class_index].item() / denominator) if denominator else float("nan")
            per_class_accuracy.append(value)
        result["per_class_accuracy"] = per_class_accuracy

    model.train(was_training)
    return result


class StreamingActivationStats:
    def __init__(
        self,
        layer_dims: dict[str, int],
        num_classes: int,
        threshold: float = 0.0,
        stats_device: torch.device | str = "cpu",
    ) -> None:
        self.layer_dims = layer_dims
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.stats_device = torch.device(stats_device)
        self.n_seen = 0

        self.activation_sum = {}
        self.activation_sq_sum = {}
        self.activation_cross_sum = {}
        self.firing_sum = {}
        self.binary_coactivation_sum = {}
        self.class_activation_sum = {}
        self.class_firing_sum = {}

        for layer_name, layer_dim in layer_dims.items():
            self.activation_sum[layer_name] = torch.zeros(layer_dim, device=self.stats_device)
            self.activation_sq_sum[layer_name] = torch.zeros(layer_dim, device=self.stats_device)
            self.activation_cross_sum[layer_name] = torch.zeros(layer_dim, layer_dim, device=self.stats_device)
            self.firing_sum[layer_name] = torch.zeros(layer_dim, device=self.stats_device)
            self.binary_coactivation_sum[layer_name] = torch.zeros(layer_dim, layer_dim, device=self.stats_device)
            self.class_activation_sum[layer_name] = torch.zeros(num_classes, layer_dim, device=self.stats_device)
            self.class_firing_sum[layer_name] = torch.zeros(num_classes, layer_dim, device=self.stats_device)

        self.class_counts = torch.zeros(num_classes, device=self.stats_device)

    @torch.no_grad()
    def update(
        self,
        activations: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> None:
        batch_size = int(targets.size(0))
        self.n_seen += batch_size
        targets = targets.detach().to(self.stats_device)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        self.class_counts += one_hot.sum(dim=0)

        for layer_name in self.layer_dims:
            activation = activations[layer_name].detach().to(self.stats_device, dtype=torch.float32)
            firing = (activation > self.threshold).float()

            self.activation_sum[layer_name] += activation.sum(dim=0)
            self.activation_sq_sum[layer_name] += activation.square().sum(dim=0)
            self.activation_cross_sum[layer_name] += activation.T @ activation
            self.firing_sum[layer_name] += firing.sum(dim=0)
            self.binary_coactivation_sum[layer_name] += firing.T @ firing
            self.class_activation_sum[layer_name] += one_hot.T @ activation
            self.class_firing_sum[layer_name] += one_hot.T @ firing

    def finalize(self) -> dict[str, torch.Tensor | int]:
        eps = 1e-8
        total_seen = max(int(self.n_seen), 1)
        stats: dict[str, torch.Tensor | int] = {
            "n_seen": int(self.n_seen),
            "class_counts": self.class_counts.detach().cpu(),
        }

        for layer_name in self.layer_dims:
            mean_activation = self.activation_sum[layer_name] / total_seen
            mean_sq_activation = self.activation_sq_sum[layer_name] / total_seen
            activation_variance = torch.clamp(mean_sq_activation - mean_activation.square(), min=0.0)
            activation_std = torch.sqrt(activation_variance)

            activation_covariance = self.activation_cross_sum[layer_name] / total_seen
            activation_covariance = activation_covariance - torch.outer(mean_activation, mean_activation)
            covariance_denominator = torch.sqrt(torch.outer(activation_variance, activation_variance)).clamp_min(eps)
            activation_correlation = torch.nan_to_num(activation_covariance / covariance_denominator)
            activation_correlation.fill_diagonal_(1.0)

            firing_rate = self.firing_sum[layer_name] / total_seen
            binary_coactivation = self.binary_coactivation_sum[layer_name] / total_seen

            firing_covariance = binary_coactivation - torch.outer(firing_rate, firing_rate)
            firing_variance = firing_rate * (1.0 - firing_rate)
            firing_denominator = torch.sqrt(torch.outer(firing_variance, firing_variance)).clamp_min(eps)
            firing_correlation = torch.nan_to_num(firing_covariance / firing_denominator)
            firing_correlation.fill_diagonal_(1.0)

            class_counts_safe = self.class_counts.clamp_min(1.0).view(-1, 1)
            class_mean_activation = self.class_activation_sum[layer_name] / class_counts_safe
            class_firing_rate = self.class_firing_sum[layer_name] / class_counts_safe
            class_selectivity = class_mean_activation.max(dim=0).values - class_mean_activation.mean(dim=0)
            class_activation_spread = class_mean_activation.max(dim=0).values - class_mean_activation.min(dim=0).values

            stats[f"{layer_name}_mean_activation"] = mean_activation.detach().cpu()
            stats[f"{layer_name}_activation_variance"] = activation_variance.detach().cpu()
            stats[f"{layer_name}_std_activation"] = activation_std.detach().cpu()
            stats[f"{layer_name}_firing_rate"] = firing_rate.detach().cpu()
            stats[f"{layer_name}_activation_correlation"] = activation_correlation.detach().cpu()
            stats[f"{layer_name}_correlation"] = activation_correlation.detach().cpu()
            stats[f"{layer_name}_firing_correlation"] = firing_correlation.detach().cpu()
            stats[f"{layer_name}_binary_coactivation"] = binary_coactivation.detach().cpu()
            stats[f"{layer_name}_class_mean_activation"] = class_mean_activation.detach().cpu()
            stats[f"{layer_name}_class_firing_rate"] = class_firing_rate.detach().cpu()
            stats[f"{layer_name}_class_selectivity"] = class_selectivity.detach().cpu()
            stats[f"{layer_name}_class_activation_spread"] = class_activation_spread.detach().cpu()

        return stats


@torch.no_grad()
def collect_activation_stats(
    model: MaskedMLP,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    threshold: float = 0.0,
) -> tuple[dict[str, torch.Tensor | int], ScoreCost]:
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()

    collector = StreamingActivationStats(
        layer_dims=model.layer_dims,
        num_classes=num_classes,
        threshold=threshold,
        stats_device="cpu",
    )
    num_forward_passes = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, activations = model(inputs, return_activations=True)
        del logits
        collector.update(activations, targets)
        num_forward_passes += 1

    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    memory_mb = current_memory_mb(device)
    stats = collector.finalize()
    model.train(was_training)

    return stats, ScoreCost(
        scoring_time_seconds=elapsed,
        scoring_memory_mb=memory_mb,
        num_forward_passes=num_forward_passes,
        num_backward_passes=0,
        calibration_size=int(stats["n_seen"]),
    )


def enrich_stats_with_model_weights(
    stats: dict[str, torch.Tensor | int],
    model: MaskedMLP,
) -> dict[str, torch.Tensor | int]:
    enriched = dict(stats)

    with torch.no_grad():
        for layer_index, layer_name in enumerate(model.layer_names):
            current_layer = model.hidden_layers[layer_index]
            if layer_index + 1 < len(model.hidden_layers):
                next_weight = model.hidden_layers[layer_index + 1].weight.detach().cpu()
            else:
                next_weight = model.output_layer.weight.detach().cpu()

            incoming_weight_norm = current_layer.weight.detach().cpu().norm(p=2, dim=1)
            outgoing_weight_norm = next_weight.norm(p=2, dim=0)
            activation_correlation = torch.as_tensor(enriched[f"{layer_name}_activation_correlation"]).clone()
            activation_correlation.fill_diagonal_(0.0)
            redundancy_mean_abs = activation_correlation.abs().sum(dim=1) / max(activation_correlation.size(0) - 1, 1)
            redundancy_max_abs = activation_correlation.abs().max(dim=1).values

            mean_activation = torch.as_tensor(enriched[f"{layer_name}_mean_activation"])
            firing_rate = torch.as_tensor(enriched[f"{layer_name}_firing_rate"])
            importance = mean_activation * firing_rate * outgoing_weight_norm
            hebbian_prune_score = importance / (1.0 + redundancy_max_abs)

            enriched[f"{layer_name}_incoming_weight_norm"] = incoming_weight_norm
            enriched[f"{layer_name}_outgoing_weight_norm"] = outgoing_weight_norm
            enriched[f"{layer_name}_redundancy_mean_abs_corr"] = redundancy_mean_abs
            enriched[f"{layer_name}_redundancy_max_abs_corr"] = redundancy_max_abs
            enriched[f"{layer_name}_importance"] = importance
            enriched[f"{layer_name}_hebbian_prune_score"] = hebbian_prune_score

    return enriched


def score_from_stats(
    stats: dict[str, torch.Tensor | int],
    model: MaskedMLP,
    criterion_name: str,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {}
    for layer_name in model.layer_names:
        if criterion_name == "output_weight_norm":
            score = torch.as_tensor(stats[f"{layer_name}_outgoing_weight_norm"]).float()
        elif criterion_name == "mean_activation":
            score = torch.as_tensor(stats[f"{layer_name}_mean_activation"]).float()
        elif criterion_name == "firing_rate":
            score = torch.as_tensor(stats[f"{layer_name}_firing_rate"]).float()
        elif criterion_name == "activation_variance":
            score = torch.as_tensor(stats[f"{layer_name}_activation_variance"]).float()
        elif criterion_name == "class_selectivity":
            score = torch.as_tensor(stats[f"{layer_name}_class_selectivity"]).float()
        elif criterion_name == "redundancy":
            score = -torch.as_tensor(stats[f"{layer_name}_redundancy_max_abs_corr"]).float()
        elif criterion_name == "hebbian":
            score = torch.as_tensor(stats[f"{layer_name}_hebbian_prune_score"]).float()
        else:
            raise ValueError(f"Unsupported loss-free criterion: {criterion_name}")
        scores[layer_name] = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).cpu()
    return scores


def activation_fisher_scores(
    model: MaskedMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], ScoreCost]:
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()

    accumulators = {
        layer_name: torch.zeros(layer_dim, dtype=torch.float32)
        for layer_name, layer_dim in model.layer_dims.items()
    }
    n_seen = 0
    num_forward_passes = 0
    num_backward_passes = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad(set_to_none=True)
        logits, activations = model(inputs, return_activations=True)
        for activation in activations.values():
            activation.retain_grad()
        loss = F.cross_entropy(logits, targets, reduction="sum")
        loss.backward()

        for layer_name, activation in activations.items():
            if activation.grad is None:
                raise RuntimeError(f"Missing activation gradient for {layer_name}")
            contribution = (activation.detach() * activation.grad.detach()).square().sum(dim=0)
            accumulators[layer_name] += contribution.cpu()

        batch_size = int(targets.size(0))
        n_seen += batch_size
        num_forward_passes += 1
        num_backward_passes += 1

    model.zero_grad(set_to_none=True)
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    memory_mb = current_memory_mb(device)
    model.train(was_training)

    scores = {
        layer_name: (accumulator / max(n_seen, 1)).cpu()
        for layer_name, accumulator in accumulators.items()
    }
    return scores, ScoreCost(
        scoring_time_seconds=elapsed,
        scoring_memory_mb=memory_mb,
        num_forward_passes=num_forward_passes,
        num_backward_passes=num_backward_passes,
        calibration_size=n_seen,
    )


def _empty_weight_fisher_accumulators(model: MaskedMLP) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    return {
        "hidden_weight": [torch.zeros_like(layer.weight, device="cpu") for layer in model.hidden_layers],
        "hidden_bias": [torch.zeros_like(layer.bias, device="cpu") for layer in model.hidden_layers],
        "output_weight": torch.zeros_like(model.output_layer.weight, device="cpu"),
        "output_bias": torch.zeros_like(model.output_layer.bias, device="cpu"),
    }


def _accumulate_weight_grad_squares(
    model: MaskedMLP,
    accumulators: dict[str, list[torch.Tensor] | torch.Tensor],
    weight: float,
) -> None:
    hidden_weight_accumulators = accumulators["hidden_weight"]
    hidden_bias_accumulators = accumulators["hidden_bias"]
    if not isinstance(hidden_weight_accumulators, list) or not isinstance(hidden_bias_accumulators, list):
        raise TypeError("Invalid Fisher accumulator structure")

    with torch.no_grad():
        for layer_index, layer in enumerate(model.hidden_layers):
            if layer.weight.grad is not None:
                hidden_weight_accumulators[layer_index] += layer.weight.grad.detach().cpu().square() * weight
            if layer.bias.grad is not None:
                hidden_bias_accumulators[layer_index] += layer.bias.grad.detach().cpu().square() * weight
        output_weight = accumulators["output_weight"]
        output_bias = accumulators["output_bias"]
        if not isinstance(output_weight, torch.Tensor) or not isinstance(output_bias, torch.Tensor):
            raise TypeError("Invalid Fisher accumulator structure")
        if model.output_layer.weight.grad is not None:
            output_weight += model.output_layer.weight.grad.detach().cpu().square() * weight
        if model.output_layer.bias.grad is not None:
            output_bias += model.output_layer.bias.grad.detach().cpu().square() * weight


def _normalise_weight_fisher_accumulators(
    accumulators: dict[str, list[torch.Tensor] | torch.Tensor],
    denominator: int,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    normalised: dict[str, list[torch.Tensor] | torch.Tensor] = {}
    for key, value in accumulators.items():
        if isinstance(value, list):
            normalised[key] = [item / max(denominator, 1) for item in value]
        else:
            normalised[key] = value / max(denominator, 1)
    return normalised


def _aggregate_weight_fisher_to_neurons(
    model: MaskedMLP,
    fisher_accumulators: dict[str, list[torch.Tensor] | torch.Tensor],
    parts: Literal["incoming", "outgoing", "both"],
) -> dict[str, torch.Tensor]:
    hidden_weight_fisher = fisher_accumulators["hidden_weight"]
    hidden_bias_fisher = fisher_accumulators["hidden_bias"]
    output_weight_fisher = fisher_accumulators["output_weight"]
    if not isinstance(hidden_weight_fisher, list) or not isinstance(hidden_bias_fisher, list):
        raise TypeError("Invalid Fisher accumulator structure")
    if not isinstance(output_weight_fisher, torch.Tensor):
        raise TypeError("Invalid Fisher accumulator structure")

    scores: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for layer_index, layer_name in enumerate(model.layer_names):
            layer = model.hidden_layers[layer_index]
            incoming = (layer.weight.detach().cpu().square() * hidden_weight_fisher[layer_index]).sum(dim=1)
            incoming = incoming + layer.bias.detach().cpu().square() * hidden_bias_fisher[layer_index]

            if layer_index + 1 < len(model.hidden_layers):
                next_layer = model.hidden_layers[layer_index + 1]
                next_weight_fisher = hidden_weight_fisher[layer_index + 1]
                outgoing = (next_layer.weight.detach().cpu().square() * next_weight_fisher).sum(dim=0)
            else:
                outgoing = (model.output_layer.weight.detach().cpu().square() * output_weight_fisher).sum(dim=0)

            if parts == "incoming":
                score = incoming
            elif parts == "outgoing":
                score = outgoing
            elif parts == "both":
                score = incoming + outgoing
            else:
                raise ValueError(f"Unsupported weight_fisher_parts: {parts}")
            scores[layer_name] = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def weight_fisher_scores(
    model: MaskedMLP,
    loader: DataLoader,
    device: torch.device,
    mode: Literal["batch", "per_sample"] = "batch",
    parts: Literal["incoming", "outgoing", "both"] = "both",
) -> tuple[dict[str, torch.Tensor], ScoreCost]:
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()

    accumulators = _empty_weight_fisher_accumulators(model)
    n_seen = 0
    num_forward_passes = 0
    num_backward_passes = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if mode == "batch":
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            loss.backward()
            batch_size = int(targets.size(0))
            _accumulate_weight_grad_squares(model, accumulators, weight=float(batch_size))
            n_seen += batch_size
            num_forward_passes += 1
            num_backward_passes += 1
        elif mode == "per_sample":
            for sample_index in range(int(targets.size(0))):
                model.zero_grad(set_to_none=True)
                logits = model(inputs[sample_index : sample_index + 1])
                loss = F.cross_entropy(logits, targets[sample_index : sample_index + 1], reduction="sum")
                loss.backward()
                _accumulate_weight_grad_squares(model, accumulators, weight=1.0)
                n_seen += 1
                num_forward_passes += 1
                num_backward_passes += 1
        else:
            raise ValueError(f"Unsupported weight Fisher mode: {mode}")

    model.zero_grad(set_to_none=True)
    normalised = _normalise_weight_fisher_accumulators(accumulators, n_seen)
    scores = _aggregate_weight_fisher_to_neurons(model, normalised, parts=parts)
    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    memory_mb = current_memory_mb(device)
    model.train(was_training)

    return scores, ScoreCost(
        scoring_time_seconds=elapsed,
        scoring_memory_mb=memory_mb,
        num_forward_passes=num_forward_passes,
        num_backward_passes=num_backward_passes,
        calibration_size=n_seen,
    )


def single_neuron_ablation_scores(
    model: MaskedMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    metric: Literal["loss", "accuracy"] = "loss",
) -> tuple[dict[str, torch.Tensor], ScoreCost]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()
    baseline = evaluate(model, loader, loss_fn, device, num_classes=num_classes)
    scores: dict[str, torch.Tensor] = {}
    num_forward_passes = int(baseline["num_forward_passes"])

    for layer_name, layer_dim in model.layer_dims.items():
        layer_scores = torch.zeros(layer_dim, dtype=torch.float32)
        for neuron_index in range(layer_dim):
            mask = torch.ones(layer_dim, dtype=torch.float32)
            mask[neuron_index] = 0.0
            result = evaluate(model, loader, loss_fn, device, masks={layer_name: mask}, num_classes=num_classes)
            num_forward_passes += int(result["num_forward_passes"])
            if metric == "loss":
                layer_scores[neuron_index] = float(result["loss"] - baseline["loss"])
            elif metric == "accuracy":
                layer_scores[neuron_index] = float(baseline["accuracy"] - result["accuracy"])
            else:
                raise ValueError(f"Unsupported ablation metric: {metric}")
        scores[layer_name] = layer_scores

    synchronize_if_needed(device)
    elapsed = time.perf_counter() - start_time
    memory_mb = current_memory_mb(device)
    return scores, ScoreCost(
        scoring_time_seconds=elapsed,
        scoring_memory_mb=memory_mb,
        num_forward_passes=num_forward_passes,
        num_backward_passes=0,
        calibration_size=len(loader.dataset),
    )


def random_scores(
    model: MaskedMLP,
    seed: int,
    draw_index: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 * (draw_index + 1))
    del device
    return {
        layer_name: torch.rand(layer_dim, generator=generator)
        for layer_name, layer_dim in model.layer_dims.items()
    }


def masks_from_scores(
    model: MaskedMLP,
    scores: dict[str, torch.Tensor],
    pruning_percentage: float,
    target_layer: str | None = None,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= pruning_percentage < 1.0:
        raise ValueError(f"pruning_percentage must be in [0, 1), got {pruning_percentage}")

    masks: dict[str, torch.Tensor] = {}
    for layer_name, layer_dim in model.layer_dims.items():
        mask = torch.ones(layer_dim, dtype=torch.float32)
        should_prune_layer = target_layer is None or target_layer == layer_name
        if should_prune_layer and pruning_percentage > 0:
            num_pruned = min(int(layer_dim * pruning_percentage), layer_dim - 1)
            if num_pruned > 0:
                layer_scores = torch.as_tensor(scores[layer_name]).detach().cpu().float()
                prune_indices = torch.argsort(layer_scores)[:num_pruned]
                mask[prune_indices] = 0.0
        masks[layer_name] = mask
    return masks


def keep_indices_from_masks(model: MaskedMLP, masks: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    keep_indices = []
    for layer_name, layer_dim in model.layer_dims.items():
        if layer_name in masks:
            keep = torch.where(torch.as_tensor(masks[layer_name]).detach().cpu().float() > 0)[0]
        else:
            keep = torch.arange(layer_dim)
        if keep.numel() == 0:
            raise ValueError(f"Layer {layer_name} has no remaining neurons")
        keep_indices.append(keep)
    return keep_indices


def parameter_count_from_masks(model: MaskedMLP, masks: dict[str, torch.Tensor]) -> int:
    keep_indices = keep_indices_from_masks(model, masks)
    total_params = 0
    previous_dim = model.input_dim
    for keep in keep_indices:
        current_dim = int(keep.numel())
        total_params += previous_dim * current_dim + current_dim
        previous_dim = current_dim
    total_params += previous_dim * model.num_classes + model.num_classes
    return total_params


def estimated_macs_from_masks(model: MaskedMLP, masks: dict[str, torch.Tensor]) -> int:
    keep_indices = keep_indices_from_masks(model, masks)
    total_macs = 0
    previous_dim = model.input_dim
    for keep in keep_indices:
        current_dim = int(keep.numel())
        total_macs += previous_dim * current_dim
        previous_dim = current_dim
    total_macs += previous_dim * model.num_classes
    return total_macs


def materialize_pruned_model(
    model: MaskedMLP,
    masks: dict[str, torch.Tensor],
    device: torch.device,
) -> MaskedMLP:
    keep_indices = keep_indices_from_masks(model, masks)
    compact_hidden_dims = tuple(int(keep.numel()) for keep in keep_indices)
    compact = MaskedMLP(
        input_dim=model.input_dim,
        hidden_dims=compact_hidden_dims,
        num_classes=model.num_classes,
        dropout=model.dropout_p,
    ).to(device)

    with torch.no_grad():
        previous_keep = None
        for layer_index, keep in enumerate(keep_indices):
            keep = keep.to(model.hidden_layers[layer_index].weight.device)
            source_layer = model.hidden_layers[layer_index]
            target_layer = compact.hidden_layers[layer_index]
            if previous_keep is None:
                target_layer.weight.copy_(source_layer.weight[keep, :])
            else:
                previous_keep = previous_keep.to(source_layer.weight.device)
                target_layer.weight.copy_(source_layer.weight[keep][:, previous_keep])
            target_layer.bias.copy_(source_layer.bias[keep])
            previous_keep = keep.detach().cpu()

        last_keep = keep_indices[-1].to(model.output_layer.weight.device)
        compact.output_layer.weight.copy_(model.output_layer.weight[:, last_keep])
        compact.output_layer.bias.copy_(model.output_layer.bias)

    return compact


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_dense_macs(model: MaskedMLP) -> int:
    total_macs = 0
    previous_dim = model.input_dim
    for hidden_dim in model.hidden_dims:
        total_macs += previous_dim * hidden_dim
        previous_dim = hidden_dim
    total_macs += previous_dim * model.num_classes
    return total_macs


def serialise_cost(cost: ScoreCost) -> dict[str, float | int]:
    return asdict(cost)


def run_pruning_sweep(
    base_model: MaskedMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: ProtocolConfig,
    metadata: DatasetMetadata,
    dataset_name: str,
    architecture: str,
    seed: int,
    epoch: int,
    scores_by_criterion: dict[str, dict[str, torch.Tensor]],
    costs_by_criterion: dict[str, ScoreCost],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original_parameter_count = model_parameter_count(base_model)
    original_macs = model_dense_macs(base_model)
    requested_fine_tune_epochs = sorted(set(int(value) for value in config.fine_tune_epochs))
    max_fine_tune_epochs = max(requested_fine_tune_epochs, default=0)

    criterion_names = list(config.criteria)
    total_settings = 0
    for criterion_name in criterion_names:
        draw_count = config.random_trials if criterion_name == "random" else 1
        for prune_scope in config.prune_scopes:
            if prune_scope == "all":
                target_count = 1
            elif prune_scope == "layerwise":
                target_count = len(base_model.layer_names)
            else:
                target_count = 0
            total_settings += draw_count * target_count * len(config.pruning_percentages)

    completed_settings = 0
    sweep_start = time.perf_counter()
    print(
        f"{dataset_name}/{architecture}/seed={seed} epoch {epoch:03d}: "
        f"pruning sweep start ({total_settings} pruning settings, "
        f"fine_tune_epochs={requested_fine_tune_epochs})",
        flush=True,
    )

    for criterion_name in criterion_names:
        if criterion_name == "random":
            draw_count = config.random_trials
        else:
            draw_count = 1

        for draw_index in range(draw_count):
            if criterion_name == "random":
                scores = random_scores(base_model, seed=seed + epoch * 100, draw_index=draw_index)
                score_cost = ScoreCost()
            else:
                if criterion_name not in scores_by_criterion:
                    continue
                scores = scores_by_criterion[criterion_name]
                score_cost = costs_by_criterion.get(criterion_name, ScoreCost())

            for prune_scope in config.prune_scopes:
                if prune_scope == "all":
                    target_layers: list[str | None] = [None]
                elif prune_scope == "layerwise":
                    target_layers = list(base_model.layer_names)
                else:
                    raise ValueError(f"Unsupported prune scope: {prune_scope}")

                for target_layer in target_layers:
                    layer_label = "all" if target_layer is None else target_layer
                    for pruning_percentage in config.pruning_percentages:
                        setting_start = time.perf_counter()
                        masks = masks_from_scores(
                            base_model,
                            scores=scores,
                            pruning_percentage=float(pruning_percentage),
                            target_layer=target_layer,
                        )
                        pruned_parameter_count = parameter_count_from_masks(base_model, masks)
                        estimated_macs = estimated_macs_from_masks(base_model, masks)
                        parameter_reduction = 1.0 - (pruned_parameter_count / max(original_parameter_count, 1))
                        mac_reduction = 1.0 - (estimated_macs / max(original_macs, 1))

                        active_model = base_model
                        active_masks = masks

                        if 0 in requested_fine_tune_epochs:
                            rows.append(
                                pruning_result_row(
                                    model=active_model,
                                    masks=active_masks,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    loss_fn=loss_fn,
                                    device=device,
                                    metadata=metadata,
                                    dataset_name=dataset_name,
                                    architecture=architecture,
                                    seed=seed,
                                    epoch=epoch,
                                    criterion_name=criterion_name,
                                    random_draw=draw_index if criterion_name == "random" else None,
                                    pruning_percentage=float(pruning_percentage),
                                    fine_tune_epochs=0,
                                    layer_label=layer_label,
                                    prune_scope=prune_scope,
                                    pruned_parameter_count=pruned_parameter_count,
                                    parameter_reduction=parameter_reduction,
                                    estimated_macs=estimated_macs,
                                    mac_reduction=mac_reduction,
                                    original_parameter_count=original_parameter_count,
                                    original_macs=original_macs,
                                    score_cost=score_cost,
                                    config=config,
                                )
                            )

                        if max_fine_tune_epochs > 0:
                            fine_tune_model = copy.deepcopy(base_model).to(device)
                            fine_tune_optimizer = make_optimizer(
                                fine_tune_model,
                                config,
                                learning_rate=config.fine_tune_learning_rate,
                            )
                            for fine_tune_epoch in range(1, max_fine_tune_epochs + 1):
                                train_one_epoch(
                                    fine_tune_model,
                                    train_loader,
                                    loss_fn,
                                    fine_tune_optimizer,
                                    device,
                                    masks=masks,
                                )
                                if fine_tune_epoch in requested_fine_tune_epochs:
                                    rows.append(
                                        pruning_result_row(
                                            model=fine_tune_model,
                                            masks=masks,
                                            val_loader=val_loader,
                                            test_loader=test_loader,
                                            loss_fn=loss_fn,
                                            device=device,
                                            metadata=metadata,
                                            dataset_name=dataset_name,
                                            architecture=architecture,
                                            seed=seed,
                                            epoch=epoch,
                                            criterion_name=criterion_name,
                                            random_draw=draw_index if criterion_name == "random" else None,
                                            pruning_percentage=float(pruning_percentage),
                                            fine_tune_epochs=fine_tune_epoch,
                                            layer_label=layer_label,
                                            prune_scope=prune_scope,
                                            pruned_parameter_count=pruned_parameter_count,
                                            parameter_reduction=parameter_reduction,
                                            estimated_macs=estimated_macs,
                                            mac_reduction=mac_reduction,
                                            original_parameter_count=original_parameter_count,
                                            original_macs=original_macs,
                                            score_cost=score_cost,
                                            config=config,
                                        )
                                    )
                        completed_settings += 1
                        elapsed = time.perf_counter() - sweep_start
                        setting_elapsed = time.perf_counter() - setting_start
                        random_suffix = f"/draw={draw_index}" if criterion_name == "random" else ""
                        print(
                            f"{dataset_name}/{architecture}/seed={seed} epoch {epoch:03d}: "
                            f"pruning {completed_settings}/{total_settings} "
                            f"{criterion_name}{random_suffix} scope={prune_scope} layer={layer_label} "
                            f"p={pruning_percentage:.2f} done in {setting_elapsed:.1f}s "
                            f"(elapsed {elapsed / 60:.1f}m)",
                            flush=True,
                        )
    print(
        f"{dataset_name}/{architecture}/seed={seed} epoch {epoch:03d}: "
        f"pruning sweep done ({len(rows)} result rows, elapsed {(time.perf_counter() - sweep_start) / 60:.1f}m)",
        flush=True,
    )
    return rows


def pruning_result_row(
    model: MaskedMLP,
    masks: dict[str, torch.Tensor],
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metadata: DatasetMetadata,
    dataset_name: str,
    architecture: str,
    seed: int,
    epoch: int,
    criterion_name: str,
    random_draw: int | None,
    pruning_percentage: float,
    fine_tune_epochs: int,
    layer_label: str,
    prune_scope: str,
    pruned_parameter_count: int,
    parameter_reduction: float,
    estimated_macs: int,
    mac_reduction: float,
    original_parameter_count: int,
    original_macs: int,
    score_cost: ScoreCost,
    config: ProtocolConfig,
) -> dict[str, Any]:
    val_metrics = evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        masks=masks,
        num_classes=metadata.num_classes,
        per_class=True,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        masks=masks,
        num_classes=metadata.num_classes,
        per_class=True,
        measure_time=True,
    )

    compact_test_accuracy = ""
    compact_test_loss = ""
    compact_inference_time_seconds = ""
    if config.measure_compact_inference:
        compact_model = materialize_pruned_model(model, masks, device=device)
        compact_metrics = evaluate(
            compact_model,
            test_loader,
            loss_fn,
            device,
            num_classes=metadata.num_classes,
            measure_time=True,
        )
        compact_test_accuracy = compact_metrics["accuracy"]
        compact_test_loss = compact_metrics["loss"]
        compact_inference_time_seconds = compact_metrics["inference_time_seconds"]

    cost_values = serialise_cost(score_cost)
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "architecture": architecture,
        "seed": seed,
        "epoch": epoch,
        "layer": layer_label,
        "prune_scope": prune_scope,
        "criterion": criterion_name,
        "criterion_group": criterion_group(criterion_name),
        "random_draw": "" if random_draw is None else random_draw,
        "pruning_percentage": pruning_percentage,
        "fine_tune_epochs": fine_tune_epochs,
        "validation_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "validation_loss": val_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "validation_per_class_accuracy_json": json.dumps(val_metrics.get("per_class_accuracy", [])),
        "test_per_class_accuracy_json": json.dumps(test_metrics.get("per_class_accuracy", [])),
        "parameter_count": pruned_parameter_count,
        "original_parameter_count": original_parameter_count,
        "parameter_reduction": parameter_reduction,
        "estimated_macs": estimated_macs,
        "original_macs": original_macs,
        "mac_reduction": mac_reduction,
        "masked_inference_time_seconds": test_metrics["inference_time_seconds"],
        "compact_test_accuracy": compact_test_accuracy,
        "compact_test_loss": compact_test_loss,
        "compact_inference_time_seconds": compact_inference_time_seconds,
        "calibration_size": score_cost.calibration_size,
        "weight_fisher_mode": config.weight_fisher_mode,
        "weight_fisher_parts": config.weight_fisher_parts,
    }
    row.update(cost_values)
    return row


def criterion_group(criterion_name: str) -> str:
    if criterion_name == "random":
        return "baseline"
    if criterion_name in {"output_weight_norm", "mean_activation", "firing_rate", "activation_variance"}:
        return "baseline"
    if criterion_name in {"class_selectivity", "redundancy", "hebbian"}:
        return "loss_free"
    if criterion_name in LOSS_AWARE_CRITERIA:
        return "loss_aware"
    if criterion_name in REFERENCE_CRITERIA:
        return "reference"
    return "unknown"


def resolve_scoring_epochs(total_epochs: int, scoring_epochs: Iterable[int | str]) -> list[int]:
    resolved = []
    for value in scoring_epochs:
        if isinstance(value, str):
            if value.lower() != "last":
                raise ValueError(f"Unsupported scoring epoch token: {value}")
            epoch = total_epochs
        else:
            epoch = int(value)
        if epoch < 1 or epoch > total_epochs:
            raise ValueError(f"Scoring epoch {epoch} outside training range 1..{total_epochs}")
        resolved.append(epoch)
    return sorted(set(resolved))


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"


def save_checkpoint(
    path: Path,
    model: MaskedMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metadata: DatasetMetadata,
    architecture: str,
    config: ProtocolConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": asdict(metadata),
            "architecture": architecture,
            "hidden_dims": model.hidden_dims,
            "config": asdict(config),
        },
        path,
    )


def load_model_checkpoint(
    path: Path,
    metadata: DatasetMetadata,
    architecture: str,
    config: ProtocolConfig,
    device: torch.device,
) -> MaskedMLP:
    checkpoint = torch.load(path, map_location=device)
    model = make_model(metadata, architecture, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_order = [
        "dataset",
        "architecture",
        "seed",
        "epoch",
        "layer",
        "prune_scope",
        "criterion",
        "criterion_group",
        "random_draw",
        "pruning_percentage",
        "fine_tune_epochs",
        "train_loss",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "validation_loss",
        "test_loss",
        "parameter_count",
        "parameter_reduction",
        "estimated_macs",
        "mac_reduction",
        "scoring_time_seconds",
        "scoring_memory_mb",
        "num_forward_passes",
        "num_backward_passes",
        "calibration_size",
    ]
    fieldnames = []
    for field_name in preferred_order:
        if any(field_name in row for row in rows):
            fieldnames.append(field_name)
    extra_fieldnames = sorted({key for row in rows for key in row.keys()} - set(fieldnames))
    fieldnames.extend(extra_fieldnames)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def score_rows(
    scores_by_criterion: dict[str, dict[str, torch.Tensor]],
    dataset_name: str,
    architecture: str,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    rows = []
    for criterion_name, layer_scores in scores_by_criterion.items():
        for layer_name, scores in layer_scores.items():
            for neuron_index, score in enumerate(torch.as_tensor(scores).detach().cpu().tolist()):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "architecture": architecture,
                        "seed": seed,
                        "epoch": epoch,
                        "criterion": criterion_name,
                        "criterion_group": criterion_group(criterion_name),
                        "layer": layer_name,
                        "neuron_index": neuron_index,
                        "score": float(score),
                    }
                )
    return rows


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    return ranks


def spearman_corr_np(left: torch.Tensor, right: torch.Tensor) -> float:
    left_values = torch.as_tensor(left).detach().cpu().numpy().astype(np.float64)
    right_values = torch.as_tensor(right).detach().cpu().numpy().astype(np.float64)
    left_ranks = rank_values(left_values)
    right_ranks = rank_values(right_values)
    left_ranks = left_ranks - left_ranks.mean()
    right_ranks = right_ranks - right_ranks.mean()
    denominator = math.sqrt(float((left_ranks**2).sum() * (right_ranks**2).sum()))
    if denominator <= 1e-12:
        return float("nan")
    return float((left_ranks * right_ranks).sum() / denominator)


def bottom_k_indices(scores: torch.Tensor, ratio: float) -> set[int]:
    tensor_scores = torch.as_tensor(scores).detach().cpu().float()
    top_count = max(1, int(len(tensor_scores) * ratio))
    return set(torch.argsort(tensor_scores)[:top_count].tolist())


def jaccard_index(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def compute_ranking_stability_rows(
    epoch_scores: dict[int, dict[str, dict[str, torch.Tensor]]],
    ratios: Iterable[float],
    dataset_name: str,
    architecture: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sorted_epochs = sorted(epoch_scores)
    if len(sorted_epochs) < 2:
        return rows

    for previous_epoch, current_epoch in zip(sorted_epochs[:-1], sorted_epochs[1:], strict=True):
        previous_scores = epoch_scores[previous_epoch]
        current_scores = epoch_scores[current_epoch]
        common_criteria = sorted(set(previous_scores) & set(current_scores))
        for criterion_name in common_criteria:
            common_layers = sorted(set(previous_scores[criterion_name]) & set(current_scores[criterion_name]))
            for layer_name in common_layers:
                previous_layer_scores = previous_scores[criterion_name][layer_name]
                current_layer_scores = current_scores[criterion_name][layer_name]
                spearman = spearman_corr_np(previous_layer_scores, current_layer_scores)
                for ratio in ratios:
                    previous_set = bottom_k_indices(previous_layer_scores, ratio)
                    current_set = bottom_k_indices(current_layer_scores, ratio)
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "architecture": architecture,
                            "seed": seed,
                            "previous_epoch": previous_epoch,
                            "epoch": current_epoch,
                            "criterion": criterion_name,
                            "criterion_group": criterion_group(criterion_name),
                            "layer": layer_name,
                            "topk_ratio": float(ratio),
                            "spearman": spearman,
                            "topk_jaccard": jaccard_index(previous_set, current_set),
                        }
                    )
    return rows


def compute_rank_agreement_rows(
    scores_by_criterion: dict[str, dict[str, torch.Tensor]],
    reference_criterion: str,
    dataset_name: str,
    architecture: str,
    seed: int,
    epoch: int,
    ratios: Iterable[float],
) -> list[dict[str, Any]]:
    if reference_criterion not in scores_by_criterion:
        return []
    rows: list[dict[str, Any]] = []
    reference_scores = scores_by_criterion[reference_criterion]

    for criterion_name, criterion_scores in scores_by_criterion.items():
        if criterion_name == reference_criterion:
            continue
        for layer_name in sorted(set(reference_scores) & set(criterion_scores)):
            reference_layer_scores = reference_scores[layer_name]
            criterion_layer_scores = criterion_scores[layer_name]
            spearman = spearman_corr_np(reference_layer_scores, criterion_layer_scores)
            for ratio in ratios:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "architecture": architecture,
                        "seed": seed,
                        "epoch": epoch,
                        "reference_criterion": reference_criterion,
                        "criterion": criterion_name,
                        "criterion_group": criterion_group(criterion_name),
                        "layer": layer_name,
                        "topk_ratio": float(ratio),
                        "spearman": spearman,
                        "topk_jaccard": jaccard_index(
                            bottom_k_indices(reference_layer_scores, ratio),
                            bottom_k_indices(criterion_layer_scores, ratio),
                        ),
                    }
                )
    return rows


def criterion_scores_for_epoch(
    model: MaskedMLP,
    stats: dict[str, torch.Tensor | int],
    stats_cost: ScoreCost,
    calibration_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: ProtocolConfig,
    metadata: DatasetMetadata,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, ScoreCost]]:
    enriched_stats = enrich_stats_with_model_weights(stats, model)
    scores_by_criterion: dict[str, dict[str, torch.Tensor]] = {}
    costs_by_criterion: dict[str, ScoreCost] = {}

    for criterion_name in config.criteria:
        if criterion_name in LOSS_FREE_CRITERIA:
            scores_by_criterion[criterion_name] = score_from_stats(enriched_stats, model, criterion_name)
            costs_by_criterion[criterion_name] = stats_cost

    if "activation_fisher" in config.criteria:
        print("  scoring activation_fisher start", flush=True)
        scores, cost = activation_fisher_scores(model, calibration_loader, device)
        scores_by_criterion["activation_fisher"] = scores
        costs_by_criterion["activation_fisher"] = cost
        print(
            f"  scoring activation_fisher done in {cost.scoring_time_seconds:.1f}s "
            f"({cost.num_forward_passes} forward, {cost.num_backward_passes} backward)",
            flush=True,
        )

    if "weight_fisher" in config.criteria:
        print(
            f"  scoring weight_fisher start "
            f"(mode={config.weight_fisher_mode}, parts={config.weight_fisher_parts})",
            flush=True,
        )
        scores, cost = weight_fisher_scores(
            model,
            calibration_loader,
            device,
            mode=config.weight_fisher_mode,
            parts=config.weight_fisher_parts,
        )
        scores_by_criterion["weight_fisher"] = scores
        costs_by_criterion["weight_fisher"] = cost
        print(
            f"  scoring weight_fisher done in {cost.scoring_time_seconds:.1f}s "
            f"({cost.num_forward_passes} forward, {cost.num_backward_passes} backward)",
            flush=True,
        )

    if "single_neuron_ablation" in config.criteria:
        print("  scoring single_neuron_ablation start", flush=True)
        scores, cost = single_neuron_ablation_scores(
            model,
            val_loader,
            loss_fn,
            device,
            num_classes=metadata.num_classes,
            metric=config.ablation_reference_metric,
        )
        scores_by_criterion["single_neuron_ablation"] = scores
        costs_by_criterion["single_neuron_ablation"] = cost
        print(
            f"  scoring single_neuron_ablation done in {cost.scoring_time_seconds:.1f}s "
            f"({cost.num_forward_passes} forward)",
            flush=True,
        )

    return scores_by_criterion, costs_by_criterion


def subset_loader_from_loader(loader: DataLoader, subset_size: int, config: ProtocolConfig) -> DataLoader:
    subset_size = min(int(subset_size), len(loader.dataset))
    subset = Subset(loader.dataset, list(range(subset_size)))
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def run_calibration_size_sensitivity(
    model: MaskedMLP,
    full_scores: dict[str, dict[str, torch.Tensor]],
    calibration_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: ProtocolConfig,
    metadata: DatasetMetadata,
    dataset_name: str,
    architecture: str,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not config.calibration_sensitivity_sizes:
        return rows

    original_criteria = config.criteria
    sensitivity_criteria = tuple(
        criterion_name
        for criterion_name in config.calibration_sensitivity_criteria
        if criterion_name in original_criteria and criterion_name != "random"
    )

    for calibration_size in config.calibration_sensitivity_sizes:
        print(
            f"{dataset_name}/{architecture}/seed={seed} epoch {epoch:03d}: "
            f"calibration sensitivity size={calibration_size} start",
            flush=True,
        )
        loader = subset_loader_from_loader(calibration_loader, calibration_size, config)
        stats, stats_cost = collect_activation_stats(
            model,
            loader,
            device,
            num_classes=metadata.num_classes,
            threshold=config.activation_threshold,
        )
        sensitivity_config = copy.copy(config)
        sensitivity_config.criteria = sensitivity_criteria
        scores_by_criterion, costs_by_criterion = criterion_scores_for_epoch(
            model,
            stats,
            stats_cost,
            loader,
            val_loader,
            loss_fn,
            device,
            sensitivity_config,
            metadata,
        )

        for criterion_name, criterion_scores in scores_by_criterion.items():
            if criterion_name not in full_scores:
                continue
            for layer_name in sorted(set(criterion_scores) & set(full_scores[criterion_name])):
                row_base = {
                    "dataset": dataset_name,
                    "architecture": architecture,
                    "seed": seed,
                    "epoch": epoch,
                    "criterion": criterion_name,
                    "criterion_group": criterion_group(criterion_name),
                    "layer": layer_name,
                    "calibration_size": calibration_size,
                    "reference_calibration_size": config.calibration_size,
                    "spearman_vs_full": spearman_corr_np(
                        full_scores[criterion_name][layer_name],
                        criterion_scores[layer_name],
                    ),
                }
                for ratio in config.stability_topk_ratios:
                    row = dict(row_base)
                    row["topk_ratio"] = float(ratio)
                    row["topk_jaccard_vs_full"] = jaccard_index(
                        bottom_k_indices(full_scores[criterion_name][layer_name], ratio),
                        bottom_k_indices(criterion_scores[layer_name], ratio),
                    )
                    row.update(serialise_cost(costs_by_criterion.get(criterion_name, ScoreCost())))
                    rows.append(row)
        print(
            f"{dataset_name}/{architecture}/seed={seed} epoch {epoch:03d}: "
            f"calibration sensitivity size={calibration_size} done",
            flush=True,
        )
    return rows


def run_single_experiment(
    config: ProtocolConfig,
    dataset_name: str,
    architecture: str,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    set_global_seed(seed)
    device = resolve_device(config.device)
    bundle = build_data_bundle(dataset_name, config, seed=seed)
    model = make_model(bundle.metadata, architecture, config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, config)

    run_dir = Path(config.run_root) / bundle.metadata.name / architecture / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    scoring_epochs = resolve_scoring_epochs(config.epochs, config.scoring_epochs)
    needs_intermediate_checkpoints = any(epoch != config.epochs for epoch in scoring_epochs)
    should_save_checkpoints = config.save_checkpoints or needs_intermediate_checkpoints

    training_rows: list[dict[str, Any]] = []
    epoch_stats: dict[int, dict[str, torch.Tensor | int]] = {}
    epoch_stats_costs: dict[int, ScoreCost] = {}
    epoch_loss_free_scores: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = train_one_epoch(model, bundle.train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(
            model,
            bundle.val_loader,
            loss_fn,
            device,
            num_classes=bundle.metadata.num_classes,
        )
        test_metrics = evaluate(
            model,
            bundle.test_loader,
            loss_fn,
            device,
            num_classes=bundle.metadata.num_classes,
        )
        stats, stats_cost = collect_activation_stats(
            model,
            bundle.calibration_loader,
            device,
            num_classes=bundle.metadata.num_classes,
            threshold=config.activation_threshold,
        )
        enriched_stats = enrich_stats_with_model_weights(stats, model)
        loss_free_scores = {
            criterion_name: score_from_stats(enriched_stats, model, criterion_name)
            for criterion_name in LOSS_FREE_CRITERIA
            if criterion_name in config.criteria
        }

        elapsed = time.perf_counter() - start_time
        epoch_stats[epoch] = stats
        epoch_stats_costs[epoch] = stats_cost
        epoch_loss_free_scores[epoch] = loss_free_scores

        training_rows.append(
            {
                "dataset": bundle.metadata.name,
                "architecture": architecture,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "epoch_time_seconds": elapsed,
                "calibration_size": bundle.calibration_size,
            }
        )

        if should_save_checkpoints:
            save_checkpoint(
                checkpoint_path(run_dir, epoch),
                model,
                optimizer,
                epoch,
                bundle.metadata,
                architecture,
                config,
            )

        print(
            f"{bundle.metadata.name}/{architecture}/seed={seed} epoch {epoch:03d}: "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} time={elapsed:.1f}s"
        )

    write_rows_csv(run_dir / "training_history.csv", training_rows)
    write_rows_jsonl(run_dir / "training_history.jsonl", training_rows)

    stability_rows = compute_ranking_stability_rows(
        epoch_loss_free_scores,
        ratios=config.stability_topk_ratios,
        dataset_name=bundle.metadata.name,
        architecture=architecture,
        seed=seed,
    )
    write_rows_csv(run_dir / "ranking_stability.csv", stability_rows)

    all_pruning_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    all_rank_agreement_rows: list[dict[str, Any]] = []
    all_calibration_sensitivity_rows: list[dict[str, Any]] = []

    for scoring_epoch in scoring_epochs:
        print(
            f"{bundle.metadata.name}/{architecture}/seed={seed}: "
            f"scoring epoch {scoring_epoch:03d} start",
            flush=True,
        )
        if scoring_epoch == config.epochs:
            score_model = model
        else:
            score_model = load_model_checkpoint(
                checkpoint_path(run_dir, scoring_epoch),
                bundle.metadata,
                architecture,
                config,
                device,
            )

        scores_by_criterion, costs_by_criterion = criterion_scores_for_epoch(
            score_model,
            epoch_stats[scoring_epoch],
            epoch_stats_costs[scoring_epoch],
            bundle.calibration_loader,
            bundle.val_loader,
            loss_fn,
            device,
            config,
            bundle.metadata,
        )
        print(
            f"{bundle.metadata.name}/{architecture}/seed={seed}: "
            f"scoring epoch {scoring_epoch:03d} done ({len(scores_by_criterion)} criteria)",
            flush=True,
        )

        current_score_rows = score_rows(
            scores_by_criterion,
            dataset_name=bundle.metadata.name,
            architecture=architecture,
            seed=seed,
            epoch=scoring_epoch,
        )
        all_score_rows.extend(current_score_rows)
        write_rows_csv(run_dir / f"score_values_epoch_{scoring_epoch:03d}.csv", current_score_rows)

        rank_agreement_rows = compute_rank_agreement_rows(
            scores_by_criterion,
            reference_criterion="activation_fisher",
            dataset_name=bundle.metadata.name,
            architecture=architecture,
            seed=seed,
            epoch=scoring_epoch,
            ratios=config.stability_topk_ratios,
        )
        all_rank_agreement_rows.extend(rank_agreement_rows)
        write_rows_csv(run_dir / f"rank_agreement_epoch_{scoring_epoch:03d}.csv", rank_agreement_rows)

        calibration_sensitivity_rows = run_calibration_size_sensitivity(
            score_model,
            scores_by_criterion,
            bundle.calibration_loader,
            bundle.val_loader,
            loss_fn,
            device,
            config,
            bundle.metadata,
            bundle.metadata.name,
            architecture,
            seed,
            scoring_epoch,
        )
        all_calibration_sensitivity_rows.extend(calibration_sensitivity_rows)
        write_rows_csv(
            run_dir / f"calibration_sensitivity_epoch_{scoring_epoch:03d}.csv",
            calibration_sensitivity_rows,
        )

        pruning_rows = run_pruning_sweep(
            base_model=score_model,
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            test_loader=bundle.test_loader,
            loss_fn=loss_fn,
            device=device,
            config=config,
            metadata=bundle.metadata,
            dataset_name=bundle.metadata.name,
            architecture=architecture,
            seed=seed,
            epoch=scoring_epoch,
            scores_by_criterion=scores_by_criterion,
            costs_by_criterion=costs_by_criterion,
        )
        all_pruning_rows.extend(pruning_rows)
        write_rows_csv(run_dir / f"pruning_results_epoch_{scoring_epoch:03d}.csv", pruning_rows)
        write_rows_jsonl(run_dir / f"pruning_results_epoch_{scoring_epoch:03d}.jsonl", pruning_rows)
        print(
            f"{bundle.metadata.name}/{architecture}/seed={seed}: "
            f"epoch {scoring_epoch:03d} result files written to {run_dir}",
            flush=True,
        )

    write_rows_csv(run_dir / "pruning_results.csv", all_pruning_rows)
    write_rows_csv(run_dir / "score_values.csv", all_score_rows)
    write_rows_csv(run_dir / "rank_agreement.csv", all_rank_agreement_rows)
    write_rows_csv(run_dir / "calibration_sensitivity.csv", all_calibration_sensitivity_rows)

    return {
        "training_rows": training_rows,
        "stability_rows": stability_rows,
        "pruning_rows": all_pruning_rows,
        "score_rows": all_score_rows,
        "rank_agreement_rows": all_rank_agreement_rows,
        "calibration_sensitivity_rows": all_calibration_sensitivity_rows,
    }


def run_protocol(config: ProtocolConfig) -> dict[str, list[dict[str, Any]]]:
    aggregate: dict[str, list[dict[str, Any]]] = {
        "training_rows": [],
        "stability_rows": [],
        "pruning_rows": [],
        "score_rows": [],
        "rank_agreement_rows": [],
        "calibration_sensitivity_rows": [],
    }

    for dataset_name in config.dataset_names:
        metadata = dataset_metadata(dataset_name)
        for architecture in config.architectures:
            if architecture == "cifar_flat" and metadata.name != "CIFAR10":
                continue
            for seed in config.seeds:
                print(
                    f"run_protocol: start dataset={metadata.name} architecture={architecture} seed={seed}",
                    flush=True,
                )
                result = run_single_experiment(config, dataset_name, architecture, seed)
                for key, rows in result.items():
                    aggregate[key].extend(rows)
                print(
                    f"run_protocol: done dataset={metadata.name} architecture={architecture} seed={seed}",
                    flush=True,
                )

    run_root = Path(config.run_root)
    write_rows_csv(run_root / "training_history_all.csv", aggregate["training_rows"])
    write_rows_csv(run_root / "ranking_stability_all.csv", aggregate["stability_rows"])
    write_rows_csv(run_root / "pruning_results_all.csv", aggregate["pruning_rows"])
    write_rows_csv(run_root / "score_values_all.csv", aggregate["score_rows"])
    write_rows_csv(run_root / "rank_agreement_all.csv", aggregate["rank_agreement_rows"])
    write_rows_csv(run_root / "calibration_sensitivity_all.csv", aggregate["calibration_sensitivity_rows"])
    return aggregate


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def group_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row.get(key, "") for key in keys)
        grouped.setdefault(group_key, []).append(row)
    return grouped


def mean_by_x(
    rows: Iterable[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = group_rows(rows, (x_key,))
    x_values = []
    y_values = []
    for group_key, group_rows_value in sorted(grouped.items(), key=lambda item: float(item[0][0])):
        values = [to_float(row, y_key) for row in group_rows_value]
        values = [value for value in values if not math.isnan(value)]
        if values:
            x_values.append(float(group_key[0]))
            y_values.append(float(np.mean(values)))
    return np.array(x_values), np.array(y_values)


def plot_accuracy_vs_pruning(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    layer: str = "all",
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and str(row.get("layer")) == layer
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    grouped = group_rows(filtered, ("criterion",))

    plt.figure(figsize=(8, 5))
    for (criterion_name,), rows in sorted(grouped.items()):
        x_values, y_values = mean_by_x(rows, "pruning_percentage", metric)
        if len(x_values):
            plt.plot(x_values * 100.0, y_values, marker="o", label=criterion_name)
    plt.xlabel("% neurons pruned")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"{dataset} {architecture} {layer}: accuracy vs pruning")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_accuracy_drop_vs_pruning(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    layer: str = "all",
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and str(row.get("layer")) == layer
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    grouped = group_rows(filtered, ("criterion",))

    plt.figure(figsize=(8, 5))
    for (criterion_name,), rows in sorted(grouped.items()):
        x_values, y_values = mean_by_x(rows, "pruning_percentage", metric)
        if len(x_values):
            baseline = y_values[x_values.argmin()]
            plt.plot(x_values * 100.0, baseline - y_values, marker="o", label=criterion_name)
    plt.xlabel("% neurons pruned")
    plt.ylabel(f"{metric.replace('_', ' ')} drop")
    plt.title(f"{dataset} {architecture} {layer}: accuracy drop")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_improvement_over_random(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    layer: str = "all",
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and str(row.get("layer")) == layer
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    random_rows = [row for row in filtered if row.get("criterion") == "random"]
    random_x, random_y = mean_by_x(random_rows, "pruning_percentage", metric)
    random_map = dict(zip(random_x.tolist(), random_y.tolist(), strict=True))
    grouped = group_rows([row for row in filtered if row.get("criterion") != "random"], ("criterion",))

    plt.figure(figsize=(8, 5))
    for (criterion_name,), rows in sorted(grouped.items()):
        x_values, y_values = mean_by_x(rows, "pruning_percentage", metric)
        improvement = np.array([value - random_map.get(float(x_value), np.nan) for x_value, value in zip(x_values, y_values, strict=True)])
        plt.plot(x_values * 100.0, improvement, marker="o", label=criterion_name)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("% neurons pruned")
    plt.ylabel(f"{metric.replace('_', ' ')} minus random")
    plt.title(f"{dataset} {architecture} {layer}: improvement over random")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_layer_sensitivity_heatmap(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    criterion: str,
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("criterion") == criterion
        and row.get("prune_scope") == "layerwise"
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    layers = sorted({row.get("layer", "") for row in rows})
    percentages = sorted({to_float(row, "pruning_percentage") for row in rows})
    matrix = np.full((len(layers), len(percentages)), np.nan)
    for layer_index, layer_name in enumerate(layers):
        for percent_index, percentage in enumerate(percentages):
            values = [
                to_float(row, metric)
                for row in rows
                if row.get("layer") == layer_name and to_float(row, "pruning_percentage") == percentage
            ]
            values = [value for value in values if not math.isnan(value)]
            if values:
                matrix[layer_index, percent_index] = np.mean(values)

    plt.figure(figsize=(8, max(3, len(layers) * 0.5)))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.xticks(range(len(percentages)), [f"{value * 100:.0f}" for value in percentages])
    plt.yticks(range(len(layers)), layers)
    plt.xlabel("% neurons pruned")
    plt.ylabel("Layer")
    plt.title(f"{dataset} {architecture}: layer sensitivity ({criterion})")
    plt.colorbar(label=metric.replace("_", " "))
    plt.show()


def plot_ranking_stability(stability_rows: list[dict[str, Any]], criterion: str, layer: str) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in stability_rows if row.get("criterion") == criterion and row.get("layer") == layer]
    grouped = group_rows(rows, ("topk_ratio",))
    plt.figure(figsize=(8, 4))
    for (ratio,), group in sorted(grouped.items(), key=lambda item: float(item[0][0])):
        x_values, y_values = mean_by_x(group, "epoch", "topk_jaccard")
        plt.plot(x_values, y_values, marker="o", label=f"top {float(ratio) * 100:.0f}%")
    spearman_x, spearman_y = mean_by_x(rows, "epoch", "spearman")
    if len(spearman_x):
        plt.plot(spearman_x, spearman_y, marker="s", linestyle="--", label="Spearman")
    plt.xlabel("Epoch")
    plt.ylabel("Agreement")
    plt.title(f"Ranking stability: {criterion} {layer}")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_criteria_correlation_matrix(
    score_rows_value: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    layer: str,
    epoch: int,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in score_rows_value
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("layer") == layer
        and int(float(row.get("epoch", 0))) == epoch
    ]
    criteria = sorted({row.get("criterion", "") for row in rows})
    by_criterion = {}
    for criterion_name in criteria:
        criterion_rows = sorted(
            [row for row in rows if row.get("criterion") == criterion_name],
            key=lambda row: int(float(row.get("neuron_index", 0))),
        )
        by_criterion[criterion_name] = np.array([to_float(row, "score") for row in criterion_rows])

    matrix = np.full((len(criteria), len(criteria)), np.nan)
    for left_index, left_name in enumerate(criteria):
        for right_index, right_name in enumerate(criteria):
            if len(by_criterion[left_name]) == len(by_criterion[right_name]) and len(by_criterion[left_name]) > 1:
                matrix[left_index, right_index] = spearman_corr_np(
                    torch.tensor(by_criterion[left_name]),
                    torch.tensor(by_criterion[right_name]),
                )

    plt.figure(figsize=(max(6, len(criteria) * 0.6), max(5, len(criteria) * 0.6)))
    plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(criteria)), criteria, rotation=45, ha="right")
    plt.yticks(range(len(criteria)), criteria)
    plt.colorbar(label="Spearman rank correlation")
    plt.title(f"{dataset} {architecture} {layer} epoch {epoch}: criterion agreement")
    plt.tight_layout()
    plt.show()


def plot_cost_benefit_pareto(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    pruning_percentage: float,
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("layer") == "all"
        and abs(to_float(row, "pruning_percentage") - pruning_percentage) < 1e-12
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    grouped = group_rows(rows, ("criterion",))
    plt.figure(figsize=(7, 5))
    for (criterion_name,), group in sorted(grouped.items()):
        x_values = [to_float(row, "scoring_time_seconds") for row in group]
        y_values = [to_float(row, metric) for row in group]
        if x_values and y_values:
            plt.scatter(np.mean(x_values), np.mean(y_values), label=criterion_name)
            plt.annotate(criterion_name, (np.mean(x_values), np.mean(y_values)), fontsize=8)
    plt.xlabel("Scoring time seconds")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"{dataset} {architecture}: cost-benefit at {pruning_percentage * 100:.0f}%")
    plt.grid(True)
    plt.show()


def plot_finetune_recovery(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    criterion: str,
    pruning_percentage: float,
    layer: str = "all",
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("criterion") == criterion
        and row.get("layer") == layer
        and abs(to_float(row, "pruning_percentage") - pruning_percentage) < 1e-12
    ]
    x_values, y_values = mean_by_x(rows, "fine_tune_epochs", metric)
    plt.figure(figsize=(7, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Fine-tuning epochs")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"{dataset} {architecture}: {criterion} recovery at {pruning_percentage * 100:.0f}%")
    plt.grid(True)
    plt.show()


def plot_score_distribution(
    score_rows_value: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    epoch: int,
    layer: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in score_rows_value
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and int(float(row.get("epoch", 0))) == epoch
        and row.get("layer") == layer
    ]
    grouped = group_rows(rows, ("criterion",))
    plt.figure(figsize=(8, 5))
    for (criterion_name,), group in sorted(grouped.items()):
        values = [to_float(row, "score") for row in group]
        values = [value for value in values if not math.isnan(value)]
        if values:
            plt.hist(values, bins=40, alpha=0.4, label=criterion_name)
    plt.xlabel("Score")
    plt.ylabel("Neuron count")
    plt.title(f"{dataset} {architecture} {layer}: score distributions")
    plt.legend()
    plt.show()


def plot_fisher_vs_lossfree_scatter(
    score_rows_value: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    epoch: int,
    layer: str,
    lossfree_criterion: str = "hebbian",
    fisher_criterion: str = "activation_fisher",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in score_rows_value
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and int(float(row.get("epoch", 0))) == epoch
        and row.get("layer") == layer
        and row.get("criterion") in {lossfree_criterion, fisher_criterion}
    ]
    by_criterion = {}
    for criterion_name in (lossfree_criterion, fisher_criterion):
        criterion_rows = sorted(
            [row for row in rows if row.get("criterion") == criterion_name],
            key=lambda row: int(float(row.get("neuron_index", 0))),
        )
        by_criterion[criterion_name] = [to_float(row, "score") for row in criterion_rows]
    if not by_criterion.get(lossfree_criterion) or not by_criterion.get(fisher_criterion):
        raise ValueError("Missing score rows for requested criteria")
    plt.figure(figsize=(6, 5))
    plt.scatter(by_criterion[lossfree_criterion], by_criterion[fisher_criterion], alpha=0.7)
    plt.xlabel(lossfree_criterion)
    plt.ylabel(fisher_criterion)
    plt.title(f"{dataset} {architecture} {layer}: Fisher vs loss-free")
    plt.grid(True)
    plt.show()


def plot_auc_damage_summary(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    layer: str = "all",
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("layer") == layer
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    grouped = group_rows(filtered, ("criterion",))
    labels = []
    damages = []
    for (criterion_name,), rows in sorted(grouped.items()):
        x_values, y_values = mean_by_x(rows, "pruning_percentage", metric)
        if len(x_values) < 2:
            continue
        baseline = y_values[x_values.argmin()]
        damage = np.trapz(baseline - y_values, x_values)
        labels.append(criterion_name)
        damages.append(damage)
    plt.figure(figsize=(8, 4))
    plt.bar(labels, damages)
    plt.ylabel("AUC pruning damage")
    plt.title(f"{dataset} {architecture}: lower is better")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_failure_cases(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    random_rows = [row for row in rows if row.get("criterion") == "random"]
    random_grouped = group_rows(random_rows, ("layer", "pruning_percentage"))
    failures = []
    labels = []
    for row in rows:
        if row.get("criterion") == "random":
            continue
        random_group = random_grouped.get((row.get("layer"), row.get("pruning_percentage")), [])
        random_values = [to_float(random_row, metric) for random_row in random_group]
        random_values = [value for value in random_values if not math.isnan(value)]
        if random_values:
            delta = to_float(row, metric) - float(np.mean(random_values))
            if delta < 0:
                labels.append(f"{row.get('criterion')} {row.get('layer')} {100 * to_float(row, 'pruning_percentage'):.0f}%")
                failures.append(delta)
    order = np.argsort(failures)[:20]
    plt.figure(figsize=(10, 5))
    plt.bar([labels[index] for index in order], [failures[index] for index in order])
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel(f"{metric.replace('_', ' ')} minus random")
    plt.title(f"{dataset} {architecture}: worst failures vs random")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.show()


def plot_fisher_advantage(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    lossfree_criterion: str = "hebbian",
    fisher_criterion: str = "activation_fisher",
    layer: str = "all",
    fine_tune_epochs: int = 0,
    metric: str = "test_accuracy",
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("layer") == layer
        and row.get("criterion") in {lossfree_criterion, fisher_criterion}
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    fisher_x, fisher_y = mean_by_x([row for row in rows if row.get("criterion") == fisher_criterion], "pruning_percentage", metric)
    lossfree_x, lossfree_y = mean_by_x([row for row in rows if row.get("criterion") == lossfree_criterion], "pruning_percentage", metric)
    lossfree_map = dict(zip(lossfree_x.tolist(), lossfree_y.tolist(), strict=True))
    advantage = np.array([value - lossfree_map.get(float(x_value), np.nan) for x_value, value in zip(fisher_x, fisher_y, strict=True)])
    plt.figure(figsize=(7, 4))
    plt.plot(fisher_x * 100.0, advantage, marker="o")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("% neurons pruned")
    plt.ylabel(f"{fisher_criterion} minus {lossfree_criterion}")
    plt.title(f"{dataset} {architecture}: Fisher advantage")
    plt.grid(True)
    plt.show()


def plot_parameter_reduction_vs_speedup(
    pruning_rows: list[dict[str, Any]],
    dataset: str,
    architecture: str,
    fine_tune_epochs: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in pruning_rows
        if row.get("dataset") == dataset
        and row.get("architecture") == architecture
        and row.get("layer") == "all"
        and int(float(row.get("fine_tune_epochs", 0))) == fine_tune_epochs
    ]
    baseline_rows = [row for row in rows if abs(to_float(row, "pruning_percentage")) < 1e-12]
    if not baseline_rows:
        baseline_time = min(
            [to_float(row, "compact_inference_time_seconds") for row in rows if to_float(row, "compact_inference_time_seconds") > 0],
            default=np.nan,
        )
    else:
        baseline_time = float(np.mean([to_float(row, "compact_inference_time_seconds") for row in baseline_rows]))

    plt.figure(figsize=(7, 5))
    for criterion_name, group in group_rows(rows, ("criterion",)).items():
        x_values = [to_float(row, "parameter_reduction") for row in group]
        speedups = [
            baseline_time / to_float(row, "compact_inference_time_seconds")
            if to_float(row, "compact_inference_time_seconds") > 0 and not math.isnan(baseline_time)
            else np.nan
            for row in group
        ]
        plt.scatter(x_values, speedups, label=criterion_name[0], alpha=0.7)
    plt.xlabel("Parameter reduction")
    plt.ylabel("Compact-model inference speedup")
    plt.title(f"{dataset} {architecture}: parameter reduction vs speedup")
    plt.grid(True)
    plt.legend()
    plt.show()
