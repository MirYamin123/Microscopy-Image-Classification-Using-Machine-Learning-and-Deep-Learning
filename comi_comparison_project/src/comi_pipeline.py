from __future__ import annotations

import json
import math
import pickle
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES = ["actin", "mitochondria", "nucleus"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
DEFAULT_SPLIT_MAP = {
    "train": ["Z004", "Z005", "Z006", "Z007", "Z008"],
    "val": ["Z009"],
    "test": ["Z010"],
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIMPLE_MEAN = [0.5, 0.5, 0.5]
SIMPLE_STD = [0.5, 0.5, 0.5]


@dataclass
class TrainingConfig:
    model_name: str
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    warmup_epochs: int = 2
    fine_tune_learning_rate: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    normalize_mode: str = "imagenet"
    verbose: bool = True


@dataclass
class RFConfig:
    image_size: int = 64
    pca_n_components: float = 0.95
    n_estimators: int = 500
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str = "balanced_subsample"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_split_for_group(z_group: str, split_map: Dict[str, Sequence[str]]) -> str:
    for split_name, groups in split_map.items():
        if z_group in groups:
            return split_name
    raise ValueError(f"Z group {z_group} was not assigned to a split.")


def build_split_manifest(
    dataset_root: Path | str,
    split_map: Optional[Dict[str, Sequence[str]]] = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    split_map = split_map or DEFAULT_SPLIT_MAP

    rows: List[Dict[str, Any]] = []
    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for z_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            split_name = _find_split_for_group(z_dir.name, split_map)
            for image_path in sorted(z_dir.glob("*.jpg")):
                rows.append(
                    {
                        "file_path": str(image_path.resolve()),
                        "class_name": class_name,
                        "class_idx": CLASS_TO_INDEX[class_name],
                        "z_group": z_dir.name,
                        "split": split_name,
                        "file_name": image_path.name,
                    }
                )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No images were found under {dataset_root}")

    verify_manifest(manifest, split_map)
    return manifest.sort_values(["split", "class_name", "z_group", "file_name"]).reset_index(drop=True)


def verify_manifest(manifest: pd.DataFrame, split_map: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:
    split_map = split_map or DEFAULT_SPLIT_MAP

    required_columns = {"file_path", "class_name", "class_idx", "z_group", "split", "file_name"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    duplicate_paths = int(manifest["file_path"].duplicated().sum())
    if duplicate_paths:
        raise ValueError(f"Manifest contains {duplicate_paths} duplicate file paths.")

    expected_groups = {group for groups in split_map.values() for group in groups}
    found_groups = set(manifest["z_group"].unique())
    missing_groups = sorted(expected_groups.difference(found_groups))
    if missing_groups:
        raise ValueError(f"Manifest is missing expected Z groups: {missing_groups}")

    group_to_splits = manifest.groupby("z_group")["split"].nunique()
    leaked_groups = sorted(group_to_splits[group_to_splits > 1].index.tolist())
    if leaked_groups:
        raise ValueError(f"Z group leakage detected across splits: {leaked_groups}")

    class_counts = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    return {
        "num_rows": int(len(manifest)),
        "num_unique_paths": int(manifest["file_path"].nunique()),
        "groups_per_split": {
            split: sorted(manifest.loc[manifest["split"] == split, "z_group"].unique().tolist())
            for split in manifest["split"].unique()
        },
        "class_counts": class_counts,
    }


def summarize_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    summary = (
        manifest.groupby(["split", "class_name"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="class_name", columns="split", values="count")
        .fillna(0)
        .astype(int)
    )
    summary["total"] = summary.sum(axis=1)
    totals_row = pd.DataFrame(summary.sum(axis=0)).T
    totals_row.index = ["total"]
    return pd.concat([summary, totals_row], axis=0)


def build_smoke_manifest(manifest: pd.DataFrame, per_class_per_split: int = 12) -> pd.DataFrame:
    sampled = (
        manifest.sort_values(["split", "class_name", "z_group", "file_name"])
        .groupby(["split", "class_name"], group_keys=False)
        .head(per_class_per_split)
        .reset_index(drop=True)
    )
    return sampled


def save_manifest(manifest: pd.DataFrame, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    manifest.to_csv(output_path, index=False)
    return output_path


def _normalize_spec(mode: str) -> Tuple[List[float], List[float]]:
    if mode == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if mode == "simple":
        return SIMPLE_MEAN, SIMPLE_STD
    raise ValueError(f"Unsupported normalize mode: {mode}")


def build_transforms(image_size: int = 224, normalize_mode: str = "imagenet") -> Dict[str, transforms.Compose]:
    mean, std = _normalize_spec(normalize_mode)

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


class MicroscopyImageDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.manifest.iloc[index]
        image = Image.open(row["file_path"]).convert("L")
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return {
            "image": image_tensor,
            "label": int(row["class_idx"]),
            "path": row["file_path"],
            "class_name": row["class_name"],
            "z_group": row["z_group"],
        }


def create_dataloaders(
    manifest: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    normalize_mode: str = "imagenet",
    smoke_test: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, MicroscopyImageDataset], pd.DataFrame]:
    effective_manifest = build_smoke_manifest(manifest) if smoke_test else manifest.copy()
    tfms = build_transforms(image_size=image_size, normalize_mode=normalize_mode)

    datasets = {
        split_name: MicroscopyImageDataset(
            effective_manifest.loc[effective_manifest["split"] == split_name].reset_index(drop=True),
            transform=tfms[split_name],
        )
        for split_name in ["train", "val", "test"]
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return dataloaders, datasets, effective_manifest


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )
        self.output_layer = nn.Linear(128, num_classes)

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.embedding_head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.forward_embedding(x))


def build_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    if model_name == "baseline_cnn":
        return SmallCNN(num_classes=num_classes)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def freeze_backbone_for_warmup(model: nn.Module, model_name: str) -> None:
    if model_name != "resnet18":
        return
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all_layers(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _forward_pass(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    logits = model(images)
    loss = criterion(logits, labels)
    return loss, logits, labels


def _epoch_stats(losses: List[float], predictions: List[int], targets: List[int]) -> Dict[str, float]:
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": accuracy_score(targets, predictions) if targets else math.nan,
        "macro_f1": f1_score(targets, predictions, average="macro") if targets else math.nan,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    predictions: List[int] = []
    targets: List[int] = []

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, logits, labels = _forward_pass(model, batch, criterion, device)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return _epoch_stats(losses, predictions, targets)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    predictions: List[int] = []
    targets: List[int] = []

    for batch in loader:
        loss, logits, labels = _forward_pass(model, batch, criterion, device)
        losses.append(loss.item())
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return _epoch_stats(losses, predictions, targets)


def fit_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    config: TrainingConfig,
    output_dir: Path | str,
    device: Optional[torch.device] = None,
    pretrained: bool = True,
) -> Tuple[nn.Module, pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    device = device or resolve_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config.model_name == "resnet18" and pretrained:
        freeze_backbone_for_warmup(model, config.model_name)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: List[Dict[str, float]] = []
    best_score = -math.inf
    best_state: Optional[Dict[str, Any]] = None
    patience_counter = 0
    unfrozen = config.model_name != "resnet18"

    for epoch in range(1, config.num_epochs + 1):
        if config.model_name == "resnet18" and pretrained and not unfrozen and epoch > config.warmup_epochs:
            unfreeze_all_layers(model)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.fine_tune_learning_rate,
                weight_decay=config.weight_decay,
            )
            unfrozen = True

        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_metrics = evaluate_one_epoch(model, dataloaders["val"], criterion, device)

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(history_row)

        if config.verbose:
            print(
                f"[{config.model_name}] epoch {epoch}/{config.num_epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        monitored_score = val_metrics["macro_f1"]
        if monitored_score > best_score:
            best_score = monitored_score
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, output_dir / f"{config.model_name}_best.pt")
            if config.verbose:
                print(f"[{config.model_name}] new best checkpoint saved")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            if config.verbose:
                print(f"[{config.model_name}] early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / f"{config.model_name}_history.csv", index=False)
    save_json(asdict(config), output_dir / f"{config.model_name}_config.json")
    return model, history


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: Optional[torch.device] = None) -> pd.DataFrame:
    device = device or resolve_device()
    model.eval()
    model.to(device)

    rows: List[Dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        for idx in range(len(batch["path"])):
            row = {
                "path": batch["path"][idx],
                "true_idx": int(labels[idx].detach().cpu().item()),
                "pred_idx": int(predictions[idx].detach().cpu().item()),
                "true_label": CLASS_NAMES[int(labels[idx].detach().cpu().item())],
                "pred_label": CLASS_NAMES[int(predictions[idx].detach().cpu().item())],
                "confidence": float(probabilities[idx, predictions[idx]].detach().cpu().item()),
                "is_correct": bool(predictions[idx].detach().cpu().item() == labels[idx].detach().cpu().item()),
            }
            for class_idx, class_name in enumerate(CLASS_NAMES):
                row[f"prob_{class_name}"] = float(probabilities[idx, class_idx].detach().cpu().item())
            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(predictions_df: pd.DataFrame, class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, Any]:
    y_true = predictions_df["true_idx"].tolist()
    y_pred = predictions_df["pred_idx"].tolist()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def classification_report_frame(metrics: Dict[str, Any]) -> pd.DataFrame:
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    numeric_columns = ["precision", "recall", "f1-score", "support"]
    for column in numeric_columns:
        if column in report.columns:
            report[column] = pd.to_numeric(report[column], errors="coerce")
    return report


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_native(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_native(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(payload: Dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_to_native(payload), file, indent=2)
    return output_path


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    output_dir: Path | str,
    prefix: str,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    predictions_df = predict_dataset(model, loader, device=device)
    metrics = compute_metrics(predictions_df)
    report_df = classification_report_frame(metrics)

    predictions_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    report_df.to_csv(output_dir / f"{prefix}_classification_report.csv")
    save_json(metrics, output_dir / f"{prefix}_metrics.json")
    return predictions_df, metrics, report_df


def plot_class_distribution(manifest: pd.DataFrame, output_path: Optional[Path | str] = None) -> plt.Figure:
    counts = (
        manifest.groupby(["split", "class_name"])
        .size()
        .reset_index(name="count")
        .pivot(index="class_name", columns="split", values="count")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("COMI Class Distribution by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.legend(title="Split")
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_sample_images(
    manifest: pd.DataFrame,
    samples_per_class: int = 3,
    split: str = "train",
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    subset = manifest.loc[manifest["split"] == split]
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(4 * samples_per_class, 4 * len(CLASS_NAMES)))

    if len(CLASS_NAMES) == 1:
        axes = np.array([axes])

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_rows = subset.loc[subset["class_name"] == class_name].head(samples_per_class).reset_index(drop=True)
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx < len(class_rows):
                image = Image.open(class_rows.loc[col_idx, "file_path"]).convert("L")
                ax.imshow(image, cmap="gray")
                ax.set_title(f"{class_name}\n{class_rows.loc[col_idx, 'z_group']}")
            ax.axis("off")

    fig.suptitle(f"Sample {split.title()} Images", fontsize=14)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_training_history(history: pd.DataFrame, model_name: str, output_path: Optional[Path | str] = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="val accuracy")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro F1")
    axes[1].set_title(f"{model_name} Accuracy / Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    class_names: Sequence[str] = CLASS_NAMES,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def denormalize_tensor(image_tensor: torch.Tensor, normalize_mode: str = "imagenet") -> torch.Tensor:
    mean, std = _normalize_spec(normalize_mode)
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(-1, 1, 1)
    return image_tensor * std_tensor + mean_tensor


def plot_prediction_examples(
    predictions_df: pd.DataFrame,
    samples: int = 6,
    correct: bool = False,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    subset = predictions_df.loc[predictions_df["is_correct"] == correct].head(samples).reset_index(drop=True)
    columns = min(samples, 3)
    rows = max(1, math.ceil(len(subset) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.array(axes).reshape(rows, columns)

    for idx in range(rows * columns):
        ax = axes[idx // columns, idx % columns]
        if idx < len(subset):
            image = Image.open(subset.loc[idx, "path"]).convert("L")
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"true={subset.loc[idx, 'true_label']}\npred={subset.loc[idx, 'pred_label']}\nconf={subset.loc[idx, 'confidence']:.2f}"
            )
        ax.axis("off")

    fig.suptitle("Correct Predictions" if correct else "Misclassified Samples", fontsize=14)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0]

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        target_score = logits[:, class_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations and gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def plot_gradcam_examples(
    model: nn.Module,
    dataset: MicroscopyImageDataset,
    predictions_df: pd.DataFrame,
    output_path: Path | str,
    device: Optional[torch.device] = None,
    normalize_mode: str = "imagenet",
    max_examples: int = 6,
) -> plt.Figure:
    device = device or resolve_device()
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model, model.layer4[-1])

    chosen = predictions_df.head(max_examples).copy()
    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 4 * len(chosen)))
    axes = np.array(axes).reshape(len(chosen), 2)

    index_lookup = {row["file_path"]: idx for idx, row in dataset.manifest.iterrows()}

    for row_idx, (_, pred_row) in enumerate(chosen.iterrows()):
        sample = dataset[index_lookup[pred_row["path"]]]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        heatmap = gradcam.generate(image_tensor, class_idx=pred_row["pred_idx"])
        display_image = denormalize_tensor(sample["image"], normalize_mode=normalize_mode).permute(1, 2, 0).cpu().numpy()
        display_image = np.clip(display_image, 0.0, 1.0)

        axes[row_idx, 0].imshow(display_image)
        axes[row_idx, 0].set_title(f"Image\ntrue={pred_row['true_label']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(display_image)
        axes[row_idx, 1].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[row_idx, 1].set_title(f"Grad-CAM\npred={pred_row['pred_label']}")
        axes[row_idx, 1].axis("off")

    gradcam.remove()
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def compare_model_metrics(metric_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metric_map.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def load_grayscale_array(image_path: Path | str, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("L").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def build_flattened_image_features(manifest: pd.DataFrame, image_size: int = 64) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features = [load_grayscale_array(path, image_size).reshape(-1) for path in manifest["file_path"].tolist()]
    X = np.stack(features, axis=0)
    y = manifest["class_idx"].to_numpy(dtype=np.int64)
    paths = manifest["file_path"].tolist()
    return X, y, paths


def prepare_rf_data(
    manifest: pd.DataFrame,
    config: RFConfig,
    output_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    split_frames = {
        split_name: manifest.loc[manifest["split"] == split_name].reset_index(drop=True)
        for split_name in ["train", "val", "test"]
    }
    raw_splits = {
        split_name: build_flattened_image_features(split_frames[split_name], image_size=config.image_size)
        for split_name in split_frames
    }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(raw_splits["train"][0])
    X_val_scaled = scaler.transform(raw_splits["val"][0])
    X_test_scaled = scaler.transform(raw_splits["test"][0])

    pca = PCA(n_components=config.pca_n_components, random_state=config.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    rf_data = {
        "manifests": split_frames,
        "raw_features": {
            "train": raw_splits["train"][0],
            "val": raw_splits["val"][0],
            "test": raw_splits["test"][0],
        },
        "scaled_features": {
            "train": X_train_scaled,
            "val": X_val_scaled,
            "test": X_test_scaled,
        },
        "pca_features": {
            "train": X_train_pca,
            "val": X_val_pca,
            "test": X_test_pca,
        },
        "labels": {
            "train": raw_splits["train"][1],
            "val": raw_splits["val"][1],
            "test": raw_splits["test"][1],
        },
        "paths": {
            "train": raw_splits["train"][2],
            "val": raw_splits["val"][2],
            "test": raw_splits["test"][2],
        },
        "scaler": scaler,
        "pca": pca,
    }

    if output_dir:
        output_dir = ensure_dir(output_dir)
        save_json(asdict(config), Path(output_dir) / "rf_config.json")
        save_pickle(scaler, Path(output_dir) / "rf_scaler.pkl")
        save_pickle(pca, Path(output_dir) / "rf_pca.pkl")
        pd.DataFrame(
            {
                "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
            }
        ).to_csv(Path(output_dir) / "rf_pca_explained_variance.csv", index=False)

    return rf_data


def save_pickle(payload: Any, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("wb") as file:
        pickle.dump(payload, file)
    return output_path


def load_pickle(input_path: Path | str) -> Any:
    with Path(input_path).open("rb") as file:
        return pickle.load(file)


def train_random_forest(
    rf_data: Dict[str, Any],
    config: RFConfig,
    output_dir: Path | str,
) -> RandomForestClassifier:
    output_dir = ensure_dir(output_dir)
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight=config.class_weight,
    )
    model.fit(rf_data["pca_features"]["train"], rf_data["labels"]["train"])
    save_pickle(model, Path(output_dir) / "random_forest_model.pkl")
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X: np.ndarray,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    rows = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        result = {
            "path": row["file_path"],
            "true_idx": int(row["class_idx"]),
            "pred_idx": int(predictions[idx]),
            "true_label": CLASS_NAMES[int(row["class_idx"])],
            "pred_label": CLASS_NAMES[int(predictions[idx])],
            "confidence": float(np.max(probabilities[idx])),
            "is_correct": bool(int(predictions[idx]) == int(row["class_idx"])),
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            result[f"prob_{class_name}"] = float(probabilities[idx, class_idx])
        rows.append(result)
    return pd.DataFrame(rows)


def evaluate_random_forest_and_save(
    model: RandomForestClassifier,
    rf_data: Dict[str, Any],
    output_dir: Path | str,
    split_name: str = "test",
    prefix: str = "random_forest_test",
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    output_dir = ensure_dir(output_dir)
    predictions_df = predict_random_forest(
        model,
        rf_data["pca_features"][split_name],
        rf_data["manifests"][split_name],
    )
    metrics = compute_metrics(predictions_df)
    report_df = classification_report_frame(metrics)
    predictions_df.to_csv(Path(output_dir) / f"{prefix}_predictions.csv", index=False)
    report_df.to_csv(Path(output_dir) / f"{prefix}_classification_report.csv")
    save_json(metrics, Path(output_dir) / f"{prefix}_metrics.json")
    return predictions_df, metrics, report_df


def plot_confidence_histogram(
    predictions_df: pd.DataFrame,
    title: str,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(predictions_df["confidence"], bins=20, color="#4C78A8", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_model_comparison_chart(comparison_df: pd.DataFrame, output_path: Optional[Path | str] = None) -> plt.Figure:
    melted = comparison_df.melt(id_vars="model", value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        subset = melted.loc[melted["variable"] == metric_name]
        ax.plot(subset["model"], subset["value"], marker="o", label=metric_name)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Summary")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_probability_bars(
    predictions_df: pd.DataFrame,
    sample_index: int = 0,
    title: str = "Class probabilities",
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    row = predictions_df.reset_index(drop=True).iloc[sample_index]
    values = [row[f"prob_{class_name}"] for class_name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_rf_featurization_panel(
    image_path: Path | str,
    image_size: int = 64,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    original = Image.open(image_path).convert("L")
    resized = original.resize((image_size, image_size))
    flattened = np.asarray(resized, dtype=np.float32).reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(resized, cmap="gray")
    axes[1].set_title(f"Resized {image_size}x{image_size}")
    axes[1].axis("off")

    axes[2].imshow(flattened.reshape(1, -1), cmap="viridis", aspect="auto")
    axes[2].set_title("Flattened vector")
    axes[2].set_xlabel("Feature index")
    axes[2].set_yticks([])
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_pixel_to_vector_illustration(
    image_path: Path | str,
    image_size: int = 64,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    resized = Image.open(image_path).convert("L").resize((image_size, image_size))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    vector = image_array.reshape(-1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Pixel grid")
    axes[0].axis("off")
    axes[1].plot(vector, linewidth=1.0)
    axes[1].set_title("Same image as 1D feature vector")
    axes[1].set_xlabel("Flattened pixel index")
    axes[1].set_ylabel("Intensity")
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_pca_explained_variance(
    pca: PCA,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", linewidth=1)
    ax.set_title("PCA Cumulative Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_pca_scatter(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    projector = PCA(n_components=2, random_state=42)
    train_2d = projector.fit_transform(train_features)
    test_2d = projector.transform(test_features)
    color_map = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, data, labels, split_name in [
        (axes[0], train_2d, train_labels, "Train"),
        (axes[1], test_2d, test_labels, "Test"),
    ]:
        for class_idx, class_name in enumerate(CLASS_NAMES):
            mask = labels == class_idx
            ax.scatter(data[mask, 0], data[mask, 1], s=18, alpha=0.7, label=class_name, color=color_map[class_idx])
        ax.set_title(f"{split_name} PCA projection")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    axes[1].legend()
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_pca_reconstructions(
    train_scaled_features: np.ndarray,
    sample_scaled_vector: np.ndarray,
    image_size: int = 64,
    component_counts: Sequence[int] = (5, 20, 50, 100),
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    component_counts = [count for count in component_counts if count < train_scaled_features.shape[1]]
    fig, axes = plt.subplots(1, len(component_counts) + 1, figsize=(4 * (len(component_counts) + 1), 4))
    original = sample_scaled_vector.reshape(image_size, image_size)
    original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original sample")
    axes[0].axis("off")

    for idx, count in enumerate(component_counts, start=1):
        projector = PCA(n_components=count, random_state=42)
        projector.fit(train_scaled_features)
        reconstructed = projector.inverse_transform(projector.transform(sample_scaled_vector.reshape(1, -1))).reshape(image_size, image_size)
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)
        axes[idx].imshow(reconstructed, cmap="gray")
        axes[idx].set_title(f"{count} PCs")
        axes[idx].axis("off")

    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_rf_decision_flow(output_path: Optional[Path | str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    labels = [
        ("Image", 0.08),
        ("64x64 grayscale", 0.27),
        ("Flattened pixels", 0.46),
        ("PCA components", 0.66),
        ("Tree votes + probabilities", 0.88),
    ]
    for text, x in labels:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E8EEF7", "edgecolor": "#4C78A8"},
        )
    for start, end in zip(labels[:-1], labels[1:]):
        ax.annotate("", xy=(end[1] - 0.08, 0.5), xytext=(start[1] + 0.08, 0.5), arrowprops={"arrowstyle": "->", "lw": 2})
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_preprocessing_panel(
    image_path: Path | str,
    transform: transforms.Compose,
    title: str,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    image = Image.open(image_path).convert("L")
    tensor = transform(image)
    if tensor.shape[0] == 3:
        display_image = tensor.permute(1, 2, 0).cpu().numpy()
        display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min() + 1e-8)
    else:
        display_image = tensor.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(display_image, cmap="gray")
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def get_feature_map_layers(model: nn.Module, model_name: str) -> Dict[str, nn.Module]:
    if model_name == "baseline_cnn":
        return {
            "conv1": model.features[0],
            "conv2": model.features[4],
            "conv3": model.features[8],
            "conv4": model.features[12],
        }
    if model_name == "resnet18":
        return {
            "conv1": model.conv1,
            "layer1": model.layer1[-1],
            "layer2": model.layer2[-1],
            "layer4": model.layer4[-1],
        }
    raise ValueError(f"Unsupported model_name for feature maps: {model_name}")


def capture_feature_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    device = device or resolve_device()
    model = model.to(device)
    model.eval()
    features: Dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            features[layer_name] = output.detach().cpu().numpy()
            return None

        return _hook

    for layer_name, layer_module in get_feature_map_layers(model, model_name).items():
        hooks.append(layer_module.register_forward_hook(_make_hook(layer_name)))

    with torch.no_grad():
        model(input_tensor.to(device))

    for hook in hooks:
        hook.remove()
    return features


def plot_feature_map_grid(
    feature_maps: Dict[str, np.ndarray],
    max_channels: int = 6,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    layer_names = list(feature_maps.keys())
    fig, axes = plt.subplots(len(layer_names), max_channels, figsize=(2.2 * max_channels, 2.2 * len(layer_names)))
    axes = np.array(axes).reshape(len(layer_names), max_channels)

    for row_idx, layer_name in enumerate(layer_names):
        fmap = feature_maps[layer_name][0]
        channel_count = min(max_channels, fmap.shape[0])
        for col_idx in range(max_channels):
            ax = axes[row_idx, col_idx]
            if col_idx < channel_count:
                ax.imshow(fmap[col_idx], cmap="viridis")
                if col_idx == 0:
                    ax.set_ylabel(layer_name)
            ax.axis("off")

    fig.suptitle("Selected feature maps", fontsize=14)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def extract_model_embeddings(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    device = device or resolve_device()
    model = model.to(device)
    model.eval()
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if model_name == "baseline_cnn":
                embeddings = model.forward_embedding(images)
                logits = model.output_layer(embeddings)
            elif model_name == "resnet18":
                features = model.conv1(images)
                features = model.bn1(features)
                features = model.relu(features)
                features = model.maxpool(features)
                features = model.layer1(features)
                features = model.layer2(features)
                features = model.layer3(features)
                features = model.layer4(features)
                features = model.avgpool(features)
                embeddings = torch.flatten(features, 1)
                logits = model.fc(embeddings)
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")

            predictions = torch.argmax(logits, dim=1)
            for idx in range(images.size(0)):
                row = {
                    "path": batch["path"][idx],
                    "true_idx": int(batch["label"][idx]),
                    "true_label": CLASS_NAMES[int(batch["label"][idx])],
                    "pred_idx": int(predictions[idx].detach().cpu().item()),
                    "pred_label": CLASS_NAMES[int(predictions[idx].detach().cpu().item())],
                }
                embedding = embeddings[idx].detach().cpu().numpy()
                for emb_idx, value in enumerate(embedding):
                    row[f"emb_{emb_idx}"] = float(value)
                rows.append(row)

    return pd.DataFrame(rows)


def plot_embedding_projection(
    embeddings_df: pd.DataFrame,
    title: str,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    embedding_columns = [column for column in embeddings_df.columns if column.startswith("emb_")]
    projector = PCA(n_components=2, random_state=42)
    projection = projector.fit_transform(embeddings_df[embedding_columns].to_numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    color_map = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = embeddings_df["true_idx"] == class_idx
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=20,
            alpha=0.7,
            label=class_name,
            color=color_map[class_idx],
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_how_models_see_images(output_path: Optional[Path | str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    columns = [
        ("Random Forest", "Pixel vector\n+ PCA components"),
        ("Baseline CNN", "Learned local\nfeature maps"),
        ("ResNet18", "Transferred deep\nhierarchical features"),
    ]
    xs = [0.18, 0.5, 0.82]
    for (title, subtitle), x in zip(columns, xs):
        ax.text(
            x,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E8EEF7", "edgecolor": "#4C78A8"},
        )
        ax.text(x, 0.32, subtitle, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_decision_logic_comparison(output_path: Optional[Path | str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    rows = [
        ("RF", "Trees vote on PCA-reduced image features"),
        ("CNN", "Convolutions learn spatial features, then linear classifier predicts class"),
        ("ResNet18", "Pretrained deep features are fine-tuned, then final classifier predicts class"),
    ]
    y_positions = [0.75, 0.5, 0.25]
    for (model_name, description), y in zip(rows, y_positions):
        ax.text(0.1, y, model_name, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(0.58, y, description, ha="center", va="center", fontsize=11)
        ax.annotate("", xy=(0.27, y), xytext=(0.16, y), arrowprops={"arrowstyle": "->", "lw": 1.8})
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def build_model_richness_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "random_forest_pca", "feature_style": "flattened pixels + PCA", "spatial_learning": "no", "expected_strength": "classical baseline"},
            {"model": "baseline_cnn", "feature_style": "learned convolutional features", "spatial_learning": "yes", "expected_strength": "moderate deep-learning baseline"},
            {"model": "resnet18", "feature_style": "pretrained deep hierarchical features", "spatial_learning": "yes", "expected_strength": "strongest expected model"},
        ]
    )


def plot_confusion_matrix_comparison(
    metrics_by_model: Dict[str, Dict[str, Any]],
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    model_names = list(metrics_by_model.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    axes = np.array(axes).reshape(1, len(model_names))[0]

    for ax, model_name in zip(axes, model_names):
        matrix = np.array(metrics_by_model[model_name]["confusion_matrix"])
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(model_name)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_classwise_metric_comparison(
    report_frames: Dict[str, pd.DataFrame],
    metric_name: str = "f1-score",
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    rows = []
    for model_name, report_df in report_frames.items():
        for class_name in CLASS_NAMES:
            rows.append(
                {
                    "model": model_name,
                    "class_name": class_name,
                    "metric_value": float(report_df.loc[class_name, metric_name]),
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x_positions = np.arange(len(CLASS_NAMES))
    for idx, model_name in enumerate(report_frames.keys()):
        subset = plot_df.loc[plot_df["model"] == model_name]
        ax.bar(x_positions + (idx - 1) * width, subset["metric_value"], width=width, label=model_name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Class-wise {metric_name} comparison")
    ax.legend()
    fig.tight_layout()
    if output_path:
        ensure_dir(Path(output_path).parent)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig
