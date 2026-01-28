import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Weights & Biases (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not installed. Run 'pip install wandb' to enable experiment tracking.")


DEFAULT_CONFIG_PATH = "configs/train_config.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    print(f"Warning: Requested device '{preference}' is not available. Falling back to CPU.")
    return torch.device("cpu")


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = config.copy()
    config.setdefault("training", {})
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("wandb", {})
    config.setdefault("checkpointing", {})

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.device is not None:
        config["device"] = args.device
    if args.wandb is not None:
        config["wandb"]["enabled"] = args.wandb
    if args.wandb_project is not None:
        config["wandb"]["project"] = args.wandb_project
    if args.checkpoint_dir is not None:
        config["checkpointing"]["checkpoint_dir"] = args.checkpoint_dir
    return config


def build_transforms(image_size: int, augmentation_cfg: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    aug_cfg = augmentation_cfg or {}
    train_ops = []
    if aug_cfg.get("horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if aug_cfg.get("random_crop", True):
        train_ops.append(transforms.RandomCrop(image_size, padding=4))
    if aug_cfg.get("color_jitter", True):
        train_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_ops = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


class _DatasetTransform:
    """Wrap torchvision transforms so HuggingFace datasets remain picklable with DataLoader workers."""

    def __init__(self, transform: transforms.Compose):
        self.transform = transform

    def __call__(self, examples):
        examples["pixel_values"] = [self.transform(image.convert("RGB")) for image in examples["image"]]
        return examples


def attach_transforms(dataset_split, transform):
    dataset_split.set_transform(_DatasetTransform(transform))


def default_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "label": labels}


def prepare_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset", "zh-plus/tiny-imagenet")
    image_size = data_cfg.get("image_size", 64)
    num_workers = data_cfg.get("num_workers", 4)
    batch_size = config.get("training", {}).get("batch_size", 64)
    pin_memory = torch.cuda.is_available()

    print(f"Loading dataset '{dataset_name}'...")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as exc:
        print(f"Error loading dataset '{dataset_name}': {exc}")
        raise

    train_transform, val_transform = build_transforms(image_size, data_cfg.get("augmentation", {}))
    attach_transforms(dataset["train"], train_transform)
    attach_transforms(dataset["valid"], val_transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate_fn,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(dataset["train"], shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(dataset["valid"], shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TinyCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [SimpleResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(SimpleResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    architecture = model_cfg.get("architecture", "TinyCNN").lower()
    num_classes = model_cfg.get("num_classes", 200)
    if architecture != "tinycnn":
        raise ValueError(f"Unsupported architecture '{architecture}'. Only TinyCNN is currently implemented.")
    return TinyCNN(num_classes=num_classes)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    clip_grad: Optional[float],
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total += batch_size

        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(total, 1)
    avg_acc = 100.0 * running_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": avg_acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch in loader:
        inputs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    avg_acc = 100.0 * running_correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": avg_acc}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_acc": best_acc,
        },
        checkpoint_path,
    )


def build_scheduler(optimizer: optim.Optimizer, training_cfg: Dict[str, Any]) -> Tuple[str, Optional[Any]]:
    scheduler_name = training_cfg.get("scheduler", "cosine").lower()
    epochs = training_cfg.get("epochs", 10)

    if scheduler_name == "cosine":
        return scheduler_name, optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        step_size = max(epochs // 3, 1)
        return scheduler_name, optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if scheduler_name == "plateau":
        return scheduler_name, optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    return scheduler_name, None


def train_model(config: Dict[str, Any], use_wandb: bool) -> None:
    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpointing", {})

    device = resolve_device(config.get("device", "auto"))
    set_seed(config.get("seed", 42))
    print(f"Using device: {device}")

    train_loader, val_loader = prepare_dataloaders(config)

    model = build_model(config.get("model", {})).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )
    scheduler_name, scheduler = build_scheduler(optimizer, training_cfg)
    use_amp = bool(training_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    clip_grad = training_cfg.get("clip_grad_norm")

    early_cfg = training_cfg.get("early_stopping", {})
    patience = early_cfg.get("patience", 5)
    min_delta = early_cfg.get("min_delta", 0.0)
    early_enabled = early_cfg.get("enabled", False)
    epochs_no_improve = 0

    best_acc = 0.0
    epochs = training_cfg.get("epochs", 10)

    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(model, log="all", log_freq=100)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, clip_grad
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        if scheduler:
            if scheduler_name == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | LR: {lr:.6f}"
        )

        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "learning_rate": lr,
                }
            )

        checkpoint_dir = Path(checkpoint_cfg.get("checkpoint_dir", "checkpoints"))
        best_path = checkpoint_dir / "best_tinycnn.pth"
        last_path = checkpoint_dir / "last_tinycnn.pth"

        improved = val_metrics["accuracy"] > (best_acc + min_delta)
        if improved:
            best_acc = val_metrics["accuracy"]
            epochs_no_improve = 0
            if checkpoint_cfg.get("save_best", True):
                save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, best_path)
                if use_wandb and WANDB_AVAILABLE:
                    artifact = wandb.Artifact(
                        name="tinycnn-model",
                        type="model",
                        description=f"Best model with val_acc={best_acc:.2f}%",
                    )
                    artifact.add_file(str(best_path))
                    wandb.log_artifact(artifact)
        else:
            epochs_no_improve += 1

        if checkpoint_cfg.get("save_last", True):
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, last_path)

        if early_enabled and epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyCNN on Tiny-ImageNet")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of epochs.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Override device selection (auto/cuda/mps/cpu).")
    parser.add_argument("--wandb", dest="wandb", action="store_true", help="Force-enable W&B logging.")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable W&B logging.")
    parser.add_argument("--wandb_project", type=str, help="W&B project name override.")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to store checkpoints.")
    parser.set_defaults(wandb=None)
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = load_config(args.config)
    config = apply_overrides(base_config, args)

    wandb_cfg = config.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    use_wandb = bool((args.wandb if args.wandb is not None else wandb_enabled) and WANDB_AVAILABLE)

    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "deepvision-tinycnn"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags"),
            config=config,
        )
        print(f"W&B run initialized: {wandb.run.name}")
    elif wandb_enabled and not WANDB_AVAILABLE:
        print("W&B requested but not installed. Install wandb or run with --no-wandb.")

    try:
        train_model(config, use_wandb=use_wandb)
    finally:
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            print("W&B run completed.")


if __name__ == "__main__":
    main()
