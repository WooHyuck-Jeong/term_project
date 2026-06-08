#! /usr/bin/env python3
"""
MLP Training Script for Neural-Guided RRT* Sampler.

Dataset columns (10 total):
    Input  (7): current_x, current_y, current_z,
                goal_x, goal_y, goal_z,
                nearest_obstacle_distance
    Output (3): target_dx, target_dy, target_dz

Output:
    models/sampling_mlp.pth
    models/training_log.csv
"""

import csv
import os
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatTensor: TypeAlias = torch.Tensor

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DATASET_PATH: str = os.path.expanduser("~/term_project/datasets/sampling_dataset.csv")
MODEL_DIR: str = os.path.expanduser("~/term_project/models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "sampling_mlp.pth")
LOG_PATH: str = os.path.join(MODEL_DIR, "training_log.csv")

INPUT_DIM: int = 7
OUTPUT_DIM: int = 3
HIDDEN_DIMS: list[int] = [128, 128, 64]

LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
NUM_EPOCHS: int = 200
TRAIN_RATIO: float = 0.8
WEIGHT_DECAY: float = 1e-4

PRINT_EVERY: int = 10       # epoch 단위 로그 출력 주기
SEED: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SamplingDataset(Dataset):
    """
    CSV 기반 샘플링 데이터셋.

    Input  : [current_x, current_y, current_z,
               goal_x,    goal_y,    goal_z,
               nearest_obstacle_distance]
    Target : [target_dx, target_dy, target_dz]
    """

    def __init__(self, csv_path: str) -> None:
        self.inputs: list[list[float]] = []
        self.targets: list[list[float]] = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.inputs.append([
                    float(row["current_x"]),
                    float(row["current_y"]),
                    float(row["current_z"]),
                    float(row["goal_x"]),
                    float(row["goal_y"]),
                    float(row["goal_z"]),
                    float(row["nearest_obstacle_distance"]),
                ])
                self.targets.append([
                    float(row["target_dx"]),
                    float(row["target_dy"]),
                    float(row["target_dz"]),
                ])

        self.x: FloatTensor = torch.tensor(
            self.inputs,
            dtype=torch.float32
        )
        self.y: FloatTensor = torch.tensor(
            self.targets,
            dtype=torch.float32
        )

        print(f"Dataset loaded: {len(self.x)} samples")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, FloatTensor]:
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class Normalizer:
    """
    Input을 평균 0, 표준편차 1로 정규화.
    학습 데이터 통계만 사용하고, 추론 시에도 동일하게 적용.
    """

    def __init__(self) -> None:
        self.mean: FloatTensor | None = None
        self.std: FloatTensor | None = None

    def fit(self, x: FloatTensor) -> None:
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        # std가 0인 feature 보호
        self.std = torch.clamp(self.std, min=1e-8)

    def transform(self, x: FloatTensor) -> FloatTensor:
        return (x - self.mean) / self.std

    def save(self, path: str) -> None:
        torch.save(
            {"mean": self.mean, "std": self.std},
            path
        )
        print(f"Normalizer stats saved: {path}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SamplingMLP(nn.Module):
    """
    Input  (7) → Hidden layers → Output (3, unit vector)

    마지막 레이어 이후 L2 정규화를 적용해
    출력이 항상 단위벡터가 되도록 보장한다.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.net(x)
        # L2 normalize → 단위벡터 출력 보장
        out = nn.functional.normalize(out, p=2, dim=1)
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class CosineLoss(nn.Module):
    """
    타겟이 단위벡터이므로 cosine similarity 기반 loss 사용.
    loss = 1 - cos(pred, target)  → 방향이 일치할수록 0에 가까워짐.
    """

    def forward(
        self,
        pred: FloatTensor,
        target: FloatTensor
    ) -> FloatTensor:
        cos_sim = nn.functional.cosine_similarity(pred, target, dim=1)
        return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model: SamplingMLP,
    normalizer: Normalizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:

    os.makedirs(MODEL_DIR, exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    criterion = CosineLoss()

    best_val_loss: float = float("inf")
    log_rows: list[dict] = []

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, NUM_EPOCHS + 1):

        # ---- Train ----
        model.train()
        train_loss_sum: float = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = normalizer.transform(x_batch).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(x_batch)

        train_loss: float = train_loss_sum / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss_sum: float = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = normalizer.transform(x_batch).to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(x_batch)

        val_loss: float = val_loss_sum / len(val_loader.dataset)

        scheduler.step()
        current_lr: float = scheduler.get_last_lr()[0]

        # ---- Logging ----
        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(current_lr, 8),
        })

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(
                f"{epoch:>6} | "
                f"{train_loss:>12.6f} | "
                f"{val_loss:>12.6f} | "
                f"{current_lr:>10.2e}"
            )

        # ---- Best model 저장 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print("-" * 50)
    print(f"Training finished. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved: {MODEL_PATH}")

    # ---- Training log CSV 저장 ----
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "lr"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Training log saved: {LOG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----
    dataset = SamplingDataset(DATASET_PATH)

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Train: {train_size} samples / Val: {val_size} samples")

    # ---- Normalizer (train set 기준) ----
    normalizer = Normalizer()
    train_x = dataset.x[train_dataset.indices]
    normalizer.fit(train_x)

    normalizer_path = os.path.join(MODEL_DIR, "normalizer.pth")
    normalizer.save(normalizer_path)

    # ---- Model ----
    model = SamplingMLP().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Architecture: {INPUT_DIM} → {HIDDEN_DIMS} → {OUTPUT_DIM}")

    # ---- Train ----
    train(model, normalizer, train_loader, val_loader, device)


if __name__ == "__main__":
    main()