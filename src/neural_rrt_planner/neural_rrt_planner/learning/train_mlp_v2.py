#! /usr/bin/env python3
"""
Improved MLP Training Script (v2) for Neural-Guided RRT* Sampler.

기존 train_mlp.py 대비 개선사항:
    [데이터셋]  rrt_dataset.csv 사용 (RRT* 경로 기반)
    [모델 구조] Dropout 추가, LeakyReLU 적용, 레이어 확장 [256, 256, 128, 64]
    [학습 방법] Epoch 500, 데이터셋 혼합 지원 (rrt + random)
    [속도]      EarlyStopping 추가 (patience=50)

Output:
    models/sampling_mlp_v2.pth
    models/normalizer_v2.pth
    models/training_log_v2.csv
"""

import argparse
import csv
import os
from typing import TypeAlias

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatTensor: TypeAlias = torch.Tensor

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# 데이터셋 경로 (두 개 모두 사용 가능)
RRT_DATASET_PATH: str = os.path.expanduser(
    "~/term_project/datasets/rrt_dataset_500.csv"
)
RANDOM_DATASET_PATH: str = os.path.expanduser(
    "~/term_project/datasets/sampling_dataset.csv"
)

MODEL_DIR: str = os.path.expanduser("~/term_project/models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "sampling_mlp_v2.pth")
NORMALIZER_PATH: str = os.path.join(MODEL_DIR, "normalizer_v2.pth")
LOG_PATH: str = os.path.join(MODEL_DIR, "training_log_v2.csv")

INPUT_DIM: int = 7
OUTPUT_DIM: int = 3

# 개선: 레이어 확장
HIDDEN_DIMS: list[int] = [256, 256, 128, 64]

# 개선: Dropout rate
DROPOUT_RATE: float = 0.2

LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256

# 개선: Epoch 증가
NUM_EPOCHS: int = 500

TRAIN_RATIO: float = 0.8
WEIGHT_DECAY: float = 1e-4

# 개선: EarlyStopping patience
EARLY_STOPPING_PATIENCE: int = 50

PRINT_EVERY: int = 20
SEED: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SamplingDataset(Dataset):
    """
    CSV 기반 샘플링 데이터셋.
    rrt_dataset.csv / sampling_dataset.csv 모두 동일한 컬럼 구조.
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

        self.x: FloatTensor = torch.tensor(self.inputs, dtype=torch.float32)
        self.y: FloatTensor = torch.tensor(self.targets, dtype=torch.float32)

        print(f"  Loaded: {len(self.x):>6} samples  ← {csv_path}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, FloatTensor]:
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self) -> None:
        self.mean: FloatTensor | None = None
        self.std: FloatTensor | None = None

    def fit(self, x: FloatTensor) -> None:
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self.std = torch.clamp(self.std, min=1e-8)

    def transform(self, x: FloatTensor) -> FloatTensor:
        return (x - self.mean) / self.std

    def save(self, path: str) -> None:
        torch.save({"mean": self.mean, "std": self.std}, path)
        print(f"Normalizer saved : {path}")


# ---------------------------------------------------------------------------
# Model (v2)
# ---------------------------------------------------------------------------
class SamplingMLPv2(nn.Module):
    """
    개선된 MLP.

    변경사항:
        - ReLU → LeakyReLU (negative slope=0.1)
          dead neuron 문제 완화, gradient 흐름 개선
        - Dropout(p=0.2) 추가
          과적합 억제, 작은 데이터셋(6,850개)에서 효과적
        - 레이어 확장: [128,128,64] → [256,256,128,64]
          더 복잡한 방향 패턴 학습 가능
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] = HIDDEN_DIMS,
        output_dim: int = OUTPUT_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim: int = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.net(x)
        out = nn.functional.normalize(out, p=2, dim=1)
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class CosineLoss(nn.Module):
    def forward(self, pred: FloatTensor, target: FloatTensor) -> FloatTensor:
        cos_sim = nn.functional.cosine_similarity(pred, target, dim=1)
        return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    val loss가 patience epoch 동안 개선되지 않으면 학습 중단.
    best model 가중치를 내부에 저장하여 학습 종료 후 복원한다.
    """

    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE) -> None:
        self.patience = patience
        self.best_val_loss: float = float("inf")
        self.counter: int = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns:
            True  → 학습 중단 신호
            False → 계속 학습
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    model: SamplingMLPv2,
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
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    log_rows: list[dict] = []
    stopped_epoch: int = NUM_EPOCHS

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'ES':>5}")
    print("-" * 58)

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

        # ---- EarlyStopping ----
        should_stop = early_stopping.step(val_loss, model)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(current_lr, 8),
        })

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            es_counter = f"{early_stopping.counter}/{EARLY_STOPPING_PATIENCE}"
            print(
                f"{epoch:>6} | "
                f"{train_loss:>12.6f} | "
                f"{val_loss:>12.6f} | "
                f"{current_lr:>10.2e} | "
                f"{es_counter:>5}"
            )

        if should_stop:
            stopped_epoch = epoch
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    # best 가중치 복원 후 저장
    early_stopping.restore_best(model)
    torch.save(model.state_dict(), MODEL_PATH)

    print("-" * 58)
    print(f"Best val loss : {early_stopping.best_val_loss:.6f}")
    print(f"Stopped epoch : {stopped_epoch}")
    print(f"Model saved   : {MODEL_PATH}")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "lr"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Log saved     : {LOG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train improved SamplingMLP v2"
    )
    parser.add_argument(
        "--use-random-dataset",
        action="store_true",
        help="rrt_dataset에 sampling_dataset을 합쳐서 학습 (default: rrt only)",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ---- Dataset ----
    print("Loading dataset...")
    rrt_dataset = SamplingDataset(RRT_DATASET_PATH)

    if args.use_random_dataset:
        random_dataset = SamplingDataset(RANDOM_DATASET_PATH)
        full_dataset = ConcatDataset([rrt_dataset, random_dataset])
        all_x = torch.cat([rrt_dataset.x, random_dataset.x], dim=0)
        print(f"  Combined     : {len(full_dataset)} samples total")
    else:
        full_dataset = rrt_dataset
        all_x = rrt_dataset.x

    train_size = int(len(full_dataset) * TRAIN_RATIO)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"  Train: {train_size} / Val: {val_size}")

    # ---- Normalizer ----
    normalizer = Normalizer()
    train_indices = train_dataset.indices
    train_x = all_x[train_indices]
    normalizer.fit(train_x)
    normalizer.save(NORMALIZER_PATH)

    # ---- Model ----
    model = SamplingMLPv2().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {INPUT_DIM} → {HIDDEN_DIMS} → {OUTPUT_DIM}")
    print(f"Params: {total_params:,}  |  Dropout: {DROPOUT_RATE}  |  Activation: LeakyReLU")
    print(f"Epochs: {NUM_EPOCHS}  |  EarlyStopping patience: {EARLY_STOPPING_PATIENCE}")

    # ---- Train ----
    train(model, normalizer, train_loader, val_loader, device)


if __name__ == "__main__":
    main()