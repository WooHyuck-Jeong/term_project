#! /usr/bin/env python3
"""
TorchScript 변환 스크립트.

학습된 sampling_mlp_v2.pth를 TorchScript 형식으로 변환하여
inference 시 Python 인터프리터 오버헤드를 줄인다.

변환 전/후 속도 비교도 함께 출력한다.

Output:
    models/sampling_mlp_v2_scripted.pt

실행:
    python3 export_torchscript.py
"""

import os
import time

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR: str = os.path.expanduser("~/term_project/models")
MODEL_PATH: str = os.path.join(MODEL_DIR, "sampling_mlp_v2.pth")
NORMALIZER_PATH: str = os.path.join(MODEL_DIR, "normalizer_v2.pth")
SCRIPTED_PATH: str = os.path.join(MODEL_DIR, "sampling_mlp_v2_scripted.pt")

# ---------------------------------------------------------------------------
# Architecture (train_mlp_v2.py와 동일)
# ---------------------------------------------------------------------------
INPUT_DIM: int = 7
OUTPUT_DIM: int = 3
HIDDEN_DIMS: list[int] = [256, 256, 128, 64]
DROPOUT_RATE: float = 0.2


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SamplingMLPv2(nn.Module):

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = nn.functional.normalize(out, p=2.0, dim=1)
        return out


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------
def benchmark_inference(
    model: nn.Module,
    normalizer_mean: torch.Tensor,
    normalizer_std: torch.Tensor,
    label: str,
    num_trials: int = 1000,
) -> float:
    """
    단일 샘플 inference를 num_trials번 반복하여 평균 시간 측정.

    Returns:
        평균 inference 시간 (ms)
    """
    dummy_input = torch.tensor(
        [[2.5, 1.0, -0.8, 4.6, 1.6, -1.0, 1.5]],
        dtype=torch.float32,
    )
    normalized = (dummy_input - normalizer_mean) / normalizer_std

    # warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(normalized)

    start = time.perf_counter()
    for _ in range(num_trials):
        with torch.no_grad():
            _ = model(normalized)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_trials * 1000
    print(f"  {label:<25}: {avg_ms:.4f} ms / call  ({num_trials} trials)")
    return avg_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cpu")

    # ---- Load model ----
    model = SamplingMLPv2()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")

    # ---- Load normalizer ----
    stats = torch.load(NORMALIZER_PATH, map_location=device)
    mean: torch.Tensor = stats["mean"]
    std: torch.Tensor = stats["std"]
    print(f"Normalizer loaded: {NORMALIZER_PATH}\n")

    # ---- Benchmark before conversion ----
    print("[ Inference speed benchmark ]")
    t_before = benchmark_inference(model, mean, std, "Original (PyTorch)")

    # ---- TorchScript 변환 ----
    # torch.jit.script: Python 코드를 정적 그래프로 컴파일
    # eval() 상태에서 변환해야 BatchNorm/Dropout이 inference 모드로 고정됨
    print("\nConverting to TorchScript...")
    scripted_model = torch.jit.script(model)
    print("Conversion successful.")

    # ---- Save ----
    scripted_model.save(SCRIPTED_PATH)
    print(f"Scripted model saved: {SCRIPTED_PATH}")

    # ---- Benchmark after conversion ----
    print()
    t_after = benchmark_inference(scripted_model, mean, std, "TorchScript")

    # ---- Summary ----
    speedup = t_before / t_after
    print(f"\n{'=' * 45}")
    print(f"  Speedup: {t_before:.4f} ms → {t_after:.4f} ms  ({speedup:.2f}x faster)")
    print(f"{'=' * 45}")

    # ---- Output 검증 (변환 전후 결과 동일한지 확인) ----
    dummy = torch.tensor(
        [[2.5, 1.0, -0.8, 4.6, 1.6, -1.0, 1.5]],
        dtype=torch.float32,
    )
    normalized = (dummy - mean) / std

    with torch.no_grad():
        out_original = model(normalized)
        out_scripted = scripted_model(normalized)

    max_diff = (out_original - out_scripted).abs().max().item()
    print(f"\nOutput diff (original vs scripted): {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✅ Output verified: results match.")
    else:
        print("⚠️  Output mismatch — check model conversion.")


if __name__ == "__main__":
    main()