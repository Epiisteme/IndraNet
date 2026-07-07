"""Benchmark IndraNet's local quantum decision paths.

The repository does not ship a real iris benchmark dataset, so this script uses
deterministic synthetic amplitude vectors and the actual local PennyLane/QSVM,
QFT, QRNG token-proof, and CKKS code paths. The resulting numbers are useful for
local regression and demo calibration; they are not biometric accuracy claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from qbas_backend.encryption.fhe_ckks import CKKSContext, CKKSUnavailable
from qbas_backend.quantum.qft_iris import QFTIrisExtractor
from qbas_backend.quantum.qrng import BiometricTokenBinder, QuantumRNG
from qbas_backend.quantum.qsvm import QuantumSVMClassifier


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    n_qubits: int
    identities: int
    max_train_per_identity: int
    test_per_identity: int
    amplitude_noise: float
    qsvm_reps: int
    token_tolerance_bits: int


@dataclass(frozen=True)
class CurvePoint:
    train_per_identity: int
    total_train_samples: int
    qsvm_accuracy: float
    qsvm_train_ms: float
    qsvm_predict_ms_per_sample: float
    qft_cosine_accuracy: float
    qft_cosine_predict_ms_per_sample: float
    qrng_token_accuracy: float
    qrng_token_predict_ms_per_sample: float


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.ones_like(vector, dtype=np.float64) / math.sqrt(vector.size)
    return vector.astype(np.float64) / norm


def select_amplitude_centers(
    extractor: QFTIrisExtractor,
    rng: np.random.Generator,
    identities: int,
    candidates: int = 96,
) -> list[np.ndarray]:
    """Pick amplitude centers that are well-separated after QFT readout."""

    candidate_amplitudes = [normalize(rng.normal(size=extractor.amplitude_dim)) for _ in range(candidates)]
    candidate_features = np.asarray([extractor.extract(amplitudes) for amplitudes in candidate_amplitudes])

    selected: list[int] = [int(np.argmax(np.linalg.norm(candidate_features, axis=1)))]
    while len(selected) < identities:
        best_idx = -1
        best_distance = -1.0
        for idx, feature in enumerate(candidate_features):
            if idx in selected:
                continue
            min_distance = min(float(np.linalg.norm(feature - candidate_features[chosen])) for chosen in selected)
            if min_distance > best_distance:
                best_distance = min_distance
                best_idx = idx
        selected.append(best_idx)
    return [candidate_amplitudes[idx] for idx in selected]


def synthesize_dataset(config: BenchmarkConfig, extractor: QFTIrisExtractor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(config.seed)
    centers = select_amplitude_centers(extractor, rng, config.identities)
    train_features: list[np.ndarray] = []
    train_labels: list[str] = []
    test_features: list[np.ndarray] = []
    test_labels: list[str] = []
    extraction_latencies: list[float] = []

    for class_idx, center in enumerate(centers):
        label = f"identity_{class_idx + 1}"
        sample_count = config.max_train_per_identity + config.test_per_identity
        for sample_idx in range(sample_count):
            amplitudes = normalize(center + rng.normal(scale=config.amplitude_noise, size=center.shape))
            start = time.perf_counter()
            features = extractor.extract(amplitudes)
            extraction_latencies.append((time.perf_counter() - start) * 1000.0)
            if sample_idx < config.max_train_per_identity:
                train_features.append(features)
                train_labels.append(label)
            else:
                test_features.append(features)
                test_labels.append(label)

    return (
        np.asarray(train_features, dtype=np.float64),
        np.asarray(train_labels),
        np.asarray(test_features, dtype=np.float64),
        np.asarray(test_labels),
        mean(extraction_latencies),
    )


def accuracy(predicted: Iterable[str], expected: np.ndarray) -> float:
    predicted_array = np.asarray(list(predicted))
    return float(np.mean(predicted_array == expected))


def qft_cosine_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> list[str]:
    labels = sorted(set(train_y.tolist()))
    centroids = []
    for label in labels:
        centroid = train_x[train_y == label].mean(axis=0)
        centroids.append(normalize(centroid))
    centroid_matrix = np.asarray(centroids)
    predictions = []
    for sample in test_x:
        sample_norm = normalize(sample)
        similarities = centroid_matrix @ sample_norm
        predictions.append(labels[int(np.argmax(similarities))])
    return predictions


def qrng_token_predict(
    binder: BiometricTokenBinder,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_per_identity: int,
) -> list[str]:
    tokens_by_label: dict[str, list[str]] = {}
    for label in sorted(set(train_y.tolist())):
        label_features = train_x[train_y == label][:train_per_identity]
        tokens_by_label[label] = [binder.encode_token(binder.enroll(features)) for features in label_features]

    predictions: list[str] = []
    for sample in test_x:
        matches = [
            label
            for label, tokens in tokens_by_label.items()
            if any(binder.verify(sample, token) for token in tokens)
        ]
        predictions.append(matches[0] if len(matches) == 1 else "unknown")
    return predictions


def benchmark_curve(config: BenchmarkConfig) -> tuple[list[CurvePoint], dict[str, float | str | None]]:
    extractor = QFTIrisExtractor(n_qubits=config.n_qubits)
    extractor.warmup()
    train_x, train_y, test_x, test_y, qft_extract_ms = synthesize_dataset(config, extractor)
    binder = BiometricTokenBinder(QuantumRNG(n_qubits=max(4, config.n_qubits)), config.token_tolerance_bits)

    points: list[CurvePoint] = []
    for train_per_identity in range(1, config.max_train_per_identity + 1):
        subset_mask = np.concatenate(
            [
                np.where(train_y == label)[0][:train_per_identity]
                for label in sorted(set(train_y.tolist()))
            ]
        )
        subset_x = train_x[subset_mask]
        subset_y = train_y[subset_mask]

        qsvm = QuantumSVMClassifier(n_qubits=config.n_qubits, reps=config.qsvm_reps)
        start = time.perf_counter()
        qsvm.fit(subset_x, subset_y)
        qsvm_train_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        qsvm_predictions = qsvm.predict(test_x)["identity"]
        qsvm_predict_ms = ((time.perf_counter() - start) * 1000.0) / len(test_x)

        start = time.perf_counter()
        cosine_predictions = qft_cosine_predict(subset_x, subset_y, test_x)
        cosine_predict_ms = ((time.perf_counter() - start) * 1000.0) / len(test_x)

        start = time.perf_counter()
        token_predictions = qrng_token_predict(binder, subset_x, subset_y, test_x, train_per_identity)
        token_predict_ms = ((time.perf_counter() - start) * 1000.0) / len(test_x)

        points.append(
            CurvePoint(
                train_per_identity=train_per_identity,
                total_train_samples=len(subset_x),
                qsvm_accuracy=accuracy(qsvm_predictions, test_y),
                qsvm_train_ms=qsvm_train_ms,
                qsvm_predict_ms_per_sample=qsvm_predict_ms,
                qft_cosine_accuracy=accuracy(cosine_predictions, test_y),
                qft_cosine_predict_ms_per_sample=cosine_predict_ms,
                qrng_token_accuracy=accuracy(token_predictions, test_y),
                qrng_token_predict_ms_per_sample=token_predict_ms,
            )
        )

    ckks_metrics = benchmark_ckks(test_x)
    ckks_metrics["qft_extract_ms_per_sample"] = qft_extract_ms
    return points, ckks_metrics


def benchmark_ckks(test_x: np.ndarray) -> dict[str, float | str | None]:
    if len(test_x) < 2:
        return {"ckks_status": "skipped", "ckks_reason": "not enough samples"}
    try:
        start = time.perf_counter()
        ckks = CKKSContext(poly_modulus_degree=8192, coeff_mod_bits=[60, 40, 40, 60], scale_bits=40)
        setup_ms = (time.perf_counter() - start) * 1000.0
    except CKKSUnavailable as exc:
        return {"ckks_status": "unavailable", "ckks_reason": str(exc)}

    try:
        probe = normalize(test_x[0])
        template = normalize(test_x[1])
        start = time.perf_counter()
        enc_probe = ckks.encrypt(probe)
        enc_template = ckks.encrypt(template)
        encrypt_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        enc_score = ckks.encrypted_cosine_similarity(enc_probe, enc_template)
        score = float(ckks.decrypt(enc_score)[0])
        score_ms = (time.perf_counter() - start) * 1000.0

        plaintext_score = float(np.dot(probe, template))
        return {
            "ckks_status": "ok",
            "ckks_setup_ms": setup_ms,
            "ckks_encrypt_two_vectors_ms": encrypt_ms,
            "ckks_score_decrypt_ms": score_ms,
            "ckks_plaintext_dot": plaintext_score,
            "ckks_encrypted_dot": score,
            "ckks_abs_error": abs(score - plaintext_score),
        }
    finally:
        ckks.teardown()


def write_csv(path: Path, points: list[CurvePoint]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(point) for point in points)


def svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="' + " ".join(
        f"{x:.2f},{y:.2f}" for x, y in points
    ) + '" />'


def write_svg(path: Path, points: list[CurvePoint]) -> None:
    width, height = 920, 560
    left, right, top, bottom = 82, 36, 36, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [point.total_train_samples for point in points]
    min_x, max_x = min(x_values), max(x_values)

    def sx(value: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + ((value - min_x) / (max_x - min_x)) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value) * plot_h

    series = {
        "QSVM": ("#0f766e", [(sx(p.total_train_samples), sy(p.qsvm_accuracy)) for p in points]),
        "QFT cosine": ("#2563eb", [(sx(p.total_train_samples), sy(p.qft_cosine_accuracy)) for p in points]),
        "QRNG token proof": ("#b45309", [(sx(p.total_train_samples), sy(p.qrng_token_accuracy)) for p in points]),
    }
    grid = []
    for idx in range(6):
        value = idx / 5
        y = sy(value)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d8e0dd" />')
        grid.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>')
    for value in x_values:
        x = sx(value)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#eef2f3" />')
        grid.append(f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="12">{value}</text>')

    legend = []
    for idx, (name, (color, _coords)) in enumerate(series.items()):
        y = top + 18 + idx * 24
        legend.append(f'<line x1="{width-220}" y1="{y}" x2="{width-184}" y2="{y}" stroke="{color}" stroke-width="3" />')
        legend.append(f'<text x="{width-174}" y="{y+4}" font-size="13">{name}</text>')

    polylines = [svg_polyline(coords, color) for color, coords in series.values()]
    dots = []
    for color, coords in series.values():
        dots.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />' for x, y in coords)

    content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="24" font-size="18" font-weight="700">IndraNet quantum model convergence</text>
  <text x="{left}" y="{height-18}" font-size="13">Training samples enrolled across all identities</text>
  <text x="20" y="{top + plot_h / 2}" font-size="13" transform="rotate(-90 20 {top + plot_h / 2})">Held-out test accuracy</text>
  {"".join(grid)}
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9aa8a3"/>
  {"".join(polylines)}
  {"".join(dots)}
  {"".join(legend)}
</svg>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IndraNet quantum model accuracy and convergence.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--identities", type=int, default=3)
    parser.add_argument("--max-train-per-identity", type=int, default=6)
    parser.add_argument("--test-per-identity", type=int, default=10)
    parser.add_argument("--amplitude-noise", type=float, default=0.10)
    parser.add_argument("--qsvm-reps", type=int, default=1)
    parser.add_argument("--token-tolerance-bits", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    config = BenchmarkConfig(
        seed=args.seed,
        n_qubits=args.n_qubits,
        identities=args.identities,
        max_train_per_identity=args.max_train_per_identity,
        test_per_identity=args.test_per_identity,
        amplitude_noise=args.amplitude_noise,
        qsvm_reps=args.qsvm_reps,
        token_tolerance_bits=args.token_tolerance_bits,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    points, ckks_metrics = benchmark_curve(config)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    csv_path = args.out_dir / "quantum_model_accuracy.csv"
    json_path = args.out_dir / "quantum_model_benchmark.json"
    svg_path = args.out_dir / "quantum_model_convergence.svg"
    write_csv(csv_path, points)
    write_svg(svg_path, points)

    final_point = points[-1]
    report = {
        "benchmark_note": (
            "Deterministic synthetic amplitude benchmark using actual local IndraNet quantum code paths; "
            "not a real biometric accuracy claim."
        ),
        "config": asdict(config),
        "elapsed_ms": elapsed_ms,
        "final_test_time_accuracy": {
            "train_per_identity": final_point.train_per_identity,
            "total_train_samples": final_point.total_train_samples,
            "qsvm_accuracy": final_point.qsvm_accuracy,
            "qft_cosine_accuracy": final_point.qft_cosine_accuracy,
            "qrng_token_accuracy": final_point.qrng_token_accuracy,
        },
        "latency_ms_per_test_sample": {
            "qft_extract": ckks_metrics.get("qft_extract_ms_per_sample"),
            "qsvm_predict": final_point.qsvm_predict_ms_per_sample,
            "qft_cosine_predict": final_point.qft_cosine_predict_ms_per_sample,
            "qrng_token_predict": final_point.qrng_token_predict_ms_per_sample,
        },
        "ckks": ckks_metrics,
        "curve": [asdict(point) for point in points],
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "svg": str(svg_path),
        },
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["final_test_time_accuracy"], indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
