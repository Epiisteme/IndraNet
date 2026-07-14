"""Train and benchmark IndraNet quantum models on a local UBIRIS.v2 export.

This script performs the explicit training/enrollment stage before evaluation:

* QFT feature extraction over the train and held-out test split.
* QFT centroid model training for a fast cosine baseline.
* QSVM fitting using IndraNet's PennyLane quantum kernel classifier.
* QRNG token enrollment for train templates, followed by verification-style
  identity prediction on train and held-out test probes.

QFT itself is a deterministic feature extractor rather than a trainable circuit;
the trainable QFT-side artifact here is the identity centroid model built from
the extracted QFT feature vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (BACKEND_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_ubiris_v2 import (  # noqa: E402
    AccuracyRow,
    ExtractedSample,
    ImageSample,
    accuracy,
    arrays,
    discover_images,
    normalize,
)
from qbas_backend.quantum.qft_iris import QFTIrisExtractor  # noqa: E402
from qbas_backend.quantum.qrng import BiometricTokenBinder, QuantumRNG  # noqa: E402
from qbas_backend.quantum.qsvm import QuantumSVMClassifier  # noqa: E402
from qbas_backend.services.pipeline import IrisFeaturePipeline  # noqa: E402


FEATURE_SCHEMA_VERSION = "qft_probs_z_zz_multi_projection_plus_texture_v1"


@dataclass(frozen=True)
class SplitStats:
    total_images: int
    total_identities: int
    eligible_identities: int
    selected_identities: int
    train_samples: int
    test_samples: int
    split_strategy: str


@dataclass(frozen=True)
class TrainedArtifacts:
    qft_centroids_npz: str
    qsvm_model_pkl: str
    qrng_tokens_json: str
    feature_cache_npz: str
    manifest_json: str


@dataclass(frozen=True)
class LearningCurveRow:
    model: str
    train_fraction: float
    train_percent: float
    train_samples: int
    test_samples: int
    identities: int
    train_accuracy: float
    test_accuracy: float
    fit_ms: float
    train_predict_ms_per_sample: float
    test_predict_ms_per_sample: float
    kernel_evaluations: int
    artifact: str


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float | int | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "unknown"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(remaining)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(remaining)}s"


def make_qsvm_progress_logger(label: str) -> Callable[[dict[str, float | int | str | None]], None]:
    def callback(event: dict[str, float | int | str | None]) -> None:
        phase = str(event.get("phase", "kernel"))
        status = str(event.get("event", "progress"))
        completed = event.get("completed")
        total = event.get("total")
        elapsed_s = float(event.get("elapsed_s") or 0.0)
        rate = event.get("rate_per_s")
        eta_s = event.get("eta_s")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            percent = (completed / total) * 100.0
            rate_text = f"{float(rate):.1f}/s" if rate is not None else "unknown"
            log(
                f"{label} {phase} {status}: {completed}/{total} kernel evals "
                f"({percent:.1f}%) elapsed={format_duration(elapsed_s)} "
                f"rate={rate_text} eta={format_duration(float(eta_s) if eta_s is not None else None)}"
            )
            return
        log(f"{label} {phase} {status}: elapsed={format_duration(elapsed_s)}")

    return callback


def grouped_by_identity(samples: list[ImageSample]) -> dict[str, list[ImageSample]]:
    grouped: dict[str, list[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)
    return grouped


def split_for_training(
    samples: list[ImageSample],
    *,
    seed: int,
    max_identities: int | None,
    train_per_identity: int | None,
    test_per_identity: int | None,
    test_fraction: float,
    min_train_per_identity: int,
    min_test_per_identity: int,
) -> tuple[list[ImageSample], list[ImageSample], SplitStats]:
    grouped = grouped_by_identity(samples)
    rng = np.random.default_rng(seed)
    train: list[ImageSample] = []
    test: list[ImageSample] = []
    selected = 0

    if (train_per_identity is None) ^ (test_per_identity is None):
        raise ValueError("Provide both train_per_identity and test_per_identity, or neither")

    selected_identities = sorted(grouped, key=lambda identity: (-len(grouped[identity]), identity))
    if max_identities is not None:
        selected_identities = selected_identities[:max_identities]

    for identity in selected_identities:
        values = list(grouped[identity])
        rng.shuffle(values)
        if train_per_identity is not None and test_per_identity is not None:
            required = train_per_identity + test_per_identity
            if len(values) < required:
                continue
            train_values = values[:train_per_identity]
            test_values = values[train_per_identity:required]
            strategy = "fixed_per_identity"
        else:
            if len(values) < min_train_per_identity + min_test_per_identity:
                continue
            proposed_test = int(round(len(values) * test_fraction))
            test_count = max(min_test_per_identity, proposed_test)
            test_count = min(test_count, len(values) - min_train_per_identity)
            train_values = values[:-test_count]
            test_values = values[-test_count:]
            strategy = "fractional_all_images"
        if len(train_values) < min_train_per_identity or len(test_values) < min_test_per_identity:
            continue
        train.extend(train_values)
        test.extend(test_values)
        selected += 1

    eligible = 0
    for values in grouped.values():
        if train_per_identity is not None and test_per_identity is not None:
            eligible += int(len(values) >= train_per_identity + test_per_identity)
        else:
            eligible += int(len(values) >= min_train_per_identity + min_test_per_identity)

    return train, test, SplitStats(
        total_images=len(samples),
        total_identities=len(grouped),
        eligible_identities=eligible,
        selected_identities=selected,
        train_samples=len(train),
        test_samples=len(test),
        split_strategy=strategy if selected else "none",
    )


def extract_samples_with_progress(
    samples: list[ImageSample],
    pipeline: IrisFeaturePipeline,
    *,
    split_name: str,
    progress_every: int,
) -> tuple[list[ExtractedSample], list[dict[str, str]]]:
    extracted: list[ExtractedSample] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        try:
            result = pipeline.extract_from_bytes(sample.path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - preserve per-image extraction failures in the report.
            failures.append({"path": str(sample.path), "identity": sample.identity, "error": str(exc)})
        else:
            extracted.append(
                ExtractedSample(
                    path=str(sample.path),
                    identity=sample.identity,
                    features=np.asarray(result.features, dtype=np.float64).reshape(-1),
                    latency_ms=float(result.latency_ms),
                )
            )
        if index == len(samples) or index % progress_every == 0:
            elapsed = time.perf_counter() - started
            log(
                f"Extracted {split_name}: {index}/{len(samples)} images "
                f"usable={len(extracted)} failed={len(failures)} elapsed={elapsed:.1f}s"
            )
    return extracted, failures


def save_feature_cache(path: Path, train_samples: list[ExtractedSample], test_samples: list[ExtractedSample]) -> None:
    train_x, train_y = arrays(train_samples)
    test_x, test_y = arrays(test_samples)
    np.savez_compressed(
        path,
        schema_version=np.asarray([FEATURE_SCHEMA_VERSION], dtype=object),
        train_features=train_x,
        train_labels=train_y,
        train_paths=np.asarray([sample.path for sample in train_samples], dtype=object),
        train_latency_ms=np.asarray([sample.latency_ms for sample in train_samples], dtype=np.float64),
        test_features=test_x,
        test_labels=test_y,
        test_paths=np.asarray([sample.path for sample in test_samples], dtype=object),
        test_latency_ms=np.asarray([sample.latency_ms for sample in test_samples], dtype=np.float64),
    )


def load_feature_cache(path: Path) -> tuple[list[ExtractedSample], list[ExtractedSample]]:
    data = np.load(path, allow_pickle=True)
    if "schema_version" not in data:
        raise ValueError("cache predates feature schema tracking")
    schema_version = str(np.asarray(data["schema_version"], dtype=object).reshape(-1)[0])
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"cache schema {schema_version!r} does not match current schema {FEATURE_SCHEMA_VERSION!r}"
        )
    train_samples = [
        ExtractedSample(path=str(path_value), identity=str(label), features=features, latency_ms=float(latency))
        for path_value, label, features, latency in zip(
            data["train_paths"], data["train_labels"], data["train_features"], data["train_latency_ms"], strict=True
        )
    ]
    test_samples = [
        ExtractedSample(path=str(path_value), identity=str(label), features=features, latency_ms=float(latency))
        for path_value, label, features, latency in zip(
            data["test_paths"], data["test_labels"], data["test_features"], data["test_latency_ms"], strict=True
        )
    ]
    return train_samples, test_samples


def train_qft_centroids(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(sorted(set(train_y.tolist())), dtype=str)
    centroids = []
    for label in labels:
        centroids.append(normalize(train_x[train_y == label].mean(axis=0)))
    return labels, np.asarray(centroids, dtype=np.float64)


def qft_centroid_predict(labels: np.ndarray, centroids: np.ndarray, probe_x: np.ndarray) -> list[str]:
    predictions: list[str] = []
    for sample in probe_x:
        similarities = centroids @ normalize(sample)
        predictions.append(str(labels[int(np.argmax(similarities))]))
    return predictions


def train_qrng_tokens(
    binder: BiometricTokenBinder,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    progress_every: int,
) -> dict[str, list[str]]:
    tokens_by_label: dict[str, list[str]] = {}
    labels = sorted(set(train_y.tolist()))
    enrolled = 0
    for label in labels:
        tokens = []
        for features in train_x[train_y == label]:
            tokens.append(binder.encode_token(binder.enroll(features)))
            enrolled += 1
            if enrolled % progress_every == 0:
                log(f"Enrolled QRNG tokens: {enrolled}/{len(train_x)} templates")
        tokens_by_label[label] = tokens
    log(f"Enrolled QRNG tokens: {enrolled}/{len(train_x)} templates")
    return tokens_by_label


def qrng_token_predict_from_enrollment(
    binder: BiometricTokenBinder,
    tokens_by_label: dict[str, list[str]],
    probe_x: np.ndarray,
) -> list[str]:
    predictions: list[str] = []
    for sample in probe_x:
        matches = [
            label
            for label, tokens in tokens_by_label.items()
            if any(binder.verify(sample, token) for token in tokens)
        ]
        predictions.append(matches[0] if len(matches) == 1 else "unknown")
    return predictions


def write_dataclass_csv(path: Path, rows: list[AccuracyRow] | list[LearningCurveRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_accuracy_csv(path: Path, rows: list[AccuracyRow]) -> None:
    write_dataclass_csv(path, rows)


def estimate_qsvm_kernel_evaluations(train_count: int, test_count: int) -> int:
    return sum(estimate_qsvm_phase_kernel_evaluations(train_count, test_count).values())


def estimate_qsvm_phase_kernel_evaluations(train_count: int, test_count: int) -> dict[str, int]:
    return {
        "fit": train_count * train_count,
        "train_predict": train_count * train_count,
        "test_predict": test_count * train_count,
    }


def parse_learning_curve_fractions(value: str) -> list[float]:
    fractions: list[float] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        fraction = float(token)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Learning-curve fractions must be in the interval (0, 1]")
        fractions.append(fraction)
    return sorted(set(fractions))


def learning_curve_tag(fraction: float) -> str:
    return f"{int(round(fraction * 100)):03d}pct"


def select_learning_curve_indices(train_y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(train_y), dtype=int)
    rng = np.random.default_rng(seed)
    labels = sorted(set(train_y.tolist()))
    selected: set[int] = set()
    for label in labels:
        label_indices = np.flatnonzero(train_y == label)
        if len(label_indices) > 0:
            selected.add(int(rng.choice(label_indices)))
    target = max(len(selected), int(round(len(train_y) * fraction)))
    remaining = np.asarray([index for index in range(len(train_y)) if index not in selected], dtype=int)
    if len(remaining) > 0 and len(selected) < target:
        take = min(target - len(selected), len(remaining))
        selected.update(int(index) for index in rng.choice(remaining, size=take, replace=False))
    return np.asarray(sorted(selected), dtype=int)


def curve_row_from_accuracy(
    row: AccuracyRow,
    *,
    fraction: float,
    train_predict_ms_per_sample: float,
    kernel_evaluations: int,
    artifact: Path | str,
) -> LearningCurveRow:
    return LearningCurveRow(
        model=row.model,
        train_fraction=fraction,
        train_percent=round(fraction * 100.0, 2),
        train_samples=row.train_samples,
        test_samples=row.test_samples,
        identities=row.identities,
        train_accuracy=row.train_accuracy,
        test_accuracy=row.test_accuracy,
        fit_ms=row.fit_ms,
        train_predict_ms_per_sample=train_predict_ms_per_sample,
        test_predict_ms_per_sample=row.predict_ms_per_sample,
        kernel_evaluations=kernel_evaluations,
        artifact=str(artifact),
    )


def accuracy_row_from_curve(row: LearningCurveRow) -> AccuracyRow:
    return AccuracyRow(
        model=row.model,
        train_accuracy=row.train_accuracy,
        test_accuracy=row.test_accuracy,
        train_samples=row.train_samples,
        test_samples=row.test_samples,
        identities=row.identities,
        fit_ms=row.fit_ms,
        predict_ms_per_sample=row.test_predict_ms_per_sample,
    )


def run_learning_curve(
    *,
    args: argparse.Namespace,
    train_samples: list[ExtractedSample],
    test_samples: list[ExtractedSample],
    train_failures: list[dict[str, str]],
    test_failures: list[dict[str, str]],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    split_stats: SplitStats,
    started_all: float,
    qft_centroids_path: Path,
    qsvm_path: Path,
    qrng_tokens_path: Path,
    accuracy_csv_path: Path,
    report_path: Path,
    manifest_path: Path,
) -> int:
    try:
        fractions = parse_learning_curve_fractions(args.learning_curve_fractions)
    except ValueError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2

    curve_dir = args.out_dir / "learning_curve"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve_csv_path = args.out_dir / "ubiris_v2_learning_curve.csv"
    curve_report_path = args.out_dir / "ubiris_v2_learning_curve.json"
    rows: list[LearningCurveRow] = []
    full_fraction_rows: list[LearningCurveRow] = []

    estimated_qsvm_calls = 0
    if not args.skip_qsvm:
        for fraction in fractions:
            indices = select_learning_curve_indices(train_y, fraction, args.seed)
            estimated_qsvm_calls += estimate_qsvm_kernel_evaluations(len(indices), len(test_x))
    log(
        f"Learning-curve mode enabled: fractions={','.join(str(fraction) for fraction in fractions)} "
        f"test_samples={len(test_x)} estimated_qsvm_kernel_calls={estimated_qsvm_calls}"
    )

    def flush_outputs() -> None:
        write_dataclass_csv(curve_csv_path, rows)
        report = {
            "benchmark_note": "UBIRIS.v2 train-size learning curve using IndraNet QFT, QSVM, and QRNG code paths.",
            "elapsed_ms": (time.perf_counter() - started_all) * 1000.0,
            "split_stats": asdict(split_stats),
            "learning_curve_fractions": fractions,
            "extracted": {
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "train_extraction_ms_mean": mean(sample.latency_ms for sample in train_samples),
                "test_extraction_ms_mean": mean(sample.latency_ms for sample in test_samples),
                "train_failure_count": len(train_failures),
                "test_failure_count": len(test_failures),
                "train_failures": train_failures[:50],
                "test_failures": test_failures[:50],
            },
            "learning_curve": [asdict(row) for row in rows],
            "artifacts": {
                "learning_curve_csv": str(curve_csv_path),
                "learning_curve_json": str(curve_report_path),
                "learning_curve_artifact_dir": str(curve_dir),
                "full_accuracy_csv": str(accuracy_csv_path),
                "full_report_json": str(report_path),
            },
        }
        curve_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for curve_index, fraction in enumerate(fractions, start=1):
        tag = learning_curve_tag(fraction)
        indices = select_learning_curve_indices(train_y, fraction, args.seed + curve_index)
        subset_x = train_x[indices]
        subset_y = train_y[indices]
        identities = len(set(subset_y.tolist()))
        log(
            f"Learning curve {tag}: train_samples={len(subset_x)}/{len(train_x)} "
            f"identities={identities} test_samples={len(test_x)}"
        )

        qft_artifact = curve_dir / f"qft_centroids_{tag}.npz"
        log(f"Learning curve {tag}: training QFT centroid model")
        qft_labels, qft_centroids = train_qft_centroids(subset_x, subset_y)
        np.savez_compressed(qft_artifact, labels=qft_labels, centroids=qft_centroids)
        if fraction >= 1.0:
            np.savez_compressed(qft_centroids_path, labels=qft_labels, centroids=qft_centroids)
        qft_train_started = time.perf_counter()
        qft_train_predictions = qft_centroid_predict(qft_labels, qft_centroids, subset_x)
        qft_train_predict_ms = ((time.perf_counter() - qft_train_started) * 1000.0) / max(1, len(subset_x))
        qft_test_started = time.perf_counter()
        qft_test_predictions = qft_centroid_predict(qft_labels, qft_centroids, test_x)
        qft_test_predict_ms = ((time.perf_counter() - qft_test_started) * 1000.0) / max(1, len(test_x))
        qft_row = AccuracyRow(
            model="QFT centroid cosine",
            train_accuracy=accuracy(qft_train_predictions, subset_y),
            test_accuracy=accuracy(qft_test_predictions, test_y),
            train_samples=len(subset_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=0.0,
            predict_ms_per_sample=qft_test_predict_ms,
        )
        qft_curve_row = curve_row_from_accuracy(
            qft_row,
            fraction=fraction,
            train_predict_ms_per_sample=qft_train_predict_ms,
            kernel_evaluations=0,
            artifact=qft_artifact,
        )
        rows.append(qft_curve_row)
        if fraction >= 1.0:
            full_fraction_rows.append(qft_curve_row)

        qrng_artifact = curve_dir / f"qrng_tokens_{tag}.json"
        log(f"Learning curve {tag}: enrolling QRNG token templates")
        binder = BiometricTokenBinder(QuantumRNG(n_qubits=max(4, args.n_qubits)), args.token_tolerance_bits)
        qrng_started = time.perf_counter()
        tokens_by_label = train_qrng_tokens(binder, subset_x, subset_y, progress_every=args.progress_every)
        qrng_train_ms = (time.perf_counter() - qrng_started) * 1000.0
        qrng_artifact.write_text(json.dumps(tokens_by_label, indent=2, sort_keys=True), encoding="utf-8")
        if fraction >= 1.0:
            qrng_tokens_path.write_text(json.dumps(tokens_by_label, indent=2, sort_keys=True), encoding="utf-8")
        qrng_train_started = time.perf_counter()
        qrng_train_predictions = qrng_token_predict_from_enrollment(binder, tokens_by_label, subset_x)
        qrng_train_predict_ms = ((time.perf_counter() - qrng_train_started) * 1000.0) / max(1, len(subset_x))
        qrng_test_started = time.perf_counter()
        qrng_test_predictions = qrng_token_predict_from_enrollment(binder, tokens_by_label, test_x)
        qrng_test_predict_ms = ((time.perf_counter() - qrng_test_started) * 1000.0) / max(1, len(test_x))
        qrng_row = AccuracyRow(
            model="QRNG token proof",
            train_accuracy=accuracy(qrng_train_predictions, subset_y),
            test_accuracy=accuracy(qrng_test_predictions, test_y),
            train_samples=len(subset_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=qrng_train_ms,
            predict_ms_per_sample=qrng_test_predict_ms,
        )
        qrng_curve_row = curve_row_from_accuracy(
            qrng_row,
            fraction=fraction,
            train_predict_ms_per_sample=qrng_train_predict_ms,
            kernel_evaluations=0,
            artifact=qrng_artifact,
        )
        rows.append(qrng_curve_row)
        if fraction >= 1.0:
            full_fraction_rows.append(qrng_curve_row)

        if not args.skip_qsvm:
            qsvm_artifact = curve_dir / f"qsvm_{tag}.pkl"
            kernel_phases = estimate_qsvm_phase_kernel_evaluations(len(subset_x), len(test_x))
            log(
                f"Learning curve {tag}: training QSVM n_qubits={args.n_qubits} reps={args.qsvm_reps} "
                f"kernel_calls={sum(kernel_phases.values())} phases={kernel_phases}"
            )
            qsvm = QuantumSVMClassifier(n_qubits=args.n_qubits, C=args.qsvm_c, reps=args.qsvm_reps)
            qsvm_started = time.perf_counter()
            qsvm.fit(
                subset_x,
                subset_y,
                progress_callback=make_qsvm_progress_logger(f"QSVM {tag}"),
                progress_every=args.qsvm_progress_every,
            )
            qsvm_fit_ms = (time.perf_counter() - qsvm_started) * 1000.0
            qsvm.save(qsvm_artifact)
            if fraction >= 1.0:
                qsvm.save(qsvm_path)
            qsvm_train_started = time.perf_counter()
            qsvm_train_predictions = qsvm.predict(
                subset_x,
                progress_callback=make_qsvm_progress_logger(f"QSVM {tag}"),
                progress_every=args.qsvm_progress_every,
                progress_label="train_predict_kernel",
            )["identity"]
            qsvm_train_predict_ms = ((time.perf_counter() - qsvm_train_started) * 1000.0) / max(1, len(subset_x))
            qsvm_test_started = time.perf_counter()
            qsvm_test_predictions = qsvm.predict(
                test_x,
                progress_callback=make_qsvm_progress_logger(f"QSVM {tag}"),
                progress_every=args.qsvm_progress_every,
                progress_label="test_predict_kernel",
            )["identity"]
            qsvm_test_predict_ms = ((time.perf_counter() - qsvm_test_started) * 1000.0) / max(1, len(test_x))
            qsvm_row = AccuracyRow(
                model="QSVM",
                train_accuracy=accuracy(qsvm_train_predictions, subset_y),
                test_accuracy=accuracy(qsvm_test_predictions, test_y),
                train_samples=len(subset_x),
                test_samples=len(test_x),
                identities=identities,
                fit_ms=qsvm_fit_ms,
                predict_ms_per_sample=qsvm_test_predict_ms,
            )
            qsvm_curve_row = curve_row_from_accuracy(
                qsvm_row,
                fraction=fraction,
                train_predict_ms_per_sample=qsvm_train_predict_ms,
                kernel_evaluations=sum(kernel_phases.values()),
                artifact=qsvm_artifact,
            )
            rows.append(qsvm_curve_row)
            if fraction >= 1.0:
                full_fraction_rows.insert(0, qsvm_curve_row)

        flush_outputs()
        log(f"Learning curve {tag}: wrote {curve_csv_path} and {curve_report_path}")

    artifacts = TrainedArtifacts(
        qft_centroids_npz=str(qft_centroids_path),
        qsvm_model_pkl=str(qsvm_path) if not args.skip_qsvm else "",
        qrng_tokens_json=str(qrng_tokens_path),
        feature_cache_npz=str(args.out_dir / "ubiris_v2_qft_features.npz"),
        manifest_json=str(manifest_path),
    )
    manifest = {
        "dataset_root": str(args.dataset_root),
        "config": vars(args),
        "split_stats": asdict(split_stats),
        "learning_curve": {
            "enabled": True,
            "fractions": fractions,
            "csv": str(curve_csv_path),
            "json": str(curve_report_path),
            "artifact_dir": str(curve_dir),
        },
        "artifacts": asdict(artifacts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if full_fraction_rows:
        write_accuracy_csv(accuracy_csv_path, [accuracy_row_from_curve(row) for row in full_fraction_rows])
    report_path.write_text(curve_report_path.read_text(encoding="utf-8"), encoding="utf-8")
    log("Learning-curve benchmark complete")
    print(json.dumps([asdict(row) for row in rows], indent=2))
    print(f"Wrote {curve_csv_path}")
    print(f"Wrote {curve_report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train IndraNet QFT/QSVM/QRNG models on UBIRIS.v2 before benchmarking.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted UBIRIS.v2 image directory")
    parser.add_argument("--identity-regex", help="Regex with named group 'id' or first capture group for identity labels")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/ubiris_trained"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-identities", type=int, help="Optional cap for smoke tests; omit to use all identities")
    parser.add_argument("--train-per-identity", type=int, help="Fixed train images per identity")
    parser.add_argument("--test-per-identity", type=int, help="Fixed held-out test images per identity")
    parser.add_argument("--test-fraction", type=float, default=0.30, help="Held-out fraction when fixed counts are omitted")
    parser.add_argument("--min-train-per-identity", type=int, default=2)
    parser.add_argument("--min-test-per-identity", type=int, default=1)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--qsvm-reps", type=int, default=2)
    parser.add_argument("--qsvm-c", type=float, default=1.0)
    parser.add_argument("--token-tolerance-bits", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--qsvm-progress-every",
        type=int,
        default=50000,
        help="Print QSVM kernel progress every N kernel evaluations; set 0 to disable progress lines",
    )
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--skip-qsvm", action="store_true", help="Train QFT/QRNG only; useful for very large first passes")
    parser.add_argument("--learning-curve", action="store_true", help="Benchmark train-size curve instead of only one full fit")
    parser.add_argument("--learning-curve-fractions", default="0.10,0.25,0.50,0.75,1.0")
    args = parser.parse_args()

    started_all = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_path = args.out_dir / "ubiris_v2_qft_features.npz"
    qft_centroids_path = args.out_dir / "ubiris_v2_qft_centroids.npz"
    qsvm_path = args.out_dir / "ubiris_v2_qsvm.pkl"
    qrng_tokens_path = args.out_dir / "ubiris_v2_qrng_tokens.json"
    accuracy_csv_path = args.out_dir / "ubiris_v2_model_accuracy.csv"
    report_path = args.out_dir / "ubiris_v2_training_benchmark.json"
    manifest_path = args.out_dir / "ubiris_v2_training_manifest.json"

    log("Discovering UBIRIS.v2 images")
    try:
        samples = discover_images(args.dataset_root, args.identity_regex)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2
    train_raw, test_raw, split_stats = split_for_training(
        samples,
        seed=args.seed,
        max_identities=args.max_identities,
        train_per_identity=args.train_per_identity,
        test_per_identity=args.test_per_identity,
        test_fraction=args.test_fraction,
        min_train_per_identity=args.min_train_per_identity,
        min_test_per_identity=args.min_test_per_identity,
    )
    if split_stats.selected_identities < 2:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "Need at least two identities with enough train/test images",
                    "split_stats": asdict(split_stats),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    log(
        f"Split ready: identities={split_stats.selected_identities} "
        f"train={split_stats.train_samples} test={split_stats.test_samples} strategy={split_stats.split_strategy}"
    )

    loaded_cache = False
    if args.reuse_feature_cache and feature_cache_path.exists():
        log(f"Loading cached QFT features from {feature_cache_path}")
        try:
            train_samples, test_samples = load_feature_cache(feature_cache_path)
        except ValueError as exc:
            log(f"Ignoring stale feature cache: {exc}; regenerating features")
        else:
            train_failures: list[dict[str, str]] = []
            test_failures: list[dict[str, str]] = []
            loaded_cache = True

    if not loaded_cache:
        log(f"Initializing QFT extractor: n_qubits={args.n_qubits}")
        extractor = QFTIrisExtractor(n_qubits=args.n_qubits)
        extractor.warmup()
        pipeline = IrisFeaturePipeline(extractor)
        train_samples, train_failures = extract_samples_with_progress(
            train_raw, pipeline, split_name="train", progress_every=args.progress_every
        )
        test_samples, test_failures = extract_samples_with_progress(
            test_raw, pipeline, split_name="test", progress_every=args.progress_every
        )
        log(f"Saving QFT feature cache to {feature_cache_path}")
        save_feature_cache(feature_cache_path, train_samples, test_samples)

    if len(set(sample.identity for sample in train_samples)) < 2 or not test_samples:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "Feature extraction left too few usable identities or test samples",
                    "split_stats": asdict(split_stats),
                    "train_failure_count": len(train_failures),
                    "test_failure_count": len(test_failures),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    train_x, train_y = arrays(train_samples)
    test_x, test_y = arrays(test_samples)
    if args.learning_curve:
        return run_learning_curve(
            args=args,
            train_samples=train_samples,
            test_samples=test_samples,
            train_failures=train_failures,
            test_failures=test_failures,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            split_stats=split_stats,
            started_all=started_all,
            qft_centroids_path=qft_centroids_path,
            qsvm_path=qsvm_path,
            qrng_tokens_path=qrng_tokens_path,
            accuracy_csv_path=accuracy_csv_path,
            report_path=report_path,
            manifest_path=manifest_path,
        )

    identities = len(set(train_y.tolist()))
    rows: list[AccuracyRow] = []

    log("Training QFT centroid model from extracted QFT features")
    qft_labels, qft_centroids = train_qft_centroids(train_x, train_y)
    np.savez_compressed(qft_centroids_path, labels=qft_labels, centroids=qft_centroids)
    qft_train_predictions = qft_centroid_predict(qft_labels, qft_centroids, train_x)
    qft_started = time.perf_counter()
    qft_test_predictions = qft_centroid_predict(qft_labels, qft_centroids, test_x)
    qft_predict_ms = ((time.perf_counter() - qft_started) * 1000.0) / max(1, len(test_x))
    rows.append(
        AccuracyRow(
            model="QFT centroid cosine",
            train_accuracy=accuracy(qft_train_predictions, train_y),
            test_accuracy=accuracy(qft_test_predictions, test_y),
            train_samples=len(train_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=0.0,
            predict_ms_per_sample=qft_predict_ms,
        )
    )

    log("Training QRNG token enrollment templates")
    binder = BiometricTokenBinder(QuantumRNG(n_qubits=max(4, args.n_qubits)), args.token_tolerance_bits)
    qrng_started = time.perf_counter()
    tokens_by_label = train_qrng_tokens(binder, train_x, train_y, progress_every=args.progress_every)
    qrng_train_ms = (time.perf_counter() - qrng_started) * 1000.0
    qrng_tokens_path.write_text(json.dumps(tokens_by_label, indent=2, sort_keys=True), encoding="utf-8")
    qrng_train_predictions = qrng_token_predict_from_enrollment(binder, tokens_by_label, train_x)
    qrng_started = time.perf_counter()
    qrng_test_predictions = qrng_token_predict_from_enrollment(binder, tokens_by_label, test_x)
    qrng_predict_ms = ((time.perf_counter() - qrng_started) * 1000.0) / max(1, len(test_x))
    rows.append(
        AccuracyRow(
            model="QRNG token proof",
            train_accuracy=accuracy(qrng_train_predictions, train_y),
            test_accuracy=accuracy(qrng_test_predictions, test_y),
            train_samples=len(train_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=qrng_train_ms,
            predict_ms_per_sample=qrng_predict_ms,
        )
    )

    if args.skip_qsvm:
        log("Skipping QSVM training because --skip-qsvm was provided")
    else:
        estimated_kernels = estimate_qsvm_kernel_evaluations(len(train_x), len(test_x))
        log(
            f"Training QSVM: train_samples={len(train_x)} test_samples={len(test_x)} "
            f"n_qubits={args.n_qubits} reps={args.qsvm_reps} estimated_kernel_calls={estimated_kernels}"
        )
        qsvm = QuantumSVMClassifier(n_qubits=args.n_qubits, C=args.qsvm_c, reps=args.qsvm_reps)
        qsvm_started = time.perf_counter()
        qsvm.fit(
            train_x,
            train_y,
            progress_callback=make_qsvm_progress_logger("QSVM full"),
            progress_every=args.qsvm_progress_every,
        )
        qsvm_fit_ms = (time.perf_counter() - qsvm_started) * 1000.0
        log(f"QSVM fit complete in {qsvm_fit_ms / 1000.0:.1f}s; saving model to {qsvm_path}")
        qsvm.save(qsvm_path)
        qsvm_train_predictions = qsvm.predict(
            train_x,
            progress_callback=make_qsvm_progress_logger("QSVM full"),
            progress_every=args.qsvm_progress_every,
            progress_label="train_predict_kernel",
        )["identity"]
        qsvm_started = time.perf_counter()
        qsvm_test_predictions = qsvm.predict(
            test_x,
            progress_callback=make_qsvm_progress_logger("QSVM full"),
            progress_every=args.qsvm_progress_every,
            progress_label="test_predict_kernel",
        )["identity"]
        qsvm_predict_ms = ((time.perf_counter() - qsvm_started) * 1000.0) / max(1, len(test_x))
        rows.insert(
            0,
            AccuracyRow(
                model="QSVM",
                train_accuracy=accuracy(qsvm_train_predictions, train_y),
                test_accuracy=accuracy(qsvm_test_predictions, test_y),
                train_samples=len(train_x),
                test_samples=len(test_x),
                identities=identities,
                fit_ms=qsvm_fit_ms,
                predict_ms_per_sample=qsvm_predict_ms,
            ),
        )

    artifacts = TrainedArtifacts(
        qft_centroids_npz=str(qft_centroids_path),
        qsvm_model_pkl=str(qsvm_path) if not args.skip_qsvm else "",
        qrng_tokens_json=str(qrng_tokens_path),
        feature_cache_npz=str(feature_cache_path),
        manifest_json=str(manifest_path),
    )
    manifest = {
        "dataset_root": str(args.dataset_root),
        "config": {
            "identity_regex": args.identity_regex,
            "seed": args.seed,
            "max_identities": args.max_identities,
            "train_per_identity": args.train_per_identity,
            "test_per_identity": args.test_per_identity,
            "test_fraction": args.test_fraction,
            "min_train_per_identity": args.min_train_per_identity,
            "min_test_per_identity": args.min_test_per_identity,
            "n_qubits": args.n_qubits,
            "qsvm_reps": args.qsvm_reps,
            "qsvm_c": args.qsvm_c,
            "token_tolerance_bits": args.token_tolerance_bits,
            "qsvm_progress_every": args.qsvm_progress_every,
            "learning_curve": args.learning_curve,
            "learning_curve_fractions": args.learning_curve_fractions,
        },
        "split_stats": asdict(split_stats),
        "artifacts": asdict(artifacts),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_accuracy_csv(accuracy_csv_path, rows)
    report = {
        "benchmark_note": "UBIRIS.v2 train-then-benchmark run using IndraNet QFT, QSVM, and QRNG code paths.",
        "elapsed_ms": (time.perf_counter() - started_all) * 1000.0,
        "split_stats": asdict(split_stats),
        "extracted": {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_extraction_ms_mean": mean(sample.latency_ms for sample in train_samples),
            "test_extraction_ms_mean": mean(sample.latency_ms for sample in test_samples),
            "train_failure_count": len(train_failures),
            "test_failure_count": len(test_failures),
            "train_failures": train_failures[:50],
            "test_failures": test_failures[:50],
        },
        "accuracy": [asdict(row) for row in rows],
        "artifacts": {
            **asdict(artifacts),
            "accuracy_csv": str(accuracy_csv_path),
            "report_json": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("Training and benchmark complete")
    print(json.dumps(report["accuracy"], indent=2))
    print(f"Wrote {accuracy_csv_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
