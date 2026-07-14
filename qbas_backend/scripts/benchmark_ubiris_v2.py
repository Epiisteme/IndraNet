"""Evaluate IndraNet quantum decision paths on a local UBIRIS.v2 export.

Kaggle-hosted data is not downloaded by this script. Download/unzip the dataset
first, then point --dataset-root at the extracted image tree. The evaluator uses
the same local preprocessing, QFT feature extraction, QSVM, and QRNG token proof
implementations as the application.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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

from qbas_backend.quantum.qft_iris import QFTIrisExtractor
from qbas_backend.quantum.qrng import BiometricTokenBinder, QuantumRNG
from qbas_backend.quantum.qsvm import QuantumSVMClassifier
from qbas_backend.services.pipeline import IrisFeaturePipeline


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
GENERIC_DIR_NAMES = {
    "archive",
    "data",
    "dataset",
    "datasets",
    "image",
    "images",
    "img",
    "jpg",
    "png",
    "raw",
    "sample",
    "samples",
    "session",
    "sessions",
    "test",
    "train",
    "ubiris",
    "ubirisv2",
    "ubiris.v2",
}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    identity: str


@dataclass(frozen=True)
class ExtractedSample:
    path: str
    identity: str
    features: np.ndarray
    latency_ms: float


@dataclass(frozen=True)
class AccuracyRow:
    model: str
    train_accuracy: float
    test_accuracy: float
    train_samples: int
    test_samples: int
    identities: int
    fit_ms: float
    predict_ms_per_sample: float


def discover_images(dataset_root: Path, identity_regex: str | None) -> list[ImageSample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    regex = re.compile(identity_regex) if identity_regex else None
    samples: list[ImageSample] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        identity = infer_identity(path, dataset_root, regex)
        if identity:
            samples.append(ImageSample(path=path, identity=identity))
    return samples


def infer_identity(path: Path, dataset_root: Path, regex: re.Pattern[str] | None) -> str | None:
    relative = path.relative_to(dataset_root).as_posix()
    if regex is not None:
        match = regex.search(relative)
        if not match:
            return None
        if "id" in match.groupdict():
            return match.group("id")
        if match.groups():
            return match.group(1)
        return match.group(0)

    for parent in path.relative_to(dataset_root).parents:
        if str(parent) == ".":
            continue
        name = parent.name.strip()
        if name and name.lower() not in GENERIC_DIR_NAMES:
            return sanitize_label(name)

    stem = path.stem
    tokens = [token for token in re.split(r"[_\-.\s]+", stem) if token]
    if not tokens:
        return None
    if len(tokens) >= 2 and tokens[0].lower() in {"c", "s", "subject", "id", "person"}:
        return sanitize_label("_".join(tokens[:2]))
    return sanitize_label(tokens[0])


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]+", "_", value.strip())


def split_samples(
    samples: list[ImageSample],
    train_per_identity: int,
    test_per_identity: int,
    max_identities: int | None,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample], dict[str, int]]:
    grouped: dict[str, list[ImageSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.identity, []).append(sample)

    required = train_per_identity + test_per_identity
    eligible = {identity: values for identity, values in grouped.items() if len(values) >= required}
    selected_identities = sorted(eligible, key=lambda identity: (-len(eligible[identity]), identity))
    if max_identities is not None:
        selected_identities = selected_identities[:max_identities]

    rng = np.random.default_rng(seed)
    train: list[ImageSample] = []
    test: list[ImageSample] = []
    for identity in selected_identities:
        values = list(eligible[identity])
        rng.shuffle(values)
        train.extend(values[:train_per_identity])
        test.extend(values[train_per_identity:required])

    stats = {
        "total_images": len(samples),
        "total_identities": len(grouped),
        "eligible_identities": len(eligible),
        "selected_identities": len(selected_identities),
    }
    return train, test, stats


def extract_samples(samples: list[ImageSample], pipeline: IrisFeaturePipeline) -> tuple[list[ExtractedSample], list[dict[str, str]]]:
    extracted: list[ExtractedSample] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        try:
            result = pipeline.extract_from_bytes(sample.path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - keep per-image failures visible in the report.
            failures.append({"path": str(sample.path), "identity": sample.identity, "error": str(exc)})
            continue
        extracted.append(
            ExtractedSample(
                path=str(sample.path),
                identity=sample.identity,
                features=np.asarray(result.features, dtype=np.float64).reshape(-1),
                latency_ms=float(result.latency_ms),
            )
        )
    return extracted, failures


def arrays(samples: list[ExtractedSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.empty((0, 0), dtype=np.float64), np.asarray([], dtype=str)
    return np.asarray([sample.features for sample in samples], dtype=np.float64), np.asarray(
        [sample.identity for sample in samples], dtype=str
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros_like(vector, dtype=np.float64)
    return vector.astype(np.float64) / norm


def qft_cosine_predict(train_x: np.ndarray, train_y: np.ndarray, probe_x: np.ndarray) -> list[str]:
    labels = sorted(set(train_y.tolist()))
    centroids = []
    for label in labels:
        centroids.append(normalize(train_x[train_y == label].mean(axis=0)))
    centroid_matrix = np.asarray(centroids)
    predictions: list[str] = []
    for sample in probe_x:
        similarities = centroid_matrix @ normalize(sample)
        predictions.append(labels[int(np.argmax(similarities))])
    return predictions


def qrng_token_predict(
    binder: BiometricTokenBinder,
    train_x: np.ndarray,
    train_y: np.ndarray,
    probe_x: np.ndarray,
) -> list[str]:
    tokens_by_label: dict[str, list[str]] = {}
    for label in sorted(set(train_y.tolist())):
        tokens_by_label[label] = [binder.encode_token(binder.enroll(features)) for features in train_x[train_y == label]]

    predictions: list[str] = []
    for sample in probe_x:
        matches = [
            label
            for label, tokens in tokens_by_label.items()
            if any(binder.verify(sample, token) for token in tokens)
        ]
        predictions.append(matches[0] if len(matches) == 1 else "unknown")
    return predictions


def accuracy(predicted: Iterable[str], expected: np.ndarray) -> float:
    predicted_array = np.asarray(list(predicted), dtype=str)
    if len(expected) == 0:
        return 0.0
    return float(np.mean(predicted_array == expected))


def evaluate_models(
    train_samples: list[ExtractedSample],
    test_samples: list[ExtractedSample],
    n_qubits: int,
    qsvm_reps: int,
    token_tolerance_bits: int,
) -> list[AccuracyRow]:
    train_x, train_y = arrays(train_samples)
    test_x, test_y = arrays(test_samples)
    identities = len(set(train_y.tolist()))
    rows: list[AccuracyRow] = []

    qsvm = QuantumSVMClassifier(n_qubits=n_qubits, reps=qsvm_reps)
    started = time.perf_counter()
    qsvm.fit(train_x, train_y)
    qsvm_fit_ms = (time.perf_counter() - started) * 1000.0
    qsvm_train_predictions = qsvm.predict(train_x)["identity"]
    started = time.perf_counter()
    qsvm_test_predictions = qsvm.predict(test_x)["identity"]
    qsvm_predict_ms = ((time.perf_counter() - started) * 1000.0) / max(1, len(test_x))
    rows.append(
        AccuracyRow(
            model="QSVM",
            train_accuracy=accuracy(qsvm_train_predictions, train_y),
            test_accuracy=accuracy(qsvm_test_predictions, test_y),
            train_samples=len(train_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=qsvm_fit_ms,
            predict_ms_per_sample=qsvm_predict_ms,
        )
    )

    started = time.perf_counter()
    cosine_train_predictions = qft_cosine_predict(train_x, train_y, train_x)
    cosine_test_predictions = qft_cosine_predict(train_x, train_y, test_x)
    cosine_elapsed_ms = (time.perf_counter() - started) * 1000.0
    rows.append(
        AccuracyRow(
            model="QFT cosine",
            train_accuracy=accuracy(cosine_train_predictions, train_y),
            test_accuracy=accuracy(cosine_test_predictions, test_y),
            train_samples=len(train_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=0.0,
            predict_ms_per_sample=cosine_elapsed_ms / max(1, len(train_x) + len(test_x)),
        )
    )

    binder = BiometricTokenBinder(QuantumRNG(n_qubits=max(4, n_qubits)), token_tolerance_bits)
    started = time.perf_counter()
    token_train_predictions = qrng_token_predict(binder, train_x, train_y, train_x)
    token_test_predictions = qrng_token_predict(binder, train_x, train_y, test_x)
    token_elapsed_ms = (time.perf_counter() - started) * 1000.0
    rows.append(
        AccuracyRow(
            model="QRNG token proof",
            train_accuracy=accuracy(token_train_predictions, train_y),
            test_accuracy=accuracy(token_test_predictions, test_y),
            train_samples=len(train_x),
            test_samples=len(test_x),
            identities=identities,
            fit_ms=0.0,
            predict_ms_per_sample=token_elapsed_ms / max(1, len(train_x) + len(test_x)),
        )
    )
    return rows


def write_csv(path: Path, rows: list[AccuracyRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark IndraNet against a local UBIRIS.v2 Kaggle dataset export.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted UBIRIS.v2 image directory")
    parser.add_argument("--identity-regex", help="Regex with named group 'id' or first capture group for identity labels")
    parser.add_argument("--train-per-identity", type=int, default=2)
    parser.add_argument("--test-per-identity", type=int, default=2)
    parser.add_argument("--max-identities", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--qsvm-reps", type=int, default=1)
    parser.add_argument("--token-tolerance-bits", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    try:
        samples = discover_images(args.dataset_root, args.identity_regex)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": str(exc),
                    "dataset_root": str(args.dataset_root),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    train_raw, test_raw, stats = split_samples(
        samples, args.train_per_identity, args.test_per_identity, args.max_identities, args.seed
    )
    if stats["selected_identities"] < 2:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "Need at least two identities with enough images for train/test evaluation",
                    "stats": stats,
                    "dataset_root": str(args.dataset_root),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    extractor = QFTIrisExtractor(n_qubits=args.n_qubits)
    extractor.warmup()
    pipeline = IrisFeaturePipeline(extractor)
    train_samples, train_failures = extract_samples(train_raw, pipeline)
    test_samples, test_failures = extract_samples(test_raw, pipeline)
    if len(set(sample.identity for sample in train_samples)) < 2 or not test_samples:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "Feature extraction left too few usable identities or test samples",
                    "stats": stats,
                    "train_failures": train_failures[:20],
                    "test_failures": test_failures[:20],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    rows = evaluate_models(train_samples, test_samples, args.n_qubits, args.qsvm_reps, args.token_tolerance_bits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "ubiris_v2_model_accuracy.csv"
    json_path = args.out_dir / "ubiris_v2_benchmark.json"
    write_csv(csv_path, rows)
    report = {
        "benchmark_note": "UBIRIS.v2 evaluation using actual local IndraNet preprocessing, QFT, QSVM, and QRNG code paths.",
        "dataset_root": str(args.dataset_root),
        "config": {
            "identity_regex": args.identity_regex,
            "train_per_identity": args.train_per_identity,
            "test_per_identity": args.test_per_identity,
            "max_identities": args.max_identities,
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "qsvm_reps": args.qsvm_reps,
            "token_tolerance_bits": args.token_tolerance_bits,
        },
        "stats": stats,
        "extracted": {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_extraction_ms_mean": mean(sample.latency_ms for sample in train_samples),
            "test_extraction_ms_mean": mean(sample.latency_ms for sample in test_samples),
            "train_failures": train_failures[:50],
            "test_failures": test_failures[:50],
            "train_failure_count": len(train_failures),
            "test_failure_count": len(test_failures),
        },
        "accuracy": [asdict(row) for row in rows],
        "artifacts": {"csv": str(csv_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["accuracy"], indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
