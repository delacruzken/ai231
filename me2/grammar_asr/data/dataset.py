"""Official gold-dataset loader for AI231 ME2.

Supports:
  - Local folder layout: {root}/{train,test,holdout}/manifest.csv + audio/
  - DGX class path: /data/ai231
  - Hugging Face course gold dataset (optional)

Never mutates official split membership. Validation is a speaker-disjoint
fold carved from train only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..schema import (
    OUT_OF_SCOPE, INTENTS, alias_intent, normalize_text, normalize_slot,
    lookup_variation, VARIATIONS,
)
from .features import SAMPLE_RATE, load_wav, log_mel, normalize_feat

HF_DATASET_ID = "airimonda/ai231-me2-voice-commands"
DEFAULT_LOCAL_CANDIDATES = [
    Path("/data/ai231"),
    Path("/data/ai231/dataset"),
    Path(__file__).resolve().parents[2] / "data" / "ai231-me2-voice-commands",
    Path(__file__).resolve().parents[2] / "data" / "gold",
]

OFFICIAL_SPLITS = ("train", "test", "holdout")
OPTIONAL_CONFIGS = (
    "supplemental_fil",
    "supplemental_synth",
    "synthetic_negatives",
    "numerals",
)


@dataclass
class Sample:
    path: Optional[Path]
    audio: Optional[np.ndarray]
    split: str
    intent: str
    slot_value: str
    variation: str
    variation_id: int
    transcript: str
    out_of_scope: bool
    speaker_id: str
    source: str
    is_synthetic: bool
    accent_group: str
    duration_s: float
    file: str = ""
    bucket: str = ""
    note: str = ""
    config: str = "default"

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("audio", None)
        d["path"] = str(self.path) if self.path else ""
        return d


def discover_dataset_root(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if _looks_like_gold(p):
            return p
        raise FileNotFoundError(f"Dataset root not found or invalid: {p}")
    env = os.environ.get("ME2_DATA_ROOT") or os.environ.get("AI231_DATA")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_gold(p):
            return p
    for cand in DEFAULT_LOCAL_CANDIDATES:
        if _looks_like_gold(cand):
            return cand.resolve()
    return None


def _looks_like_gold(root: Path) -> bool:
    if not root.is_dir():
        return False
    # Prefer folder layout with train/manifest.csv
    if (root / "train" / "manifest.csv").exists():
        return True
    # Some DGX layouts nest under dataset/
    if (root / "dataset" / "train" / "manifest.csv").exists():
        return True
    return False


def _resolve_root(root: Path) -> Path:
    if (root / "train" / "manifest.csv").exists():
        return root
    if (root / "dataset" / "train" / "manifest.csv").exists():
        return root / "dataset"
    raise FileNotFoundError(f"No train/manifest.csv under {root}")


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _row_to_sample(row: dict, split_dir: Path, split: str,
                   config: str = "default") -> Sample:
    rel = row.get("file") or row.get("path") or ""
    path = (split_dir / rel) if rel else None
    if path is not None and not path.exists():
        # Some layouts put audio/ under the split
        alt = split_dir / "audio" / Path(rel).name
        path = alt if alt.exists() else path

    intent_raw = row.get("command") or row.get("intent") or ""
    oos = _parse_bool(row.get("out_of_scope")) or (
        str(intent_raw).upper() in {OUT_OF_SCOPE, "UNKNOWN", "NONE", ""}
    )
    intent = OUT_OF_SCOPE if oos else (alias_intent(intent_raw) or intent_raw)
    slot = normalize_slot(row.get("slot_value") or "")
    variation = (row.get("variation") or "").strip()
    transcript = row.get("transcript") or variation or ""

    var = None
    if variation:
        var = lookup_variation(variation)
    if var is None and transcript and not oos:
        var = lookup_variation(transcript)
    variation_id = var.variation_id if var else -1
    if var is not None:
        intent = var.intent
        slot = var.slot_value
        variation = var.phrase

    return Sample(
        path=path,
        audio=None,
        split=split,
        intent=intent if not oos else OUT_OF_SCOPE,
        slot_value="" if oos else slot,
        variation="" if oos else variation,
        variation_id=-1 if oos else variation_id,
        transcript=transcript,
        out_of_scope=oos,
        speaker_id=str(row.get("speaker_id") or row.get("speaker") or ""),
        source=str(row.get("source") or ""),
        is_synthetic=_parse_bool(row.get("is_synthetic")),
        accent_group=str(row.get("accent_group") or ""),
        duration_s=float(row.get("duration_s") or row.get("duration_sec") or 0.0),
        file=rel,
        bucket=str(row.get("bucket") or ""),
        note=str(row.get("note") or ""),
        config=config,
    )


def load_split_samples(root: Path, split: str,
                       config: str = "default") -> List[Sample]:
    """Load one official or optional split from local folder layout."""
    root = _resolve_root(root)
    if config == "default":
        split_dir = root / split
        manifest = split_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        samples = []
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                samples.append(_row_to_sample(row, split_dir, split, config))
        return samples

    # Optional configs
    cfg_dir = root / config.replace("_", "-")
    if not cfg_dir.exists():
        cfg_dir = root / config
    # HF export style: supplemental_fil/train/ or synthetic-negatives/train/
    candidates = [
        cfg_dir / split / "manifest.csv",
        cfg_dir / "manifest.csv",
        root / config / split / "manifest.csv",
        root / config.replace("_", "-") / split / "manifest.csv",
    ]
    for man in candidates:
        if man.exists():
            split_dir = man.parent
            samples = []
            with open(man, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # supplemental_synth: only add train-voice rows to train
                    voice_split = (row.get("voice_split") or "").strip()
                    if voice_split and voice_split != split and split == "train":
                        if voice_split != "train":
                            continue
                    samples.append(_row_to_sample(row, split_dir, split, config))
            return samples
    raise FileNotFoundError(
        f"Could not find manifest for config={config} split={split} under {root}"
    )


def load_hf_split(split: str, config: str = "default",
                  revision: Optional[str] = None) -> List[Sample]:
    """Load via Hugging Face datasets (network / cache)."""
    from datasets import load_dataset
    kwargs = {"path": HF_DATASET_ID}
    if config != "default":
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(**kwargs)
    if split not in ds:
        raise KeyError(f"Split {split!r} not in HF config {config}: {list(ds)}")
    samples: List[Sample] = []
    for row in ds[split]:
        audio_arr = None
        path = None
        if "audio" in row and row["audio"] is not None:
            audio_arr = np.asarray(row["audio"]["array"], dtype=np.float32)
            path = Path(row["audio"].get("path") or row.get("file") or "")
        intent_raw = row.get("command") or ""
        oos = bool(row.get("out_of_scope")) or (
            str(intent_raw).upper() == OUT_OF_SCOPE
        )
        variation = (row.get("variation") or "").strip()
        var = lookup_variation(variation) if variation else None
        samples.append(Sample(
            path=path if path and str(path) else None,
            audio=audio_arr,
            split=split,
            intent=OUT_OF_SCOPE if oos else (
                var.intent if var else (alias_intent(intent_raw) or intent_raw)
            ),
            slot_value="" if oos else (
                var.slot_value if var else normalize_slot(row.get("slot_value") or "")
            ),
            variation="" if oos else (var.phrase if var else variation),
            variation_id=-1 if oos else (var.variation_id if var else -1),
            transcript=row.get("transcript") or variation or "",
            out_of_scope=oos,
            speaker_id=str(row.get("speaker_id") or ""),
            source=str(row.get("source") or ""),
            is_synthetic=bool(row.get("is_synthetic")),
            accent_group=str(row.get("accent_group") or ""),
            duration_s=float(row.get("duration_s") or 0.0),
            file=str(row.get("file") or ""),
            bucket=str(row.get("bucket") or ""),
            note=str(row.get("note") or ""),
            config=config,
        ))
    return samples


def make_speaker_val_split(
    train_samples: Sequence[Sample],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample]]:
    """Deterministic speaker-disjoint validation fold from train only."""
    speakers = sorted({s.speaker_id for s in train_samples if s.speaker_id})
    rng = np.random.default_rng(seed)
    order = speakers[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction))) if order else 0
    val_speakers: Set[str] = set(order[:n_val])
    train, val = [], []
    for s in train_samples:
        if s.speaker_id and s.speaker_id in val_speakers:
            val.append(s)
        else:
            train.append(s)
    return train, val


def check_speaker_leakage(splits: Dict[str, Sequence[Sample]]) -> dict:
    sets = {k: {s.speaker_id for s in v if s.speaker_id} for k, v in splits.items()}
    leaks = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sorted(sets[a] & sets[b])
            if inter:
                leaks[f"{a}∩{b}"] = inter
    return {
        "speakers": {k: len(v) for k, v in sets.items()},
        "leaks": leaks,
        "ok": not leaks,
    }


def dataset_fingerprint(samples_by_split: Dict[str, Sequence[Sample]],
                        root: Optional[Path] = None,
                        revision: Optional[str] = None) -> dict:
    h = hashlib.sha256()
    summary = {}
    for split, samples in sorted(samples_by_split.items()):
        files = sorted(s.file or (s.path.name if s.path else s.transcript)
                       for s in samples)
        payload = "\n".join(files).encode("utf-8")
        h.update(split.encode())
        h.update(payload)
        summary[split] = {
            "n": len(samples),
            "n_oos": sum(1 for s in samples if s.out_of_scope),
            "n_synthetic": sum(1 for s in samples if s.is_synthetic),
            "n_real": sum(1 for s in samples if not s.is_synthetic),
            "n_speakers": len({s.speaker_id for s in samples if s.speaker_id}),
            "intents": sorted({s.intent for s in samples}),
            "hours": round(sum(s.duration_s for s in samples) / 3600.0, 4),
        }
    return {
        "sha256_file_list": h.hexdigest(),
        "root": str(root) if root else None,
        "hf_dataset": HF_DATASET_ID,
        "revision": revision,
        "splits": summary,
        "speaker_leakage": check_speaker_leakage(samples_by_split),
    }


class GoldDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], max_seconds: float = 6.0,
                 pad: bool = True):
        self.samples = list(samples)
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self.pad = pad

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if s.audio is not None:
            audio = s.audio.astype(np.float32)
        elif s.path is not None:
            audio = load_wav(s.path)
        else:
            raise RuntimeError(f"Sample has neither audio nor path: {s.to_meta()}")
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif self.pad and len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        feat = normalize_feat(log_mel(audio))
        return {
            "feat": torch.from_numpy(feat),
            "T": feat.shape[0],
            "intent": s.intent,
            "slot": s.slot_value,
            "variation": s.variation,
            "variation_id": s.variation_id,
            "transcript": s.transcript,
            "out_of_scope": s.out_of_scope,
            "speaker_id": s.speaker_id,
            "source": s.source,
            "is_synthetic": s.is_synthetic,
            "accent_group": s.accent_group,
            "duration_s": s.duration_s,
            "file": s.file,
            "audio_ms": int(len(audio) / SAMPLE_RATE * 1000),
        }


def collate_batch(batch: List[dict]) -> dict:
    feats = [b["feat"] for b in batch]
    Ts = torch.tensor([b["T"] for b in batch], dtype=torch.long)
    max_t = int(Ts.max())
    n_mels = feats[0].shape[1]
    padded = torch.zeros(len(batch), max_t, n_mels)
    for i, f in enumerate(feats):
        padded[i, :f.shape[0]] = f
    return {
        "feat": padded,
        "T": Ts,
        "intent": [b["intent"] for b in batch],
        "slot": [b["slot"] for b in batch],
        "variation": [b["variation"] for b in batch],
        "variation_id": torch.tensor([b["variation_id"] for b in batch],
                                     dtype=torch.long),
        "transcripts": [b["transcript"] for b in batch],
        "out_of_scope": torch.tensor([b["out_of_scope"] for b in batch],
                                     dtype=torch.bool),
        "speaker_id": [b["speaker_id"] for b in batch],
        "source": [b["source"] for b in batch],
        "is_synthetic": torch.tensor([b["is_synthetic"] for b in batch],
                                     dtype=torch.bool),
        "accent_group": [b["accent_group"] for b in batch],
        "file": [b["file"] for b in batch],
        "audio_ms": torch.tensor([b["audio_ms"] for b in batch], dtype=torch.long),
    }


def build_mixture(
    root: Optional[Path],
    *,
    use_hf: bool = False,
    revision: Optional[str] = None,
    include_negatives: bool = False,
    include_fil: bool = False,
    include_synth_extra: bool = False,
    include_numerals: bool = False,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[Sample]]:
    """Assemble train/val/test; holdout is loaded separately and never trained."""
    if use_hf or root is None:
        train = load_hf_split("train", revision=revision)
        test = load_hf_split("test", revision=revision)
        if include_negatives:
            train += load_hf_split("train", "synthetic_negatives", revision)
        if include_fil:
            train += load_hf_split("train", "supplemental_fil", revision)
        if include_synth_extra:
            extra = load_hf_split("train", "supplemental_synth", revision)
            train += [s for s in extra if True]  # HF rows already train-only config
        if include_numerals:
            train += load_hf_split("numerals", revision=revision)
    else:
        root = _resolve_root(root)
        train = load_split_samples(root, "train")
        test = load_split_samples(root, "test")
        if include_negatives:
            try:
                train += load_split_samples(root, "train", "synthetic_negatives")
            except FileNotFoundError:
                pass
        if include_fil:
            try:
                train += load_split_samples(root, "train", "supplemental_fil")
            except FileNotFoundError:
                pass
        if include_synth_extra:
            try:
                train += load_split_samples(root, "train", "supplemental_synth")
            except FileNotFoundError:
                pass
        if include_numerals:
            try:
                train += load_split_samples(root, "numerals", "numerals")
            except FileNotFoundError:
                pass

    train, val = make_speaker_val_split(train, val_fraction=val_fraction, seed=seed)
    return {"train": train, "val": val, "test": test}


def make_dataloaders(
    samples_by_split: Dict[str, Sequence[Sample]],
    batch_size: int = 32,
    num_workers: int = 0,
    max_seconds: float = 6.0,
) -> Dict[str, DataLoader]:
    out = {}
    for name, samples in samples_by_split.items():
        ds = GoldDataset(samples, max_seconds=max_seconds)
        out[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_batch,
        )
    return out
