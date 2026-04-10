import argparse
import os
from pathlib import Path

import numpy as np


def build_splits(subject_ids, num_splits=5):
    """
    Build K-fold style split list:
    each item: {"train": [...], "test": [...], "val": [...]}
    """
    ids = sorted(subject_ids)
    n = len(ids)
    if n < 3:
        raise ValueError("Need at least 3 subjects to build train/test/val splits.")

    folds = []
    for k in range(num_splits):
        test = [sid for i, sid in enumerate(ids) if i % num_splits == k]
        val = [sid for i, sid in enumerate(ids) if i % num_splits == (k + 1) % num_splits]
        train = [sid for sid in ids if sid not in test and sid not in val]

        # Keep split valid in very small datasets.
        if len(train) == 0:
            train = ids[:]
            for sid in test + val:
                if sid in train:
                    train.remove(sid)
        if len(train) == 0:
            raise RuntimeError("Failed to build non-empty train set.")

        folds.append({"train": train, "test": test, "val": val})
    return folds


def generate_subject_npz(
    save_path,
    n_epochs,
    seed,
    eeg_channels=1,
    hbo_channels=6,
    hb_channels=6,
    ppg_channels=12,
    eeg_epoch_samples=3000,
    low_sr_epoch_samples=750,
    num_classes=5,
):
    rng = np.random.default_rng(seed)

    # One epoch = 30 seconds
    # EEG/EOG @100Hz => 3000 points, low-rate modalities @25Hz => 750 points
    eeg = rng.normal(0.0, 1.0, size=(n_epochs, eeg_channels, eeg_epoch_samples)).astype(np.float32)
    hbo = rng.normal(0.0, 1.0, size=(n_epochs, hbo_channels, low_sr_epoch_samples)).astype(np.float32)
    hb = rng.normal(0.0, 1.0, size=(n_epochs, hb_channels, low_sr_epoch_samples)).astype(np.float32)
    ppg = rng.normal(0.0, 1.0, size=(n_epochs, ppg_channels, low_sr_epoch_samples)).astype(np.float32)
    eog = rng.normal(0.0, 1.0, size=(n_epochs, eeg_epoch_samples)).astype(np.float32)

    # label per epoch, expected by loader.py
    label = rng.integers(0, num_classes, size=(n_epochs,), dtype=np.int64)

    # score is currently optional in training, but keep a compatible field.
    # Expected convention in comments: (n_modalities, seq_len, 8)
    score = rng.random(size=(3, n_epochs, 8), dtype=np.float32)

    # Save with names expected by loader.py
    np.savez_compressed(
        save_path,
        eeg=eeg,
        hbo=hbo,
        hb=hb,
        ppg=ppg,
        eog=eog,
        label=label,
        score=score,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate random dummy EFSleep-format data.")
    parser.add_argument("--root-dir", type=str, default=".", help="Project root (same meaning as config dataset.root_dir)")
    parser.add_argument("--dataset-name", type=str, default="EFSleep", help="Dataset folder name under dset/")
    parser.add_argument("--eeg-channel-dir", type=str, default="Fpz", help="Channel subdir under npz/")
    parser.add_argument("--num-subjects", type=int, default=18, help="How many subject files to generate")
    parser.add_argument("--num-epochs", type=int, default=80, help="Epochs per subject file")
    parser.add_argument("--num-splits", type=int, default=5, help="Number of folds in split_idx")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    npz_dir = root_dir / "dset" / args.dataset_name / "npz" / args.eeg_channel_dir
    split_dir = root_dir / "split_idx"
    npz_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = list(range(1, args.num_subjects + 1))
    for sid in subject_ids:
        # Naming pattern compatible with loader subject parsing: int(name[1:3])
        # Example: S01D1.npz, S12D1.npz
        fname = f"S{sid:02d}D1.npz"
        save_path = npz_dir / fname
        generate_subject_npz(
            save_path=save_path,
            n_epochs=args.num_epochs,
            seed=args.seed + sid,
        )

    split_list = build_splits(subject_ids, num_splits=args.num_splits)
    split_path = split_dir / f"idx_{args.dataset_name}.npy"
    np.save(split_path, split_list)

    print(f"[OK] Dummy npz saved to: {npz_dir}")
    print(f"[OK] Split file saved to: {split_path}")
    print(f"[INFO] subjects={args.num_subjects}, epochs_per_subject={args.num_epochs}, num_splits={args.num_splits}")


if __name__ == "__main__":
    main()
