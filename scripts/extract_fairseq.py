# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tqdm>=4.67.1",
#     "torch>=2.6",
#     "numpy>=2.2",
# ]
# ///
"""Utility to unpack concatenated fairseq features."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def read_manifest(path: Path | str) -> list[tuple[Path, int]]:
    """Read TSV fairseq-like manifest file."""
    path = Path(path)
    if path.suffix != ".tsv":
        raise ValueError(path)
    with path.open("r") as f:
        root = Path(f.readline().strip())
        manifest = []
        for line in f.readlines():
            file, num_samples = line.strip().split("\t")
            manifest.append((root / file, num_samples))
    return manifest


def fairseq_to_torch(output: Path, tsv: Path, npy_file: Path, len_file: Path) -> None:
    """Extract concatenated fairseq features."""
    output.mkdir(exist_ok=True)
    manifest = read_manifest(tsv)
    with Path(len_file).open("r") as file:
        lengths = [int(length) for length in file.read().splitlines()]
    fairseq_feats, start_idx = np.load(npy_file, mmap_mode="r+"), 0
    already_seen = set()
    for length, (file, _) in tqdm(list(zip(lengths, manifest, strict=True))):
        tensor = torch.from_numpy(fairseq_feats[start_idx : start_idx + length])
        if file.stem in already_seen:
            raise ValueError(file)
        torch.save(tensor, output / file.with_suffix(".pt").name)
        already_seen.add(file.stem)
        start_idx += length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract concatenated fairseq features to individual PyTorch tensors")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("tsv", type=Path, help="Path to manifest file")
    parser.add_argument("npy", type=Path, help="Path to concatenated features")
    parser.add_argument("len", type=Path, help="Path to generated len file")
    args = parser.parse_args()
    fairseq_to_torch(args.output, args.tsv, args.npy, args.len)
