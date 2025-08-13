# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "tqdm>=4.67.1",
#    "torch>=2.6",
#    "numpy>=2.2",
#    "h5features==1.4.1",
# ]
# ///
"""Utility to convert h5features to torch tensors, and back, intended to be used alongside ABXpy."""

import argparse
from pathlib import Path

import h5features
import numpy as np
import torch
from tqdm import tqdm


def torch_to_h5features(root: Path, dest: Path, step: float, group: str = "features") -> None:
    """Convert a list of torch tensors to a single h5features file."""
    items, labels, features = [], [], []
    for path in sorted(root.glob("*.pt")):
        feats = torch.load(path, map_location="cpu").squeeze().numpy(force=True)
        if feats.ndim != 2:
            raise ValueError(path)
        times = np.arange(step / 2, feats.shape[0] * step, step, dtype=np.float64)
        items.append(path.stem)
        labels.append(times)
        features.append(feats)
    dest.parent.mkdir(exist_ok=True, parents=True)
    h5features.write(dest, group, items, labels, features)


def torch_to_h5features_with_times(root_features: Path, root_times: Path, dest: Path, group: str = "features") -> None:
    """Convert lists of torch tensors (features and times) to a single h5features file."""
    items, labels, features = [], [], []
    for path in sorted(root_features.glob("*.pt")):
        feats = torch.load(path, map_location="cpu").squeeze().numpy(force=True)
        if feats.ndim != 2:
            raise ValueError(path)
        times = root_times / path.name
        items.append(path.stem)
        labels.append(times)
        features.append(feats)
    dest.parent.mkdir(exist_ok=True, parents=True)
    h5features.write(dest, group, items, labels, features)


def h5features_to_torch(path: Path, root_features: Path, root_times: Path) -> None:
    """Convert a h5features file to torch tensors."""
    root_features.mkdir(exist_ok=True, parents=True)
    root_times.mkdir(exist_ok=True, parents=True)
    times, features = h5features.read(path)
    for item, feats in tqdm(features.items()):
        torch.save(torch.as_tensor(feats, dtype=torch.float32), root_features / f"{item}.pt")
        torch.save(torch.as_tensor(times[item], dtype=torch.float64), root_times / f"{item}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversion between h5features and torch tensors")
    subparsers = parser.add_subparsers(dest="subcommand", help="Destination format")
    parser_to_torch = subparsers.add_parser("torch", help="Convert from h5features to torch")
    parser_to_h5 = subparsers.add_parser("h5", help="Convert from torch to h5features")
    parser_to_torch.add_argument("path", type=Path, help="The input h5features file")
    parser_to_torch.add_argument("features", type=Path, help="Destination to the features directory of torch tensors")
    parser_to_torch.add_argument("times", type=Path, help="Destination to the times directory of torch tensors")
    parser_to_h5.add_argument("features", type=Path, help="The path to the features directory of torch tensors")
    parser_to_h5.add_argument("destination", type=Path, help="The output h5features file")
    parser_to_h5.add_argument("--times", type=Path, help="Path to the optional directory of times tensor")
    parser_to_h5.add_argument("--step", type=float, help="Time step in seconds (should be 0.02 for 50Hz models)")
    args = parser.parse_args()

    if args.subcommand == "torch":
        h5features_to_torch(args.path, args.features, args.times)
    elif args.subcommand == "h5":
        if args.step is not None and args.times is None:
            torch_to_h5features(args.features, args.destination, args.step)
        elif args.step is None and args.times is not None:
            torch_to_h5features_with_times(args.features, args.times, args.destination)
        else:
            parser.error("Set exactly one of --times or --step")
    else:
        parser.error(f"Invalid subcommand: {args.subcommand}")
