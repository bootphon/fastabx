"""Check that the configuration of the convolutions is compatible with the item file."""

# ruff: noqa: D101, D102, D103, S603, T201
import argparse
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from fastabx.dataset import dummy_dataset_from_item, find_all_files


class NoAudioError(ValueError):
    def __init__(self, extension: str, root: Path | str) -> None:
        super().__init__(f"No audio found with extension {extension} in {root}")


def num_samples(path: str | Path) -> int:
    cmd = ["soxi", "-s", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr) from error
    return int(out)


def get_all_num_samples(root: str | Path, extension: str) -> dict[str, int]:
    files = find_all_files(root, extension)
    with ThreadPoolExecutor() as executor:
        lengths = list(executor.map(num_samples, files.values()))
    return dict(zip(files, lengths, strict=True))


@dataclass(frozen=True)
class Conv1d:
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1

    @classmethod
    def from_string(cls, inp: str) -> "Conv1d":
        values = [int(part) for part in inp.split(",")]
        if len(values) > 4:
            raise ValueError(inp)
        return cls(*values)


def conv_length(input_length: int, convs: list[Conv1d]) -> int:
    length = torch.tensor(input_length, dtype=torch.int64)
    for conv in convs:
        length = torch.div(
            length + 2 * conv.padding - conv.dilation * (conv.kernel_size - 1) - 1 + conv.stride,
            conv.stride,
            rounding_mode="floor",
        )
    return int(length)


CONVS_20MS = "10,5-3,2-3,2-3,2-3,2-2,2-2,2"
CONVS_40MS = "10,5-3,2-3,2-3,2-3,2-2,2-2,2-2,2"
CONVS_80MS = "10,5-3,2-3,2-3,2-3,2-2,2-2,2-2,2-2,2"


def invalid_entries_in_item(
    item: Path,
    root: Path,
    frequency: int = 50,
    convs_string: str = "CONVS_20MS",
    extension: str = ".wav",
) -> pl.DataFrame:
    if convs_string in (defaults := {"CONVS_20MS": CONVS_20MS, "CONVS_40MS": CONVS_40MS, "CONVS_80MS": CONVS_80MS}):
        convs_string = defaults[convs_string]
    convs = [Conv1d.from_string(conv) for conv in convs_string.split("-")]
    lengths = {name: conv_length(length, convs) for name, length in get_all_num_samples(root, extension).items()}
    if not lengths:
        raise NoAudioError(extension, root)
    return (
        dummy_dataset_from_item(item, frequency)
        .labels.select(pl.exclude("left", "right"))
        .with_columns(pl.col("#file").replace_strict(lengths).alias("feature_length"))
        .filter((pl.col("end") > pl.col("feature_length")) | (pl.col("end") <= pl.col("start")))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check item file and features")
    parser.add_argument("item", type=Path, help="Path to item file")
    parser.add_argument("root", type=Path, help="Root to audio directory")
    parser.add_argument("--frequency", type=float, default=50, help="Feature frequency in Hz. Default: 50")
    parser.add_argument("--extension", type=str, default=".wav", help="Extension of the audio files. Default: .wav")
    parser.add_argument(
        "--convs",
        type=str,
        help="Conv1d: kernel_size,stride,padding,dilation. "
        "The configs for each layer are separated by '-'. For a given layer, separate the values with ','. "
        "stride, padding and dilation are optional. "
        "Also accepts 'CONV_20MS', 'CONV_40MS' and 'CONV_80MS' for configs following wav2vec 2.0-like setting. "
        f"Default: 'CONVS_20MS' (equal to '{CONVS_20MS}')",
        default="CONVS_20MS",
    )
    args = parser.parse_args()

    invalid = invalid_entries_in_item(args.item, args.root, args.frequency, args.convs, args.extension)
    if len(invalid) == 0:
        print("Everything good!")
    else:
        path = f"invalid-{args.item.stem}-{args.frequency}-{uuid.uuid4()}.csv"
        print(f"{len(invalid)} entries are invalid. Saving them to {path}")
        invalid.write_csv(path)
