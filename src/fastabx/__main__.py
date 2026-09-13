"""Entry point for the ZeroSpeech ABX evaluation."""

import argparse
import importlib.metadata
from argparse import ArgumentDefaultsHelpFormatter

from fastabx.utils import print_fastabx_output
from fastabx.verify import MIN_A_LEN
from fastabx.zerospeech import zerospeech_abx


def subsample_size(value: str) -> int:
    """Parse a subsampling size: an integer of at least ``MIN_A_LEN``, or a negative one to disable it."""
    try:
        size = int(value)
    except ValueError:
        msg = f"invalid integer value: {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if 0 <= size < MIN_A_LEN:
        msg = f"must be at least {MIN_A_LEN}, or negative to disable the subsampling, not {size}"
        raise argparse.ArgumentTypeError(msg)
    return size


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError:
        msg = f"invalid integer value: {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if parsed < 1:
        msg = f"must be strictly positive, not {parsed}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``fastabx`` CLI. Also used by sphinx-argparse to document it."""
    parser = argparse.ArgumentParser(
        prog="fastabx",
        description="ZeroSpeech ABX",
        allow_abbrev=False,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {importlib.metadata.version('fastabx')}"
    )
    parser.add_argument("item", help="Path to the item file")
    parser.add_argument("features", help="Path to the features directory")
    parser.add_argument(
        "--max-size-group",
        type=subsample_size,
        required=True,
        help="Maximum number of A, B, or X in a cell, at least 2. Set to 10 in the original ZeroSpeech ABX. "
        "Disabled if negative value.",
    )
    parser.add_argument(
        "--max-x-across",
        type=subsample_size,
        help="With 'across', maximum number of X given (A, B), at least 2. Set to 5 in the original "
        "ZeroSpeech ABX. Disabled if negative value.",
    )
    parser.add_argument("--frequency", type=positive_int, default=50, help="Feature frequency (in Hz)")
    parser.add_argument("--speaker", choices=["within", "across"], default="within", help="Speaker mode")
    parser.add_argument("--context", choices=["within", "any"], default="within", help="Context mode")
    parser.add_argument(
        "--distance",
        choices=["angular", "euclidean", "kl_symmetric", "identical"],
        default="angular",
        help="Distance",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        default=None,
        help="Device on which to store the features, such as 'cpu' or 'cuda:1'. "
        "Defaults to CUDA if available, and CPU otherwise.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format. 'text' prints the ABX error rate, 'json' a single object with the score and "
        "every argument.",
    )
    parser.add_argument(
        "--write-csv",
        default=None,
        metavar="PATH",
        help="Write the score of every cell to this CSV file",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Hide the progress bars shown while building the dataset and scoring the cells",
    )
    return parser


def main() -> None:
    """ZeroSpeech ABX evaluation."""
    parser = build_parser()
    args = parser.parse_args()

    if args.max_x_across is None and args.speaker == "across":
        parser.error("--max-x-across is required when using 'across' speaker mode")
    score = zerospeech_abx(
        args.item,
        args.features,
        max_size_group=args.max_size_group if args.max_size_group >= 0 else None,
        max_x_across=args.max_x_across if args.max_x_across is not None and args.max_x_across >= 0 else None,
        speaker=args.speaker,
        context=args.context,
        distance=args.distance,
        frequency=args.frequency,
        seed=args.seed,
        device=args.device,
        write_csv=args.write_csv,
        progress=not args.quiet,
    )
    arguments = dict(vars(args))
    print_fastabx_output(score, output=arguments.pop("output"), **arguments)


if __name__ == "__main__":  # pragma: no cover
    main()
