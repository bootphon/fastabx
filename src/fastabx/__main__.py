"""Entry point for the ZeroSpeech ABX evaluation."""

import argparse
import importlib.metadata
from argparse import ArgumentDefaultsHelpFormatter

from fastabx.distance import available_distances
from fastabx.utils import print_fastabx_output
from fastabx.zerospeech import zerospeech_abx


def main() -> None:
    """ZeroSpeech ABX evaluation."""
    parser = argparse.ArgumentParser(
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
        type=int,
        required=True,
        help="Maximum number of A, B, or X in a cell. Set to 10 in the original ZeroSpeech ABX. "
        "Disabled if negative value.",
    )
    parser.add_argument(
        "--max-x-across",
        type=int,
        help="With 'across', maximum number of X given (A, B). Set to 5 in the original ZeroSpeech ABX. "
        "Disabled if negative value.",
    )
    parser.add_argument("--frequency", type=int, default=50, help="Feature frequency (in Hz)")
    parser.add_argument("--speaker", choices=["within", "across"], default="within", help="Speaker mode")
    parser.add_argument("--context", choices=["within", "any"], default="within", help="Context mode")
    parser.add_argument("--distance", choices=available_distances(), default="angular", help="Distance")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
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
    )
    print_fastabx_output(score, **vars(args))


if __name__ == "__main__":
    main()
