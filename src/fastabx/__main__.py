"""Entry point for the ZeroSpeech ABX evaluation."""

import argparse
import typing as tp

from fastabx.distance import DistanceName
from fastabx.zerospeech import zerospeech_abx


def main() -> None:
    """ZeroSpeech ABX evaluation."""
    parser = argparse.ArgumentParser(description="ZeroSpeech ABX")
    parser.add_argument("item", help="Path to the item file")
    parser.add_argument("features", help="Path to the features directory")
    parser.add_argument("--frequency", type=int, default=50, help="Feature frequency (in Hz)")
    parser.add_argument("--speaker", choices=["within", "across"], default="within")
    parser.add_argument("--context", choices=["within", "any"], default="within")
    parser.add_argument("--distance", choices=tp.get_args(DistanceName), default="cosine")
    parser.add_argument("--max-size-group", type=int, default=10, help="Maximum size of a cell")
    parser.add_argument("--max-x-across", type=int, default=5, help="With 'across', maximum number of X given (A, B)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    score = zerospeech_abx(
        args.item,
        args.features,
        speaker=args.speaker,
        context=args.context,
        distance=args.distance,
        frequency=args.frequency,
        max_size_group=args.max_size_group,
        max_x_across=args.max_x_across,
        seed=args.seed,
    )
    print(f"ABX error rate: {score:.3%}")  # noqa: T201


if __name__ == "__main__":
    main()
