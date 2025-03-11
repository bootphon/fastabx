"""Pytest configuration."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """CLI arguments."""
    parser.addoption("--item", action="store", default=None, help="Path to the item file")
    parser.addoption("--features", action="store", default=None, help="Path to the features directory")


def pytest_configure(config: pytest.Config) -> None:
    """Global configuration."""
    config.reference_scores = {  # HuBERT base L11
        "triphone-dev-clean.item": {
            ("within", "within"): 0.03074,
            ("across", "within"): 0.03777,
        },
        "phoneme-dev-clean.item": {
            ("within", "within"): 0.01579,
            ("across", "within"): 0.02216,
            ("within", "any"): 0.07738,
            ("across", "any"): 0.08357,
        },
    }
    config.distance = "cosine"
    config.max_size_group = 50
    config.max_x_across = 10
    config.seed = 0
    config.frequency = 50
