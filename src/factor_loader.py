"""Dynamic discovery and execution of factor plugins."""

from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
import re

import pandas as pd

from src.p03_factor_processing import validate_factor_data


FACTOR_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def load_factor_module(factor_name: str) -> ModuleType:
    """Import ``src.factors.<factor_name>`` and verify its public interface."""
    if not isinstance(factor_name, str):
        raise TypeError("factor_name must be a string")
    if not FACTOR_NAME_PATTERN.fullmatch(factor_name):
        raise ValueError("factor_name may contain only letters, digits and underscores")

    try:
        module = import_module(f"src.factors.{factor_name}")
    except ModuleNotFoundError as error:
        if error.name == f"src.factors.{factor_name}":
            raise ValueError(f"unknown factor: {factor_name}") from error
        raise

    if not callable(getattr(module, "calculate_factor", None)):
        raise ValueError(f"factor module {factor_name} has no calculate_factor function")
    return module


def calculate_factor(
    factor_name: str,
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run one factor plugin and validate its standard output."""
    module = load_factor_module(factor_name)
    defaults = dict(getattr(module, "DEFAULT_PARAMETERS", {}))
    defaults.update(dict(parameters or {}))
    result = module.calculate_factor(start_date, end_date, defaults)
    return validate_factor_data(result)

