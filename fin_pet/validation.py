"""Validation utilities and custom exceptions for fin_pet.

This module provides strongly-typed, defensive validation helpers used across the library,
and a set of custom exception classes with clear error messages.
"""

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np


class FinPetError(Exception):
    """Base exception for all fin_pet errors."""


class FinPetTypeError(FinPetError, TypeError):
    """Raised when an input has an invalid type."""


class FinPetValueError(FinPetError, ValueError):
    """Raised when an input has an invalid numeric value or violates constraints."""


class FinPetKeyError(FinPetError, KeyError):
    """Raised when a required key or label is missing."""


Number = Union[int, float, np.number]


def ensure_numeric(value: object, name: str) -> float:
    """Ensure a value can be safely interpreted as a finite float.

    Parameters
    ----------
    value : object
        Input value to validate.
    name : str
        Human-readable name of the value for error messages.

    Returns
    -------
    float
        The value cast to float.

    Raises
    ------
    FinPetTypeError
        If value cannot be cast to float.
    FinPetValueError
        If value is NaN or infinite.
    """

    try:
        numeric_value = float(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        raise FinPetTypeError(f"{name} must be a numeric type, got {type(value)}") from exc
    if not np.isfinite(numeric_value):
        raise FinPetValueError(f"{name} must be a finite float, got {numeric_value}")
    return numeric_value


def ensure_positive(value: Number, name: str, allow_zero: bool = False) -> float:
    """Ensure a numeric value is positive (or non-negative if allow_zero=True)."""

    numeric_value = ensure_numeric(value, name)
    if allow_zero and numeric_value < 0.0:
        raise FinPetValueError(f"{name} must be >= 0, got {numeric_value}")
    if not allow_zero and numeric_value <= 0.0:
        raise FinPetValueError(f"{name} must be > 0, got {numeric_value}")
    return numeric_value


def ensure_probability(value: Number, name: str) -> float:
    """Ensure a numeric value is in [0, 1]."""

    numeric_value = ensure_numeric(value, name)
    if numeric_value < 0.0 or numeric_value > 1.0:
        raise FinPetValueError(f"{name} must be within [0, 1], got {numeric_value}")
    return numeric_value


def ensure_array_like(values: Union[Sequence[Number], Number], name: str) -> np.ndarray:
    """Ensure input is convertible to a 1D numpy array of finite floats.

    If a single scalar is provided, it will be converted to a length-1 array.
    """

    if isinstance(values, (list, tuple, np.ndarray)):
        array = np.asarray(values, dtype=float)
    else:
        # Single value becomes a 1-element array
        array = np.asarray([ensure_numeric(values, name)], dtype=float)
    if array.ndim != 1:
        raise FinPetTypeError(f"{name} must be a 1D array-like, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise FinPetValueError(f"{name} must contain only finite floats")
    return array


def ensure_bounds(value: Number, name: str, lower: Optional[float] = None, upper: Optional[float] = None) -> float:
    """Ensure a numeric value is within optional [lower, upper] bounds (inclusive)."""

    numeric_value = ensure_numeric(value, name)
    if lower is not None and numeric_value < lower:
        raise FinPetValueError(f"{name} must be >= {lower}, got {numeric_value}")
    if upper is not None and numeric_value > upper:
        raise FinPetValueError(f"{name} must be <= {upper}, got {numeric_value}")
    return numeric_value


def ensure_str_in(value: object, name: str, allowed: Iterable[str]) -> str:
    """Ensure a string value is within an allowed set (case sensitive)."""

    if not isinstance(value, str):
        raise FinPetTypeError(f"{name} must be a string, got {type(value)}")
    allowed_set = set(allowed)
    if value not in allowed_set:
        raise FinPetValueError(f"{name} must be one of {sorted(allowed_set)}, got '{value}'")
    return value


def safe_divide(numerator: Number, denominator: Number, name: str) -> float:
    """Safely divide two numbers, guarding against zero division and non-finite results."""

    num = ensure_numeric(numerator, f"{name}.numerator")
    den = ensure_numeric(denominator, f"{name}.denominator")
    if den == 0.0:
        raise FinPetValueError(f"{name} denominator must be non-zero")
    result = num / den
    if not np.isfinite(result):
        raise FinPetValueError(f"{name} result must be finite, got {result}")
    return result


