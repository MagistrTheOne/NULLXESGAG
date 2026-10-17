"""Rule-based interest-rate scenario calculations, not policy predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TaylorInputs:
    neutral_real_rate: float
    inflation: float
    inflation_target: float
    output_gap_percent: float
    inflation_weight: float = 0.5
    output_gap_weight: float = 0.5


def taylor_rate(inputs: TaylorInputs) -> float:
    vals = [inputs.neutral_real_rate, inputs.inflation, inputs.inflation_target,
            inputs.output_gap_percent, inputs.inflation_weight, inputs.output_gap_weight]
    if not all(map(lambda x: isinstance(x, (int, float)) and math.isfinite(x), vals)):
        raise ValueError("Taylor rule inputs must be finite numbers")
    return (inputs.neutral_real_rate + inputs.inflation
            + inputs.inflation_weight * (inputs.inflation - inputs.inflation_target)
            + inputs.output_gap_weight * inputs.output_gap_percent)


def rate_scenario(current_rate: float, shock_basis_points: float) -> float:
    if not math.isfinite(current_rate) or not math.isfinite(shock_basis_points):
        raise ValueError("Rate and shock must be finite")
    return current_rate + shock_basis_points / 100
