"""BASE/BULL/BEAR macro scenarios without invented probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


ScenarioName = Literal["BASE", "BULL", "BEAR"]


@dataclass(frozen=True, slots=True)
class MacroScenario:
    name: ScenarioName
    assumptions: dict[str, float]
    evidence_refs: tuple[str, ...]
    probability: float | None = None
    probability_basis: str = "UNASSIGNED"

    def __post_init__(self) -> None:
        if not self.assumptions or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                       for v in self.assumptions.values()):
            raise ValueError("Scenario requires explicit numeric assumptions")
        if self.probability is None:
            if self.probability_basis != "UNASSIGNED":
                raise ValueError("A probability basis requires a probability")
        elif (not math.isfinite(self.probability) or not 0 <= self.probability <= 1
              or self.probability_basis not in {"EMPIRICAL", "EXPERT"}):
            raise ValueError("Probability must be labeled EMPIRICAL or EXPERT")


def validate_scenario_set(scenarios: list[MacroScenario]) -> None:
    expected = {"BASE", "BULL", "BEAR"}
    if len(scenarios) != 3 or {scenario.name for scenario in scenarios} != expected:
        raise ValueError("Provide exactly one BASE, BULL and BEAR scenario")
    probabilities = [scenario.probability for scenario in scenarios]
    if any(probability is not None for probability in probabilities):
        if any(probability is None for probability in probabilities):
            raise ValueError("Either assign probabilities to all scenarios or none")
        if abs(sum(float(probability) for probability in probabilities) - 1) > 1e-8:
            raise ValueError("Scenario probabilities must sum to 1")
