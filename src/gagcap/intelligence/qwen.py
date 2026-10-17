"""Optional CPU-only Qwen GGUF event extraction; never emits synthetic results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ExtractedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_type: Literal["sanctions", "conflict", "trade_restriction", "export_rule",
                        "energy_crisis", "central_bank", "trade_relations", "other"]
    event_status: Literal["CONFIRMED_EVENT", "UNVERIFIED_REPORT", "ANALYST_ASSUMPTION"]
    summary: str = Field(min_length=1, max_length=600)
    entities: list[str]
    countries: list[str]
    sectors: list[str]
    assets: list[str]
    evidence_quotes: list[str]
    source_url: str
    published_at: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}(T.*(Z|[+-]\d{2}:\d{2}))?$")
    event_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    unsupported_claims: list[str]


@dataclass(frozen=True, slots=True)
class QwenResult:
    status: Literal["AVAILABLE", "QWEN_UNAVAILABLE", "INVALID_MODEL_OUTPUT"]
    event: ExtractedEvent | None
    reason: str | None
    calibration_status: str = "UNCALIBRATED"


def _schema_prompt(text: str, source_url: str) -> str:
    schema = json.dumps(ExtractedEvent.model_json_schema(), ensure_ascii=False)
    return ("Extract only facts explicitly supported by SOURCE_TEXT. Do not infer prices, rates, "
            "probabilities, causality, or facts absent from the source. Use empty arrays/null when "
            "unknown. Include verbatim evidence quotes for every event claim. Return JSON only.\n"
            f"SCHEMA={schema}\nSOURCE_URL={source_url}\nSOURCE_TEXT={text}")


def extract_event(text: str, *, source_url: str, model_path: str | None = None,
                  context_size: int = 4096, max_tokens: int = 700,
                  cpu_threads: int | None = None) -> QwenResult:
    path = Path(model_path or os.environ.get("QWEN_GGUF_PATH", ""))
    parsed_url = urlparse(source_url)
    if not text.strip() or parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username:
        return QwenResult("INVALID_MODEL_OUTPUT", None, "Non-empty source text and safe absolute HTTPS URL are required")
    if not model_path and not os.environ.get("QWEN_GGUF_PATH"):
        return QwenResult("QWEN_UNAVAILABLE", None, "Configure QWEN_GGUF_PATH in Colab")
    if not path.is_file():
        return QwenResult("QWEN_UNAVAILABLE", None, f"GGUF file is absent: {path}")
    if context_size < 512 or max_tokens < 1 or (cpu_threads is not None and cpu_threads < 1):
        return QwenResult("QWEN_UNAVAILABLE", None, "Invalid CPU runtime limits")
    try:
        from llama_cpp import Llama

        threads = cpu_threads or min(8, max(1, os.cpu_count() or 1))
        model = Llama(model_path=str(path), n_ctx=context_size, n_threads=threads,
                      n_gpu_layers=0, verbose=False)
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": _schema_prompt(text, source_url)}],
            temperature=0.0, max_tokens=max_tokens, response_format={"type": "json_object"})
        raw = response["choices"][0]["message"]["content"]
        parsed = ExtractedEvent.model_validate_json(raw)
        if parsed.source_url != source_url:
            return QwenResult("INVALID_MODEL_OUTPUT", None, "Output source URL does not match supplied document")
        if parsed.published_at:
            from datetime import datetime
            published = datetime.fromisoformat(parsed.published_at.replace("Z", "+00:00"))
            if published.tzinfo is None:
                return QwenResult("INVALID_MODEL_OUTPUT", None, "Publication timestamp must include timezone")
        quoted_source = text.casefold()
        if not parsed.evidence_quotes or any(quote.casefold() not in quoted_source
                                              for quote in parsed.evidence_quotes):
            return QwenResult("INVALID_MODEL_OUTPUT", None, "Evidence quotes are not verbatim source spans")
        if parsed.unsupported_claims:
            return QwenResult("INVALID_MODEL_OUTPUT", None, "Output declares unsupported claims")
        cited_text = " ".join(parsed.evidence_quotes).casefold()
        extracted_entities = parsed.entities + parsed.countries + parsed.sectors + parsed.assets
        if any(entity.casefold() not in cited_text for entity in extracted_entities):
            return QwenResult("INVALID_MODEL_OUTPUT", None, "An extracted entity is absent from cited evidence")
        # LLMs cannot establish source authority or human confirmation; extracted claims start unverified.
        parsed.event_status = "UNVERIFIED_REPORT"
        parsed.summary = " ".join(parsed.evidence_quotes)
        return QwenResult("AVAILABLE", parsed, None)
    except ImportError as exc:
        return QwenResult("QWEN_UNAVAILABLE", None, f"CPU runtime unavailable: {exc}")
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        return QwenResult("INVALID_MODEL_OUTPUT", None, f"Structured extraction rejected: {exc}")
    except Exception as exc:
        return QwenResult("QWEN_UNAVAILABLE", None, f"CPU runtime failed: {type(exc).__name__}: {exc}")


def calibrate_confidence(raw_scores: list[float], labels: list[int]) -> dict[str, float | str]:
    """Fit a one-dimensional logistic calibration only with real labeled validation data."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if len(x) != len(y) or len(y) < 30 or not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Calibration requires >=30 finite labeled cases containing both classes")
    calibrator = LogisticRegression().fit(x, y)
    return {"method": "logistic_platt", "status": "FITTED_UNVERIFIED",
            "intercept": float(calibrator.intercept_[0]), "slope": float(calibrator.coef_[0, 0])}
