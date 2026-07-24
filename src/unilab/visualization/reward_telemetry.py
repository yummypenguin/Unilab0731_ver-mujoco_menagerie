"""Backend-neutral reward telemetry for interactive playback overlays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _first_scalar(value: Any) -> float | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    scalar = float(array.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else None


def parse_reward_value_keys(value: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize optional reward-term selectors to ``reward/<name>`` keys."""
    raw_keys = value.split(",") if isinstance(value, str) else value
    normalized: list[str] = []
    for raw_key in raw_keys:
        key = str(raw_key).strip()
        if not key:
            continue
        if not key.startswith("reward/"):
            key = f"reward/{key}"
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


@dataclass
class RewardTelemetry:
    """Track current reward terms, deltas, and the actual episode return."""

    selected_keys: tuple[str, ...] = ()
    max_terms: int = 12
    step_reward: float = 0.0
    episode_return: float = 0.0
    current_terms: dict[str, float] = field(default_factory=dict)
    term_deltas: dict[str, float] = field(default_factory=dict)
    _last_step: int | None = field(default=None, init=False, repr=False)
    _last_log_signature: tuple[tuple[str, float], ...] | None = field(
        default=None, init=False, repr=False
    )
    _last_log_object: Mapping[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.selected_keys = parse_reward_value_keys(self.selected_keys)
        self.max_terms = max(int(self.max_terms), 1)

    def update(
        self,
        reward: Any,
        info: Mapping[str, Any] | None,
        *,
        advanced: bool,
    ) -> None:
        """Consume one playback state without counting paused render frames."""
        if not advanced:
            return

        step = _first_scalar(info.get("steps")) if info is not None else None
        if step is not None:
            step_int = int(step)
            if self._last_step is not None and step_int < self._last_step:
                self.episode_return = 0.0
            self._last_step = step_int

        step_reward = _first_scalar(reward)
        if step_reward is not None:
            self.step_reward = step_reward
            self.episode_return += step_reward

        log = info.get("log") if info is not None else None
        if not isinstance(log, Mapping):
            return

        sampled_terms: list[tuple[str, float]] = []
        for raw_key, raw_value in log.items():
            key = str(raw_key)
            if not key.startswith("reward/"):
                continue
            value = _first_scalar(raw_value)
            if value is not None:
                sampled_terms.append((key, value))

        signature = tuple(sampled_terms)
        if not signature:
            return
        if log is self._last_log_object and signature == self._last_log_signature:
            return
        self._last_log_object = log
        self._last_log_signature = signature

        previous = self.current_terms
        self.current_terms = dict(sampled_terms)
        self.term_deltas = {key: value - previous.get(key, value) for key, value in sampled_terms}

    def visible_terms(self) -> list[tuple[str, float, float]]:
        """Return configured terms in stable display order, keeping total last."""
        if self.selected_keys:
            keys = [key for key in self.selected_keys if key in self.current_terms]
        else:
            keys = list(self.current_terms)

        total_key = "reward/total"
        keys = [key for key in keys if key != total_key]
        include_total = total_key in self.current_terms and (
            not self.selected_keys or total_key in self.selected_keys
        )
        if include_total:
            keys = keys[: max(self.max_terms - 1, 0)]
            keys.append(total_key)
        else:
            keys = keys[: self.max_terms]
        return [
            (
                key.removeprefix("reward/"),
                self.current_terms[key],
                self.term_deltas.get(key, 0.0),
            )
            for key in keys
        ]

    def overlay_columns(self) -> tuple[str, str]:
        """Build MuJoCo overlay label/value columns."""
        labels = ["Reward telemetry", "Step reward", "Episode return"]
        values = [
            "current (delta)",
            f"{self.step_reward:+.4f}",
            f"{self.episode_return:+.4f}",
        ]
        for name, value, delta in self.visible_terms():
            labels.append(name)
            values.append(f"{value:+.4f} ({delta:+.4f})")
        return "\n".join(labels), "\n".join(values)


__all__ = ["RewardTelemetry", "parse_reward_value_keys"]
