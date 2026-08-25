"""Local UI settings. Horizon is persisted; manager id is a constant / env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from fpl_analytics.live import resolve_manager_id

ROOT = Path(__file__).resolve().parent.parent.parent
UI_CONFIG_PATH = ROOT / "config" / "ui.yaml"


def load_ui_settings() -> dict[str, Any]:
    data: dict[str, Any] = {}
    if UI_CONFIG_PATH.exists():
        data = yaml.safe_load(UI_CONFIG_PATH.read_text()) or {}
    return {
        "manager_id": resolve_manager_id(),
        "horizon": int(data.get("horizon") or os.environ.get("FPL_HORIZON") or 6),
    }


def save_ui_settings(horizon: int | None = None) -> dict[str, Any]:
    current = load_ui_settings()
    if horizon is not None:
        current["horizon"] = int(horizon)
    UI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_CONFIG_PATH.write_text(yaml.safe_dump({"horizon": current["horizon"]}, sort_keys=False))
    return current
