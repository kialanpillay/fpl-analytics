"""In-memory analysis bundle for the UI process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fpl_analytics.optimiser import DEFAULT_OBJECTIVES
from fpl_analytics.pipeline import AnalysisBundle, run_pipeline
from fpl_analytics.server.settings import load_ui_settings
from fpl_analytics.squad import DEFAULT_SQUAD_PATH


@dataclass
class AppState:
    bundle: AnalysisBundle | None = None
    last_params: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    solving: bool = False

    def run(
        self,
        *,
        horizon: int | None = None,
        force_refresh: bool = False,
        max_transfers: int = 2,
        objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
        bank: float | None = None,
        free_transfers: int | None = None,
    ) -> AnalysisBundle:
        settings = load_ui_settings()
        params = {
            "horizon": horizon if horizon is not None else settings["horizon"],
            "force_refresh": force_refresh,
            "max_transfers": max_transfers,
            "objectives": objectives,
            "bank": bank,
            "free_transfers": free_transfers,
        }
        self.solving = True
        self.last_error = None
        try:
            self.bundle = run_pipeline(
                squad_path=DEFAULT_SQUAD_PATH,
                horizon=int(params["horizon"]),
                force_refresh=force_refresh,
                objectives=objectives,
                max_transfers=max_transfers,
                bank=bank,
                free_transfers=free_transfers,
            )
            self.last_params = {**params, "fetched_at": self.bundle.fetched_at}
            return self.bundle
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            self.solving = False

    def get_bundle(self) -> AnalysisBundle:
        if self.bundle is None:
            return self.run()
        return self.bundle


STATE = AppState()
