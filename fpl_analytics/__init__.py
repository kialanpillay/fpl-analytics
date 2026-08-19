"""Season-long Fantasy Premier League analytics and squad optimisation."""

__version__ = "2.0.0"

from fpl_analytics.pipeline import AnalysisBundle, run_pipeline

__all__ = ["AnalysisBundle", "run_pipeline", "__version__"]
