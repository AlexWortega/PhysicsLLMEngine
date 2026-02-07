"""
Evaluation module for physics prediction models.

Provides metrics computation, OOD scene generation, report generation,
rollout evaluation, and visualization tools for evaluating model
performance on physics prediction tasks.

Key components:
- MetricsComputer: Position/velocity MSE, conservation metrics
- RolloutEvaluator: Autoregressive multi-step evaluation
- EvaluationRunner: Main orchestrator for complete evaluation
- OOD generator: Out-of-distribution test scenarios
- Report generator: Markdown report generation
- Visualization: Trajectory GIF generation
"""

from src.evaluation.metrics import MetricsComputer
from src.evaluation.rollout import RolloutEvaluator
from src.evaluation.runner import EvaluationRunner
from src.evaluation.ood_generator import (
    generate_ood_scene,
    generate_ood_test_suite,
    OOD_TYPES,
)
from src.evaluation.report import generate_evaluation_report

__all__ = [
    "MetricsComputer",
    "RolloutEvaluator",
    "EvaluationRunner",
    "generate_ood_scene",
    "generate_ood_test_suite",
    "OOD_TYPES",
    "generate_evaluation_report",
]

# Lazy import for visualization (requires matplotlib)
try:
    from src.evaluation.visualization import (
        create_trajectory_gif,
        create_trajectory_overlay_gif,
    )
    __all__.extend(["create_trajectory_gif", "create_trajectory_overlay_gif"])
except ImportError:
    pass
