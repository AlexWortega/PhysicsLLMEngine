"""
Markdown report generator for physics prediction evaluation.

Generates comprehensive evaluation reports with:
- Per-model performance sections
- Curriculum breakdown tables
- Rollout stability analysis
- OOD performance breakdown
- Model comparison summary

Uses pandas DataFrame.to_markdown() for clean table formatting.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


def _format_float(value: float, decimals: int = 6) -> str:
    """Format float with fixed decimal places."""
    return f"{value:.{decimals}f}"


def _format_metric(metric_dict: Dict[str, float], decimals: int = 6) -> str:
    """Format a metric dict with mean +/- std."""
    mean = metric_dict.get("mean", 0.0)
    std = metric_dict.get("std", 0.0)
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def _generate_header(test_set_size: int) -> str:
    """Generate report header with title and metadata."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# Physics Prediction Evaluation Report

**Generated:** {timestamp}
**Test Set Size:** {test_set_size:,} scenes

---

"""


def _generate_overall_metrics_table(model_results: Dict[str, Any]) -> str:
    """Generate table of overall metrics for a model."""
    rows = []
    rows.append({
        "Metric": "Position MSE",
        "Mean": _format_float(model_results.get("position_mse", {}).get("mean", 0.0)),
        "Std": _format_float(model_results.get("position_mse", {}).get("std", 0.0)),
    })
    rows.append({
        "Metric": "Velocity MSE",
        "Mean": _format_float(model_results.get("velocity_mse", {}).get("mean", 0.0)),
        "Std": _format_float(model_results.get("velocity_mse", {}).get("std", 0.0)),
    })
    rows.append({
        "Metric": "Energy Violation",
        "Mean": _format_float(model_results.get("energy_violation", {}).get("mean", 0.0)),
        "Std": _format_float(model_results.get("energy_violation", {}).get("std", 0.0)),
    })
    rows.append({
        "Metric": "Momentum Violation",
        "Mean": _format_float(model_results.get("momentum_violation", {}).get("mean", 0.0)),
        "Std": _format_float(model_results.get("momentum_violation", {}).get("std", 0.0)),
    })

    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def _generate_curriculum_table(model_results: Dict[str, Any]) -> str:
    """Generate curriculum breakdown table."""
    curriculum = model_results.get("by_curriculum_stage", {})
    if not curriculum:
        return "*No curriculum data available*"

    rows = []
    # Sort stages by start number
    stages = sorted(curriculum.keys(), key=lambda x: int(x.split("-")[0]))

    for stage in stages:
        stage_data = curriculum[stage]
        pos_mse = stage_data.get("position_mse", {})
        vel_mse = stage_data.get("velocity_mse", {})
        rows.append({
            "Stage (Objects)": stage,
            "Position MSE": _format_metric(pos_mse),
            "Velocity MSE": _format_metric(vel_mse),
        })

    df = pd.DataFrame(rows)
    return df.to_markdown(index=False)


def _generate_rollout_section(model_results: Dict[str, Any]) -> str:
    """Generate rollout stability section."""
    rollout = model_results.get("rollout", {})
    if not rollout:
        return "*No rollout data available*"

    lines = []

    # Divergence step
    divergence_step = rollout.get("divergence_step")
    if divergence_step is not None:
        lines.append(f"**Divergence Step:** {divergence_step}")
    else:
        lines.append("**Divergence Step:** None (stable throughout)")

    # Step errors summary
    step_errors = rollout.get("step_errors", [])
    if step_errors:
        import numpy as np
        errors_arr = np.array(step_errors)
        lines.append(f"**Step Errors:**")
        lines.append(f"- Initial (step 0): {errors_arr[0]:.6f}")
        lines.append(f"- Final (step {len(errors_arr)-1}): {errors_arr[-1]:.6f}")
        lines.append(f"- Mean: {np.mean(errors_arr):.6f}")
        lines.append(f"- Max: {np.max(errors_arr):.6f}")

    return "\n".join(lines)


def _generate_ood_section(model_results: Dict[str, Any]) -> str:
    """Generate OOD performance section."""
    ood = model_results.get("ood", {})
    if not ood:
        return "*No OOD data available*"

    sections = []

    # By type table
    by_type = ood.get("by_type", {})
    if by_type:
        rows = []
        for ood_type, metrics in sorted(by_type.items()):
            pos_mse = metrics.get("position_mse", {})
            rows.append({
                "OOD Type": ood_type,
                "Position MSE": _format_metric(pos_mse),
            })

        df = pd.DataFrame(rows)
        sections.append("**Performance by OOD Type:**\n")
        sections.append(df.to_markdown(index=False))

    # By distance (graceful degradation)
    by_distance = ood.get("by_distance", [])
    if by_distance:
        sections.append("\n**Graceful Degradation (by OOD distance):**\n")
        rows = []
        for distance, mean, std in by_distance:
            rows.append({
                "OOD Distance": _format_float(distance, 2),
                "Mean Error": _format_float(mean),
                "Std Error": _format_float(std),
            })
        df = pd.DataFrame(rows)
        sections.append(df.to_markdown(index=False))

    return "\n".join(sections)


def _generate_model_section(model_name: str, model_results: Dict[str, Any]) -> str:
    """Generate complete section for a single model."""
    lines = []

    lines.append(f"## {model_name}\n")

    # Overall metrics
    lines.append("### Overall Metrics\n")
    lines.append(_generate_overall_metrics_table(model_results))
    lines.append("")

    # Curriculum breakdown
    lines.append("### Curriculum Breakdown\n")
    lines.append(_generate_curriculum_table(model_results))
    lines.append("")

    # Rollout stability
    lines.append("### Rollout Stability\n")
    lines.append(_generate_rollout_section(model_results))
    lines.append("")

    # OOD performance
    lines.append("### Out-of-Distribution Performance\n")
    lines.append(_generate_ood_section(model_results))
    lines.append("")

    lines.append("---\n")

    return "\n".join(lines)


def _generate_comparison_summary(
    all_results: Dict[str, Dict[str, Any]],
) -> str:
    """Generate model comparison summary."""
    if len(all_results) < 2:
        return "*Comparison requires at least 2 models*"

    model_names = list(all_results.keys())
    metrics = ["position_mse", "velocity_mse", "energy_violation", "momentum_violation"]
    metric_labels = {
        "position_mse": "Position MSE",
        "velocity_mse": "Velocity MSE",
        "energy_violation": "Energy Violation",
        "momentum_violation": "Momentum Violation",
    }

    lines = []
    lines.append("## Model Comparison Summary\n")

    # Side-by-side comparison table
    rows = []
    for metric in metrics:
        row = {"Metric": metric_labels[metric]}
        best_model = None
        best_value = float("inf")

        for model_name in model_names:
            mean = all_results[model_name].get(metric, {}).get("mean", float("inf"))
            row[model_name] = _format_float(mean)
            if mean < best_value:
                best_value = mean
                best_model = model_name

        row["Winner"] = best_model if best_model else "-"
        rows.append(row)

    df = pd.DataFrame(rows)
    lines.append("### Side-by-Side Comparison\n")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    # Winner summary
    lines.append("### Summary\n")
    winner_counts: Dict[str, int] = {}
    for row in rows:
        winner = row.get("Winner", "-")
        if winner != "-":
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

    for model_name, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{model_name}** wins {count}/{len(metrics)} metrics")

    return "\n".join(lines)


def _generate_visualization_section(
    visualization_paths: Optional[List[str]] = None,
) -> str:
    """Generate section with visualization references."""
    lines = []
    lines.append("## Visualizations\n")

    if not visualization_paths:
        lines.append("*No visualizations generated for this report.*\n")
        lines.append("*Run visualization pipeline to generate trajectory GIFs.*")
        return "\n".join(lines)

    lines.append("The following trajectory visualizations are available:\n")
    for path in visualization_paths:
        filename = Path(path).name
        lines.append(f"![{filename}]({path})")
        lines.append("")

    return "\n".join(lines)


def generate_evaluation_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: str,
    test_set_size: int = 10000,
    visualization_paths: Optional[List[str]] = None,
) -> str:
    """
    Generate comprehensive markdown evaluation report.

    Args:
        model_results: Dictionary mapping model names to their results.
            Each model's results should contain:
            - position_mse: {"mean": float, "std": float}
            - velocity_mse: {"mean": float, "std": float}
            - energy_violation: {"mean": float, "std": float}
            - momentum_violation: {"mean": float, "std": float}
            - by_curriculum_stage: {stage: {metric: {mean, std}}}
            - rollout: {step_errors: [...], divergence_step: int|None}
            - ood: {by_type: {...}, by_distance: [...]}
        output_path: Path to write the markdown report
        test_set_size: Number of test scenes evaluated
        visualization_paths: Optional list of paths to visualization GIFs

    Returns:
        Path to the written report file
    """
    sections = []

    # Header
    sections.append(_generate_header(test_set_size))

    # Per-model sections
    for model_name, results in model_results.items():
        sections.append(_generate_model_section(model_name, results))

    # Model comparison (if multiple models)
    if len(model_results) >= 2:
        sections.append(_generate_comparison_summary(model_results))
        sections.append("\n---\n")

    # Visualizations
    sections.append(_generate_visualization_section(visualization_paths))

    # Footer
    sections.append("\n---\n")
    sections.append("*Report generated by src/evaluation/report.py*")

    # Write report
    report_content = "\n".join(sections)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report_content)

    return str(output_file)
