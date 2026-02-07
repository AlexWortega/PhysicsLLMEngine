"""
Visualization tools for physics prediction evaluation.

Generates GIFs showing predicted vs ground truth trajectories
for visual inspection of model performance.
"""

from typing import List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_trajectory_gif(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
    output_path: str,
    fps: int = 10,
    scene_bounds: Tuple[float, float, float, float] = (0, 800, 0, 600),
    title: str = "",
    object_size: float = 50.0,
) -> str:
    """
    Create a GIF comparing predicted vs ground truth trajectories.

    Generates a side-by-side animation showing predicted positions (left)
    and ground truth positions (right) for visual comparison.

    Args:
        pred_positions: Predicted positions, shape (num_frames, num_objects, 2)
        true_positions: Ground truth positions, shape (num_frames, num_objects, 2)
        output_path: Path to save the output GIF
        fps: Frames per second for the GIF (default 10)
        scene_bounds: (x_min, x_max, y_min, y_max) for plot bounds
        title: Optional title prefix for the animation
        object_size: Size of scatter plot markers (default 50)

    Returns:
        output_path on success

    Raises:
        ValueError: If array shapes don't match or are invalid
    """
    # Validate inputs
    if pred_positions.shape != true_positions.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_positions.shape} vs true {true_positions.shape}"
        )

    if len(pred_positions.shape) != 3 or pred_positions.shape[2] != 2:
        raise ValueError(
            f"Expected shape (num_frames, num_objects, 2), got {pred_positions.shape}"
        )

    num_frames = pred_positions.shape[0]
    x_min, x_max, y_min, y_max = scene_bounds

    # Create figure with side-by-side subplots
    fig, (ax_pred, ax_true) = plt.subplots(1, 2, figsize=(14, 6))

    # Set up axes
    for ax, label in [(ax_pred, "Predicted"), (ax_true, "Ground Truth")]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    # Initialize scatter plots
    scatter_pred = ax_pred.scatter([], [], s=object_size, c="blue", alpha=0.7)
    scatter_true = ax_true.scatter([], [], s=object_size, c="green", alpha=0.7)

    # Title text for frame counter
    title_prefix = f"{title} - " if title else ""
    fig_title = fig.suptitle(f"{title_prefix}Frame 0/{num_frames}")

    def init():
        """Initialize animation."""
        scatter_pred.set_offsets(np.empty((0, 2)))
        scatter_true.set_offsets(np.empty((0, 2)))
        return scatter_pred, scatter_true, fig_title

    def update(frame_idx: int):
        """Update animation for each frame."""
        # Get positions for this frame
        pred_frame = pred_positions[frame_idx]  # (num_objects, 2)
        true_frame = true_positions[frame_idx]  # (num_objects, 2)

        # Update scatter plots
        scatter_pred.set_offsets(pred_frame)
        scatter_true.set_offsets(true_frame)

        # Update title with frame counter
        fig_title.set_text(f"{title_prefix}Frame {frame_idx + 1}/{num_frames}")

        return scatter_pred, scatter_true, fig_title

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        blit=False,
        interval=1000 // fps,  # milliseconds between frames
    )

    # Save as GIF using PillowWriter
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    finally:
        # Always close figure to prevent memory leaks
        plt.close(fig)

    return output_path


def create_trajectory_overlay_gif(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
    output_path: str,
    fps: int = 10,
    scene_bounds: Tuple[float, float, float, float] = (0, 800, 0, 600),
    title: str = "",
    show_trails: bool = True,
    trail_length: int = 5,
) -> str:
    """
    Create a GIF overlaying predicted and ground truth on same plot.

    Shows both predictions (blue) and ground truth (green) on a single
    plot for direct visual comparison of errors.

    Args:
        pred_positions: Predicted positions, shape (num_frames, num_objects, 2)
        true_positions: Ground truth positions, shape (num_frames, num_objects, 2)
        output_path: Path to save the output GIF
        fps: Frames per second (default 10)
        scene_bounds: (x_min, x_max, y_min, y_max) for plot bounds
        title: Optional title for the animation
        show_trails: Whether to show position trails (default True)
        trail_length: Number of frames to show in trail (default 5)

    Returns:
        output_path on success
    """
    # Validate inputs
    if pred_positions.shape != true_positions.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_positions.shape} vs true {true_positions.shape}"
        )

    num_frames = pred_positions.shape[0]
    x_min, x_max, y_min, y_max = scene_bounds

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    # Legend
    ax.scatter([], [], c="blue", alpha=0.7, label="Predicted")
    ax.scatter([], [], c="green", alpha=0.7, label="Ground Truth")
    ax.legend(loc="upper right")

    # Main scatter plots
    scatter_pred = ax.scatter([], [], s=80, c="blue", alpha=0.7)
    scatter_true = ax.scatter([], [], s=80, c="green", alpha=0.7)

    # Trail scatter plots (smaller, more transparent)
    trail_scatters_pred = []
    trail_scatters_true = []
    if show_trails:
        for i in range(trail_length):
            alpha = 0.1 + 0.1 * (i / trail_length)
            size = 20 + 20 * (i / trail_length)
            trail_scatters_pred.append(
                ax.scatter([], [], s=size, c="blue", alpha=alpha)
            )
            trail_scatters_true.append(
                ax.scatter([], [], s=size, c="green", alpha=alpha)
            )

    # Title
    title_prefix = f"{title} - " if title else ""
    fig_title = ax.set_title(f"{title_prefix}Frame 0/{num_frames}")

    def update(frame_idx: int):
        """Update animation for each frame."""
        # Current positions
        pred_frame = pred_positions[frame_idx]
        true_frame = true_positions[frame_idx]

        scatter_pred.set_offsets(pred_frame)
        scatter_true.set_offsets(true_frame)

        # Update trails
        if show_trails:
            for i, (trail_pred, trail_true) in enumerate(
                zip(trail_scatters_pred, trail_scatters_true)
            ):
                trail_idx = frame_idx - (trail_length - i)
                if trail_idx >= 0:
                    trail_pred.set_offsets(pred_positions[trail_idx])
                    trail_true.set_offsets(true_positions[trail_idx])
                else:
                    trail_pred.set_offsets(np.empty((0, 2)))
                    trail_true.set_offsets(np.empty((0, 2)))

        ax.set_title(f"{title_prefix}Frame {frame_idx + 1}/{num_frames}")

        return [scatter_pred, scatter_true] + trail_scatters_pred + trail_scatters_true

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=False,
        interval=1000 // fps,
    )

    # Save
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    finally:
        plt.close(fig)

    return output_path


def create_comparison_gif(
    pred_positions: np.ndarray,
    true_positions: np.ndarray,
    output_path: str,
    step_errors: Optional[List[float]] = None,
    fps: int = 10,
    scene_bounds: Tuple[float, float, float, float] = (0, 800, 0, 600),
    title: str = "",
) -> str:
    """
    Create a GIF with GT vs Predicted overlay + error lines + error bar.

    Single panel showing:
    - Green circles = ground truth positions
    - Red circles = predicted positions
    - Gray dashed lines connecting GT-to-Pred for each object (error vectors)
    - Bottom bar chart showing per-step MSE

    Args:
        pred_positions: Predicted positions, shape (num_frames, num_objects, 2)
        true_positions: Ground truth positions, shape (num_frames, num_objects, 2)
        output_path: Path to save the output GIF
        step_errors: Optional list of per-step errors for error bar
        fps: Frames per second (default 10)
        scene_bounds: (x_min, x_max, y_min, y_max) for plot bounds
        title: Optional title for the animation

    Returns:
        output_path on success
    """
    if pred_positions.shape != true_positions.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_positions.shape} vs true {true_positions.shape}"
        )

    num_frames, num_objects, _ = pred_positions.shape
    x_min, x_max, y_min, y_max = scene_bounds

    # Compute step errors if not provided
    if step_errors is None:
        step_errors = []
        for f in range(num_frames):
            mse = float(np.mean((pred_positions[f] - true_positions[f]) ** 2))
            step_errors.append(mse)

    # Create figure: main scene on top, error bar on bottom
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax_scene = fig.add_subplot(gs[0])
    ax_error = fig.add_subplot(gs[1])

    # Scene axes
    ax_scene.set_xlim(x_min, x_max)
    ax_scene.set_ylim(y_min, y_max)
    ax_scene.set_aspect("equal")
    ax_scene.grid(True, alpha=0.2)

    # Legend entries
    ax_scene.scatter([], [], s=60, c="#2ecc71", edgecolors="black",
                     linewidth=0.5, label="Ground Truth")
    ax_scene.scatter([], [], s=60, c="#e74c3c", edgecolors="black",
                     linewidth=0.5, label="Predicted")
    ax_scene.legend(loc="upper right", fontsize=9)

    # Scatter for GT and Pred
    scatter_true = ax_scene.scatter([], [], s=60, c="#2ecc71",
                                    edgecolors="black", linewidth=0.5, zorder=5)
    scatter_pred = ax_scene.scatter([], [], s=60, c="#e74c3c",
                                    edgecolors="black", linewidth=0.5, zorder=5)

    # Error lines (one per object)
    error_lines = []
    for _ in range(num_objects):
        line, = ax_scene.plot([], [], color="gray", linewidth=0.8,
                              linestyle="--", alpha=0.5, zorder=3)
        error_lines.append(line)

    title_prefix = f"{title} - " if title else ""
    scene_title = ax_scene.set_title(
        f"{title_prefix}Frame 0/{num_frames}", fontsize=11)

    # Error bar axes
    max_err = max((e for e in step_errors if np.isfinite(e)), default=1.0) * 1.2
    if max_err == 0:
        max_err = 1.0
    ax_error.set_xlim(0, num_frames)
    ax_error.set_ylim(0, max_err)
    ax_error.set_xlabel("Frame", fontsize=9)
    ax_error.set_ylabel("Position MSE", fontsize=9)
    ax_error.grid(True, alpha=0.2)

    # Pre-draw all bars as light gray
    bar_colors = ["#cccccc"] * num_frames
    bars = ax_error.bar(range(num_frames), step_errors, color=bar_colors,
                        width=0.8, alpha=0.6)
    # Highlight marker
    highlight_bar = ax_error.axvline(x=0, color="#e74c3c", linewidth=1.5, alpha=0.8)
    error_text = ax_error.text(
        0.02, 0.9, "", transform=ax_error.transAxes, fontsize=8,
        verticalalignment="top")

    def update(frame_idx: int):
        pred_frame = pred_positions[frame_idx]
        true_frame = true_positions[frame_idx]

        scatter_true.set_offsets(true_frame)
        scatter_pred.set_offsets(pred_frame)

        # Error lines connecting each GT-Pred pair
        for obj_i in range(num_objects):
            px, py = pred_frame[obj_i]
            tx, ty = true_frame[obj_i]
            error_lines[obj_i].set_data([tx, px], [ty, py])

        scene_title.set_text(
            f"{title_prefix}Frame {frame_idx + 1}/{num_frames}")

        # Update error bar highlight
        highlight_bar.set_xdata([frame_idx])
        # Color bars: past=blue, current=red, future=gray
        for i, bar in enumerate(bars):
            if i < frame_idx:
                bar.set_color("#3498db")
                bar.set_alpha(0.7)
            elif i == frame_idx:
                bar.set_color("#e74c3c")
                bar.set_alpha(1.0)
            else:
                bar.set_color("#cccccc")
                bar.set_alpha(0.4)

        err_val = step_errors[frame_idx] if frame_idx < len(step_errors) else 0
        error_text.set_text(f"MSE: {err_val:.4f}")

        return ([scatter_true, scatter_pred, highlight_bar, scene_title,
                 error_text] + error_lines + list(bars))

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, blit=False,
        interval=1000 // fps,
    )

    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    finally:
        plt.close(fig)

    return output_path
