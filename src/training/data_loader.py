"""
Data loader for converting JSONL physics scenes to LLM training format.

Converts JSONL simulation data (header + frames) to next-frame prediction format
suitable for LLM training with text-based input/output.
"""

import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List


def load_physics_scene(jsonl_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load a physics scene from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        Tuple of (header, frames) where:
        - header: Scene metadata dict with keys like seed, description, object_count, gravity, timestep, objects
        - frames: List of frame dicts with keys like frame, description, objects
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene file not found: {jsonl_path}")

    with open(path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"Scene file must have at least header + 1 frame: {jsonl_path}")

    # First line is header
    header = json.loads(lines[0])
    if header.get("type") != "scene_header":
        raise ValueError(f"First line must be scene_header, got: {header.get('type')}")

    # Remaining lines are frames
    frames = [json.loads(line) for line in lines[1:]]

    return header, frames


def _format_frame_text(frame: Dict[str, Any]) -> str:
    """Format a single frame as text lines."""
    lines = []
    lines.append(f"Frame {frame['frame']}: {frame['description']}")
    for obj in frame['objects']:
        pos = obj['position']
        vel = obj['velocity']
        angle = obj.get('angle', 0)
        ang_vel = obj.get('angular_velocity', 0)
        obj_str = f"  obj_{obj['id']}: pos=({pos['x']:.4f}, {pos['y']:.4f}), vel=({vel['x']:.4f}, {vel['y']:.4f})"
        if abs(angle) > 0.001 or abs(ang_vel) > 0.001:
            obj_str += f", a={angle:.4f}, av={ang_vel:.4f}"
        lines.append(obj_str)
    lines.append("")
    return "\n".join(lines)


def _format_header_text(header: Dict[str, Any]) -> str:
    """Format scene header as text lines."""
    lines = []
    lines.append(f"Scene: {header['description']}")
    gravity = header['gravity']
    lines.append(f"Gravity: ({gravity['x']}, {gravity['y']})")
    lines.append(f"Timestep: {header['timestep']:.5f}")

    if header.get('scenario_type'):
        lines.append(f"Type: {header['scenario_type']}")
    if header.get('difficulty'):
        lines.append(f"Difficulty: {header['difficulty']}")

    static_geom = header.get('static_geometry', [])
    if static_geom:
        geom_parts = []
        for sg in static_geom:
            if sg['type'] == 'segment':
                p1, p2 = sg['p1'], sg['p2']
                geom_parts.append(f"seg ({p1['x']:.0f},{p1['y']:.0f})-({p2['x']:.0f},{p2['y']:.0f})")
            elif sg['type'] == 'circle':
                c = sg['center']
                geom_parts.append(f"peg ({c['x']:.0f},{c['y']:.0f}) r={sg['radius']:.0f}")
        if geom_parts:
            lines.append(f"Static: {'; '.join(geom_parts)}")

    constraints = header.get('constraints', [])
    if constraints:
        con_parts = []
        for c in constraints:
            con_parts.append(f"{c['type']} {c['body_a']}->{c['body_b']}")
        lines.append(f"Constraints: {'; '.join(con_parts)}")

    lines.append("")
    return "\n".join(lines)


def format_training_example(
    header: Dict[str, Any],
    context_frames: List[Dict[str, Any]],
    target_frame: Dict[str, Any],
    max_total_tokens: int = 0,
) -> Tuple[str, str]:
    """
    Format a training example as text for LLM.

    When max_total_tokens > 0, dynamically fills context frames to fit within
    the token budget. Header and output are always preserved; oldest context
    frames are dropped first. Uses chars/1.83 as token estimate.

    Args:
        header: Scene header with metadata
        context_frames: List of frames 0..N-1 (context)
        target_frame: Frame N to predict (target)
        max_total_tokens: Token budget (0 = unlimited)

    Returns:
        Tuple of (input_text, output_text) for training
    """
    CHARS_PER_TOKEN = 1.83  # measured empirically on physics data

    header_text = _format_header_text(header)
    output_text = _format_frame_text(target_frame).rstrip("\n")
    # Reformat output: "Frame N:" header without description
    output_lines = [f"Frame {target_frame['frame']}:"]
    for obj in target_frame['objects']:
        pos = obj['position']
        vel = obj['velocity']
        angle = obj.get('angle', 0)
        ang_vel = obj.get('angular_velocity', 0)
        obj_str = f"  obj_{obj['id']}: pos=({pos['x']:.4f}, {pos['y']:.4f}), vel=({vel['x']:.4f}, {vel['y']:.4f})"
        if abs(angle) > 0.001 or abs(ang_vel) > 0.001:
            obj_str += f", a={angle:.4f}, av={ang_vel:.4f}"
        output_lines.append(obj_str)
    output_text = "\n".join(output_lines)

    suffix = "Predict next frame:"
    # Fixed chars: header + suffix + separator + output + separator between input/output
    fixed_chars = len(header_text) + len(suffix) + len(output_text) + 4

    if max_total_tokens > 0:
        max_total_chars = int(max_total_tokens * CHARS_PER_TOKEN)
        char_budget = max_total_chars - fixed_chars

        # Add frames from most recent backward until budget is exhausted
        frame_texts = []
        total_frame_chars = 0
        for frame in reversed(context_frames):
            ft = _format_frame_text(frame)
            if total_frame_chars + len(ft) > char_budget:
                break
            frame_texts.append(ft)
            total_frame_chars += len(ft)
        # Reverse back to chronological order
        frame_texts.reverse()
    else:
        frame_texts = [_format_frame_text(frame) for frame in context_frames]

    input_text = header_text + "".join(frame_texts) + suffix

    return input_text, output_text


def jsonl_to_training_examples(
    jsonl_path: str,
    min_context_frames: int = 1,
    max_context_frames: int = None,
    max_total_tokens: int = 0,
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Convert a JSONL scene to training examples using sliding window.

    For a scene with N frames, generates N-1 examples:
    - Example 1: context=[frame1], target=frame2
    - Example 2: context=[frame1, frame2], target=frame3
    - ...
    - Example N-1: context=[frame1..frameN-1], target=frameN

    Args:
        jsonl_path: Path to the JSONL file
        min_context_frames: Minimum number of context frames (default 1)
        max_context_frames: Maximum context frames (None = no limit)
        max_total_tokens: Max tokens for input+output (0 = unlimited).
            When set, context frames are dynamically trimmed (oldest first)
            to fit within the token budget while preserving header and output.

    Yields:
        Tuples of (input_text, output_text, metadata) where metadata contains:
        - seed: Scene random seed
        - object_count: Number of objects
        - frame_index: Target frame index (1-indexed)
        - context_length: Number of context frames used
    """
    header, frames = load_physics_scene(jsonl_path)

    # For each frame after the first, create a training example
    for i in range(min_context_frames, len(frames)):
        # Determine context window
        if max_context_frames is not None:
            start_idx = max(0, i - max_context_frames)
        else:
            start_idx = 0

        ctx_frames = frames[start_idx:i]
        target_frame = frames[i]

        input_text, output_text = format_training_example(
            header, ctx_frames, target_frame, max_total_tokens=max_total_tokens,
        )

        metadata = {
            "seed": header["seed"],
            "object_count": header["object_count"],
            "frame_index": target_frame["frame"],
            "context_length": len(ctx_frames),
            "scene_path": str(jsonl_path),
        }

        yield input_text, output_text, metadata


def load_scene_header(jsonl_path: str) -> Dict[str, Any]:
    """
    Load only the header from a JSONL scene file (for efficient scanning).

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        Header dict with scene metadata
    """
    with open(jsonl_path, 'r') as f:
        header = json.loads(f.readline())
    return header
