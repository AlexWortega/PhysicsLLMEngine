"""
Data export functions for physics simulations.

Exports simulation data to JSONL format with:
- First line: scene metadata and description
- Following lines: one frame per line
"""

import json
from typing import TYPE_CHECKING

from .formats import format_scene_header, format_frame

if TYPE_CHECKING:
    from src.physics.simulation import PhysicsSimulation


def export_simulation(
    sim: "PhysicsSimulation",
    num_frames: int,
    output_path: str,
    seed: int = 0,
    metadata: dict = None,
) -> None:
    """
    Export simulation to JSONL file.

    First line: scene metadata and description
    Following lines: one frame per line

    Args:
        sim: The physics simulation to export
        num_frames: Number of frames to simulate and export
        output_path: Path to the output JSONL file
        seed: The random seed used (for metadata)
        metadata: Optional scenario metadata (from generate_scenario)
    """
    with open(output_path, 'w') as f:
        # Write scene header (initial state before any simulation)
        header = format_scene_header(sim, seed, metadata=metadata)
        f.write(json.dumps(header) + '\n')

        # Step simulation and write each frame
        for frame_num in range(num_frames):
            sim.step()
            frame_data = format_frame(sim, frame_num + 1)  # 1-indexed frames
            f.write(json.dumps(frame_data) + '\n')


def export_to_dict(
    sim: "PhysicsSimulation",
    num_frames: int,
    seed: int = 0,
    metadata: dict = None,
) -> list:
    """
    Export simulation to a list of dicts (for testing).

    Args:
        sim: The physics simulation to export
        num_frames: Number of frames to simulate and export
        seed: The random seed used (for metadata)
        metadata: Optional scenario metadata (from generate_scenario)

    Returns:
        List of dicts, first is header, rest are frames
    """
    result = []

    # Add scene header
    header = format_scene_header(sim, seed, metadata=metadata)
    result.append(header)

    # Step simulation and capture each frame
    for frame_num in range(num_frames):
        sim.step()
        frame_data = format_frame(sim, frame_num + 1)
        result.append(frame_data)

    return result
