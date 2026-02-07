"""
Hybrid format functions for physics simulation data.

Combines human-readable text descriptions with structured numerical data
optimized for LLM training and tokenization.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.physics.simulation import PhysicsSimulation


def generate_scene_description(sim: "PhysicsSimulation") -> str:
    """
    Generate human-readable scene description.

    Args:
        sim: The physics simulation

    Returns:
        A descriptive string like "A scene with 23 objects (15 circles, 8 rectangles) under gravity."
    """
    bodies = sim.bodies
    total = len(bodies)

    # Count shape types
    circles = sum(1 for b in bodies if getattr(b, 'shape_type', '') == 'circle')
    rectangles = sum(1 for b in bodies if getattr(b, 'shape_type', '') == 'rectangle')

    # Build description
    if circles > 0 and rectangles > 0:
        shape_desc = f"{circles} circles and {rectangles} rectangles"
    elif circles > 0:
        shape_desc = f"{circles} circles"
    elif rectangles > 0:
        shape_desc = f"{rectangles} rectangles"
    else:
        shape_desc = "objects"

    # Describe gravity
    gx, gy = sim.space.gravity
    if gy < 0:
        gravity_desc = "under downward gravity"
    elif gy > 0:
        gravity_desc = "under upward gravity"
    elif gx != 0:
        gravity_desc = "under horizontal gravity"
    else:
        gravity_desc = "in zero gravity"

    return f"A scene with {total} objects ({shape_desc}) {gravity_desc}."


def format_frame(sim: "PhysicsSimulation", frame_num: int) -> dict:
    """
    Format a single frame as hybrid data.

    Args:
        sim: The physics simulation
        frame_num: The frame number

    Returns:
        dict with frame number, description, and object states
    """
    state = sim.get_state()

    # Count moving objects (velocity > threshold)
    VELOCITY_THRESHOLD = 1.0
    moving = sum(
        1 for obj in state['objects']
        if abs(obj['velocity']['x']) > VELOCITY_THRESHOLD
        or abs(obj['velocity']['y']) > VELOCITY_THRESHOLD
    )

    # Generate frame description
    if moving == 0:
        motion_desc = "All objects are at rest"
    elif moving == len(state['objects']):
        motion_desc = "All objects are in motion"
    else:
        motion_desc = f"{moving} of {len(state['objects'])} objects are moving"

    return {
        "frame": frame_num,
        "description": f"Frame {frame_num}: {motion_desc}.",
        "objects": state['objects']
    }


def format_scene_header(sim: "PhysicsSimulation", seed: int, metadata: dict = None) -> dict:
    """
    Format the scene header (first line of JSONL output).

    Args:
        sim: The physics simulation
        seed: The random seed used to generate the scene
        metadata: Optional scenario metadata (from generate_scenario)

    Returns:
        dict with scene metadata and object initial states
    """
    if metadata and "description" in metadata:
        description = metadata["description"]
    else:
        description = generate_scene_description(sim)

    # Get initial state of all objects
    objects = []
    for body in sim.bodies:
        obj = {
            "id": getattr(body, 'object_id', len(objects)),
            "type": getattr(body, 'shape_type', 'unknown'),
            "position": {"x": round(body.position.x, 4), "y": round(body.position.y, 4)},
            "material": getattr(body, 'material', {}),
        }
        # Add size info if available
        size_info = getattr(body, 'size_info', {})
        if size_info:
            obj.update(size_info)
        objects.append(obj)

    header = {
        "type": "scene_header",
        "seed": seed,
        "description": description,
        "object_count": len(objects),
        "gravity": {"x": sim.space.gravity[0], "y": sim.space.gravity[1]},
        "timestep": sim.DT,
        "objects": objects,
    }

    # Add scenario-specific fields if present
    if metadata:
        for key in ("scenario_type", "scenario_category", "difficulty",
                     "static_geometry", "constraints"):
            if key in metadata and metadata[key]:
                header[key] = metadata[key]

    return header
