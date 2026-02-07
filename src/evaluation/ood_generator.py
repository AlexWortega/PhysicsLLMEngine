"""
Out-of-distribution (OOD) scene generator for evaluation.

Generates physics scenes outside the training distribution to test
model generalization. OOD types include:
- Unseen object counts (fewer than 10, more than 50)
- Extreme materials (mass, friction, elasticity outside training ranges)
- Gravity variants (zero, low/moon, high/jupiter)

Training distribution reference:
- Object count: 10-50
- Mass: 0.5-10.0
- Friction: 0.1-1.0
- Elasticity: 0.1-0.95
- Gravity: (0, -981)
"""

import random
from typing import Dict, Iterator, Optional

from src.physics.simulation import PhysicsSimulation
from src.physics.objects import create_circle, create_rectangle, create_static_segment


# Scene bounds (matching scene_generator.py)
SCENE_WIDTH = 800
SCENE_HEIGHT = 600
GROUND_Y = 50
WALL_MARGIN = 10

# Training distribution reference values
TRAIN_OBJECT_COUNT_MIN = 10
TRAIN_OBJECT_COUNT_MAX = 50
TRAIN_MASS_MIN = 0.5
TRAIN_MASS_MAX = 10.0
TRAIN_FRICTION_MIN = 0.1
TRAIN_FRICTION_MAX = 1.0
TRAIN_ELASTICITY_MIN = 0.1
TRAIN_ELASTICITY_MAX = 0.95
TRAIN_GRAVITY = (0, -981)

# OOD type definitions
OOD_TYPES = [
    "unseen_count_low",    # 1-9 objects
    "unseen_count_high",   # 51-100 objects
    "extreme_mass",        # mass outside 0.5-10.0
    "extreme_friction",    # friction outside 0.1-1.0
    "extreme_elasticity",  # elasticity outside 0.1-0.95 (NEVER >= 1.0)
    "zero_gravity",        # gravity=(0, 0)
    "low_gravity",         # gravity=(0, -163) ~ Moon
    "high_gravity",        # gravity=(0, -2450) ~ Jupiter
]


def _compute_ood_distance(ood_type: str, ood_level: float, actual_value: float = None) -> float:
    """
    Compute how far out-of-distribution a scene is.

    Returns a normalized distance where:
    - 0.0 = at boundary of training distribution
    - 1.0 = moderate OOD
    - 2.0+ = extreme OOD

    Args:
        ood_type: Type of OOD scenario
        ood_level: OOD level used for generation (1.0=moderate, 2.0=extreme)
        actual_value: Actual value used (for count-based OOD)

    Returns:
        Normalized OOD distance
    """
    if ood_type == "unseen_count_low":
        # Distance based on how far below 10
        if actual_value is not None:
            return (TRAIN_OBJECT_COUNT_MIN - actual_value) / TRAIN_OBJECT_COUNT_MIN
        return ood_level * 0.5  # rough estimate
    elif ood_type == "unseen_count_high":
        # Distance based on how far above 50
        if actual_value is not None:
            return (actual_value - TRAIN_OBJECT_COUNT_MAX) / TRAIN_OBJECT_COUNT_MAX
        return ood_level * 0.5
    elif ood_type == "extreme_mass":
        return ood_level
    elif ood_type == "extreme_friction":
        return ood_level
    elif ood_type == "extreme_elasticity":
        return ood_level * 0.5  # smaller range
    elif ood_type == "zero_gravity":
        return 1.0  # always max distance
    elif ood_type == "low_gravity":
        return 0.83  # (981 - 163) / 981
    elif ood_type == "high_gravity":
        return 1.5  # (2450 - 981) / 981
    else:
        return ood_level


def _get_ood_object_count(ood_type: str, ood_level: float, rng: random.Random) -> int:
    """Get object count based on OOD type."""
    if ood_type == "unseen_count_low":
        # 1-9 objects, level affects minimum
        min_count = max(1, int(9 - 4 * ood_level))  # level 1: 5-9, level 2: 1-9
        return rng.randint(min_count, 9)
    elif ood_type == "unseen_count_high":
        # 51-100 objects, level affects maximum
        max_count = min(100, int(50 + 25 * ood_level))  # level 1: 51-75, level 2: 51-100
        return rng.randint(51, max_count)
    else:
        # Normal training range for non-count OOD types
        return rng.randint(TRAIN_OBJECT_COUNT_MIN, TRAIN_OBJECT_COUNT_MAX)


def _get_ood_mass(ood_type: str, ood_level: float, rng: random.Random) -> float:
    """Get mass value based on OOD type."""
    if ood_type == "extreme_mass":
        # Either very light (0.01-0.1) or very heavy (50-200)
        if rng.random() < 0.5:
            # Light: 0.01 to 0.1 * (2 - level)
            max_light = 0.1 * (2.0 - min(ood_level, 1.9))
            return rng.uniform(0.01, max(0.02, max_light))
        else:
            # Heavy: 50 * level to 200 * level
            return rng.uniform(50.0 * ood_level, 200.0 * ood_level)
    else:
        # Normal training range
        return rng.uniform(TRAIN_MASS_MIN, TRAIN_MASS_MAX)


def _get_ood_friction(ood_type: str, ood_level: float, rng: random.Random) -> float:
    """Get friction value based on OOD type."""
    if ood_type == "extreme_friction":
        # Either very low (0.0-0.05) or very high (2.0-5.0)
        if rng.random() < 0.5:
            # Low friction
            return rng.uniform(0.0, 0.05)
        else:
            # High friction
            return rng.uniform(2.0 * ood_level, 5.0 * ood_level)
    else:
        # Normal training range
        return rng.uniform(TRAIN_FRICTION_MIN, TRAIN_FRICTION_MAX)


def _get_ood_elasticity(ood_type: str, ood_level: float, rng: random.Random) -> float:
    """
    Get elasticity value based on OOD type.

    CRITICAL: Elasticity must NEVER be >= 1.0 to prevent Pymunk instability.
    """
    if ood_type == "extreme_elasticity":
        # High elasticity: 0.96-0.99 (NEVER >= 1.0)
        # Level affects how close to 0.99 we get
        min_elasticity = 0.96
        max_elasticity = min(0.99, 0.96 + 0.015 * ood_level)  # caps at 0.99
        return rng.uniform(min_elasticity, max_elasticity)
    else:
        # Normal training range
        return rng.uniform(TRAIN_ELASTICITY_MIN, TRAIN_ELASTICITY_MAX)


def _get_ood_gravity(ood_type: str) -> tuple:
    """Get gravity vector based on OOD type."""
    if ood_type == "zero_gravity":
        return (0, 0)
    elif ood_type == "low_gravity":
        return (0, -163)  # Moon gravity (~1.62 m/s^2 scaled)
    elif ood_type == "high_gravity":
        return (0, -2450)  # Jupiter gravity (~24.5 m/s^2 scaled)
    else:
        return TRAIN_GRAVITY


def _create_scene_header(
    seed: int,
    num_objects: int,
    gravity: tuple,
    ood_type: str,
    ood_distance: float,
) -> Dict:
    """Create scene header with metadata."""
    return {
        "seed": seed,
        "num_objects": num_objects,
        "gravity": {"x": gravity[0], "y": gravity[1]},
        "scene_bounds": {
            "width": SCENE_WIDTH,
            "height": SCENE_HEIGHT,
        },
        "timestep": 1 / 60.0,
        "ood_type": ood_type,
        "ood_distance": round(ood_distance, 4),
    }


def generate_ood_scene(
    seed: int,
    ood_type: str,
    ood_level: float = 1.0,
    num_frames: int = 60,
) -> Dict:
    """
    Generate an out-of-distribution physics scene.

    Args:
        seed: Random seed for reproducibility
        ood_type: Type of OOD scenario (see OOD_TYPES)
        ood_level: How extreme the OOD is (1.0=moderate, 2.0=extreme)
        num_frames: Number of simulation frames to generate

    Returns:
        Dictionary with:
            - header: Scene metadata including ood_type and ood_distance
            - frames: List of frame states
            - ood_type: The OOD type string
            - ood_distance: Normalized distance from training distribution
    """
    if ood_type not in OOD_TYPES:
        raise ValueError(f"Unknown OOD type: {ood_type}. Valid types: {OOD_TYPES}")

    # Create deterministic RNG for this scene
    rng = random.Random(seed)

    # Determine parameters based on OOD type
    num_objects = _get_ood_object_count(ood_type, ood_level, rng)
    gravity = _get_ood_gravity(ood_type)

    # Compute OOD distance
    ood_distance = _compute_ood_distance(ood_type, ood_level, actual_value=num_objects)

    # Create simulation with appropriate gravity
    sim = PhysicsSimulation(gravity=gravity)

    # Add ground (static segment)
    ground_body, ground_shape = create_static_segment(
        p1=(WALL_MARGIN, GROUND_Y),
        p2=(SCENE_WIDTH - WALL_MARGIN, GROUND_Y),
        friction=0.8,
        elasticity=0.5
    )
    sim.add_static(ground_body, ground_shape)

    # Add walls
    left_wall_body, left_wall_shape = create_static_segment(
        p1=(WALL_MARGIN, GROUND_Y),
        p2=(WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        friction=0.5,
        elasticity=0.3
    )
    sim.add_static(left_wall_body, left_wall_shape)

    right_wall_body, right_wall_shape = create_static_segment(
        p1=(SCENE_WIDTH - WALL_MARGIN, GROUND_Y),
        p2=(SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        friction=0.5,
        elasticity=0.3
    )
    sim.add_static(right_wall_body, right_wall_shape)

    # Create objects with OOD properties
    for i in range(num_objects):
        shape_type = rng.choice(['circle', 'rectangle'])

        # Position (above ground, within bounds)
        x = rng.uniform(50, SCENE_WIDTH - 50)
        y = rng.uniform(200, SCENE_HEIGHT - 50)

        # Material properties (potentially OOD)
        mass = _get_ood_mass(ood_type, ood_level, rng)
        friction = _get_ood_friction(ood_type, ood_level, rng)
        elasticity = _get_ood_elasticity(ood_type, ood_level, rng)

        # SAFETY CHECK: elasticity must NEVER be >= 1.0
        elasticity = min(elasticity, 0.99)

        if shape_type == 'circle':
            radius = rng.uniform(10, 30)
            body, shape = create_circle((x, y), radius, mass, friction, elasticity)
            body.size_info = {'radius': radius}
        else:
            width = rng.uniform(20, 60)
            height = rng.uniform(20, 60)
            body, shape = create_rectangle((x, y), width, height, mass, friction, elasticity)
            body.size_info = {'width': width, 'height': height}

        # Store metadata
        body.object_id = i
        body.shape_type = shape_type
        body.material = {
            'mass': round(mass, 4),
            'friction': round(friction, 4),
            'elasticity': round(elasticity, 4)
        }

        sim.add_body(body, shape)

    # Run simulation and collect frames
    frames = []
    frames.append(sim.get_state())  # Initial state (frame 0)

    for _ in range(num_frames - 1):
        sim.step()
        frames.append(sim.get_state())

    # Create header
    header = _create_scene_header(seed, num_objects, gravity, ood_type, ood_distance)

    return {
        "header": header,
        "frames": frames,
        "ood_type": ood_type,
        "ood_distance": ood_distance,
    }


def generate_ood_test_suite(
    base_seed: int = 200000,
    scenes_per_type: int = 500,
    num_frames: int = 60,
    ood_levels: Optional[list] = None,
) -> Iterator[Dict]:
    """
    Generate a complete OOD test suite.

    Args:
        base_seed: Starting seed (should be outside training range 0-99999)
        scenes_per_type: Number of scenes per OOD type (~4000 total with 8 types)
        num_frames: Number of frames per scene
        ood_levels: List of OOD levels to use (default: [1.0, 2.0] alternating)

    Yields:
        OOD scene dictionaries
    """
    if ood_levels is None:
        ood_levels = [1.0, 2.0]

    seed_counter = base_seed

    for ood_type in OOD_TYPES:
        for i in range(scenes_per_type):
            seed = seed_counter + i
            level = ood_levels[i % len(ood_levels)]

            scene = generate_ood_scene(
                seed=seed,
                ood_type=ood_type,
                ood_level=level,
                num_frames=num_frames,
            )
            yield scene

        # Move to next seed range for next OOD type
        seed_counter += scenes_per_type


def get_ood_type_description(ood_type: str) -> str:
    """Get human-readable description of OOD type."""
    descriptions = {
        "unseen_count_low": "Fewer objects (1-9) than training (10-50)",
        "unseen_count_high": "More objects (51-100) than training (10-50)",
        "extreme_mass": "Mass outside training range (0.01-0.1 or 50-200 vs 0.5-10)",
        "extreme_friction": "Friction outside training range (0-0.05 or 2-5 vs 0.1-1)",
        "extreme_elasticity": "Elasticity outside training range (0.96-0.99 vs 0.1-0.95)",
        "zero_gravity": "Zero gravity (0, 0) vs training (0, -981)",
        "low_gravity": "Moon gravity (0, -163) vs training (0, -981)",
        "high_gravity": "Jupiter gravity (0, -2450) vs training (0, -981)",
    }
    return descriptions.get(ood_type, "Unknown OOD type")
