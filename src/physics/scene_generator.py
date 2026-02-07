"""
Deterministic scene generator for physics simulations.

Key determinism requirements:
1. Set random.seed() ONCE at the very start
2. Create objects in fixed order (use loop index)
3. Never use threaded mode
4. Store object metadata for export
"""

import random
from typing import Optional

from .simulation import PhysicsSimulation
from .objects import create_circle, create_rectangle, create_static_segment


# Scene bounds (in pixels)
SCENE_WIDTH = 800
SCENE_HEIGHT = 600
GROUND_Y = 50
WALL_MARGIN = 10


def generate_scene(seed: int, num_objects: Optional[int] = None) -> PhysicsSimulation:
    """
    Generate a deterministic physics scene.

    Args:
        seed: Random seed for reproducibility
        num_objects: Number of objects (default: random 10-50)

    Returns:
        PhysicsSimulation with objects added
    """
    random.seed(seed)  # CRITICAL: Set seed ONCE at start

    sim = PhysicsSimulation()

    # Determine object count if not specified
    if num_objects is None:
        num_objects = random.randint(10, 50)

    # Add ground (static segment)
    ground_body, ground_shape = create_static_segment(
        p1=(WALL_MARGIN, GROUND_Y),
        p2=(SCENE_WIDTH - WALL_MARGIN, GROUND_Y),
        friction=0.8,
        elasticity=0.5
    )
    sim.add_static(ground_body, ground_shape)

    # Add left wall
    left_wall_body, left_wall_shape = create_static_segment(
        p1=(WALL_MARGIN, GROUND_Y),
        p2=(WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        friction=0.5,
        elasticity=0.3
    )
    sim.add_static(left_wall_body, left_wall_shape)

    # Add right wall
    right_wall_body, right_wall_shape = create_static_segment(
        p1=(SCENE_WIDTH - WALL_MARGIN, GROUND_Y),
        p2=(SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        friction=0.5,
        elasticity=0.3
    )
    sim.add_static(right_wall_body, right_wall_shape)

    # Create objects in DETERMINISTIC ORDER (crucial for reproducibility)
    for i in range(num_objects):
        # Randomly choose shape type
        shape_type = random.choice(['circle', 'rectangle'])

        # Random position (above ground, within bounds)
        x = random.uniform(50, SCENE_WIDTH - 50)
        y = random.uniform(200, SCENE_HEIGHT - 50)

        # Random material properties (within valid ranges from research)
        mass = random.uniform(0.5, 10.0)
        friction = random.uniform(0.1, 1.0)
        elasticity = random.uniform(0.1, 0.95)  # Never >= 1.0

        if shape_type == 'circle':
            radius = random.uniform(10, 30)
            body, shape = create_circle((x, y), radius, mass, friction, elasticity)
            # Store size info
            body.size_info = {'radius': radius}
        else:
            width = random.uniform(20, 60)
            height = random.uniform(20, 60)
            body, shape = create_rectangle((x, y), width, height, mass, friction, elasticity)
            # Store size info
            body.size_info = {'width': width, 'height': height}

        # Store metadata on body for export
        body.object_id = i
        body.shape_type = shape_type
        body.material = {
            'mass': round(mass, 4),
            'friction': round(friction, 4),
            'elasticity': round(elasticity, 4)
        }

        sim.add_body(body, shape)

    return sim
