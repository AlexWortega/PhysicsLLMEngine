"""
Physics simulation module.

Provides deterministic 2D rigid body physics using Pymunk.
"""

from .simulation import PhysicsSimulation
from .objects import create_circle, create_rectangle, create_static_segment, create_static_circle
from .scene_generator import generate_scene
from .scenario_generator import generate_scenario, SCENARIO_TYPES
from .scenario_registry import list_scenarios, list_categories, get_scenario

__all__ = [
    'PhysicsSimulation',
    'create_circle',
    'create_rectangle',
    'create_static_segment',
    'create_static_circle',
    'generate_scene',
    'generate_scenario',
    'SCENARIO_TYPES',
    'list_scenarios',
    'list_categories',
    'get_scenario',
]
