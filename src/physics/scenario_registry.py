"""Scenario registry for physics scene generation."""

import random
from typing import Callable, Dict, List, Optional, Tuple, Any

from src.physics.simulation import PhysicsSimulation

ScenarioFn = Callable[[random.Random, int], Tuple[PhysicsSimulation, Dict[str, Any]]]

_REGISTRY: Dict[str, Dict[str, Any]] = {}

CATEGORIES = {
    "collision": "Collision & Ballistics",
    "stacking": "Stacking & Structural",
    "ramp": "Ramps & Terrain",
    "constraint": "Pendulums & Constraints",
    "minigame": "Mini-game Physics",
    "complex": "Complex & Chaotic",
}


def register_scenario(
    name: str,
    category: str,
    description: str,
    difficulty_range: Tuple[int, int] = (1, 5),
):
    """Decorator to register a scenario generator function.

    The decorated function must have signature:
        fn(rng: random.Random, difficulty: int) -> Tuple[PhysicsSimulation, Dict[str, Any]]
    """
    def decorator(fn: ScenarioFn) -> ScenarioFn:
        _REGISTRY[name] = {
            "fn": fn,
            "name": name,
            "category": category,
            "description": description,
            "difficulty_range": difficulty_range,
        }
        return fn
    return decorator


def get_scenario(name: str) -> Dict[str, Any]:
    """Look up a registered scenario by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_scenarios(category: Optional[str] = None) -> List[str]:
    """List all registered scenario names, optionally filtered by category."""
    if category is None:
        return sorted(_REGISTRY.keys())
    return sorted(k for k, v in _REGISTRY.items() if v["category"] == category)


def list_categories() -> List[str]:
    """List all scenario categories."""
    return sorted(CATEGORIES.keys())
