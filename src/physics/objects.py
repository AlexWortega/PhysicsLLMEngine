"""
Factory functions for creating physics objects with validated properties.

Material property constraints (from research):
- elasticity: 0.0-0.99 (NEVER >= 1.0 to prevent energy gain)
- friction: 0.0-1.5 typically
- mass: positive float
"""

import pymunk
from typing import Tuple


def validate_material_properties(mass: float, friction: float, elasticity: float) -> None:
    """
    Validate material properties to prevent physics instability.

    Args:
        mass: Mass of the object (must be positive)
        friction: Friction coefficient (typically 0.0-1.5)
        elasticity: Coefficient of restitution (must be < 1.0)

    Raises:
        ValueError: If any property is out of valid range
    """
    if mass <= 0:
        raise ValueError(f"Mass must be positive, got {mass}")
    if friction < 0:
        raise ValueError(f"Friction must be non-negative, got {friction}")
    if elasticity < 0 or elasticity >= 1.0:
        raise ValueError(f"Elasticity must be in [0, 1.0), got {elasticity}")


def create_circle(
    pos: Tuple[float, float],
    radius: float,
    mass: float,
    friction: float,
    elasticity: float
) -> Tuple[pymunk.Body, pymunk.Circle]:
    """
    Create a circular physics body.

    Args:
        pos: Initial position as (x, y)
        radius: Circle radius
        mass: Mass of the body
        friction: Friction coefficient
        elasticity: Coefficient of restitution (bounciness)

    Returns:
        Tuple of (body, shape)
    """
    validate_material_properties(mass, friction, elasticity)

    # Calculate moment of inertia for a circle
    moment = pymunk.moment_for_circle(mass, 0, radius)

    body = pymunk.Body(mass, moment)
    body.position = pos

    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity

    return body, shape


def create_rectangle(
    pos: Tuple[float, float],
    width: float,
    height: float,
    mass: float,
    friction: float,
    elasticity: float
) -> Tuple[pymunk.Body, pymunk.Poly]:
    """
    Create a rectangular physics body.

    Args:
        pos: Initial position as (x, y) - center of rectangle
        width: Width of the rectangle
        height: Height of the rectangle
        mass: Mass of the body
        friction: Friction coefficient
        elasticity: Coefficient of restitution (bounciness)

    Returns:
        Tuple of (body, shape)
    """
    validate_material_properties(mass, friction, elasticity)

    # Calculate moment of inertia for a box
    moment = pymunk.moment_for_box(mass, (width, height))

    body = pymunk.Body(mass, moment)
    body.position = pos

    # Create a box centered at the body position
    shape = pymunk.Poly.create_box(body, (width, height))
    shape.friction = friction
    shape.elasticity = elasticity

    return body, shape


def create_static_circle(
    pos: Tuple[float, float],
    radius: float,
    friction: float = 0.3,
    elasticity: float = 0.5,
) -> Tuple[pymunk.Body, pymunk.Circle]:
    """Create a static circular body (for pegs, bumpers)."""
    if elasticity >= 1.0:
        raise ValueError(f"Elasticity must be < 1.0, got {elasticity}")
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    return body, shape


def create_static_segment(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    friction: float = 0.8,
    elasticity: float = 0.5
) -> Tuple[pymunk.Body, pymunk.Segment]:
    """
    Create a static line segment (for ground/walls).

    Args:
        p1: Start point as (x, y)
        p2: End point as (x, y)
        friction: Friction coefficient (default 0.8)
        elasticity: Coefficient of restitution (default 0.5)

    Returns:
        Tuple of (body, shape)
    """
    if elasticity >= 1.0:
        raise ValueError(f"Elasticity must be < 1.0, got {elasticity}")

    body = pymunk.Body(body_type=pymunk.Body.STATIC)

    shape = pymunk.Segment(body, p1, p2, radius=2)  # radius is thickness
    shape.friction = friction
    shape.elasticity = elasticity

    return body, shape
