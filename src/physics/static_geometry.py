"""Factory functions for creating static geometry (ramps, walls, pegs, platforms)."""

import math
from typing import List, Tuple

import pymunk

from src.physics.objects import create_static_segment, create_static_circle


def create_u_box(
    sim,
    left: float = 10,
    right: float = 790,
    bottom: float = 50,
    top: float = 590,
    ground_friction: float = 0.8,
    ground_elasticity: float = 0.5,
    wall_friction: float = 0.5,
    wall_elasticity: float = 0.3,
):
    """Create standard U-shaped box (ground + two walls) and add to sim."""
    parts = []
    for p1, p2, fr, el in [
        ((left, bottom), (right, bottom), ground_friction, ground_elasticity),
        ((left, bottom), (left, top), wall_friction, wall_elasticity),
        ((right, bottom), (right, top), wall_friction, wall_elasticity),
    ]:
        body, shape = create_static_segment(p1, p2, fr, el)
        sim.add_static(body, shape)
        parts.append((body, shape))
    return parts


def create_ramp(sim, start, end, friction=0.5, elasticity=0.3):
    """Create a ramp (inclined static segment) and add to sim."""
    body, shape = create_static_segment(start, end, friction, elasticity)
    sim.add_static(body, shape)
    return body, shape


def create_platform(sim, center_x, center_y, width, friction=0.8, elasticity=0.5):
    """Create a horizontal static platform and add to sim."""
    half = width / 2
    body, shape = create_static_segment(
        (center_x - half, center_y), (center_x + half, center_y), friction, elasticity
    )
    sim.add_static(body, shape)
    return body, shape


def create_funnel(sim, center_x, top_y, bottom_y, top_width, bottom_gap,
                  friction=0.5, elasticity=0.3):
    """Create V-shaped funnel walls directing objects through a narrow gap."""
    parts = []
    left_top = (center_x - top_width / 2, top_y)
    left_bottom = (center_x - bottom_gap / 2, bottom_y)
    right_top = (center_x + top_width / 2, top_y)
    right_bottom = (center_x + bottom_gap / 2, bottom_y)
    for p1, p2 in [(left_top, left_bottom), (right_top, right_bottom)]:
        body, shape = create_static_segment(p1, p2, friction, elasticity)
        sim.add_static(body, shape)
        parts.append((body, shape))
    return parts


def create_peg_grid(sim, x_start, y_start, rows, cols, spacing,
                    peg_radius=5, friction=0.3, elasticity=0.5):
    """Create offset grid of static circular pegs (Plinko/Galton board)."""
    pegs = []
    for row in range(rows):
        offset = spacing / 2 if row % 2 == 1 else 0
        for col in range(cols):
            x = x_start + col * spacing + offset
            y = y_start - row * spacing
            body, shape = create_static_circle((x, y), peg_radius, friction, elasticity)
            sim.add_static(body, shape)
            pegs.append((body, shape))
    return pegs


def create_bumper(sim, center, radius, elasticity=0.9, friction=0.1):
    """Create a static circular bumper (for pinball-type scenarios)."""
    body, shape = create_static_circle(center, radius, friction, elasticity)
    sim.add_static(body, shape)
    return body, shape


def create_hourglass_chamber(sim, center_x, width, top_y, bottom_y, gap_y, gap_width,
                             friction=0.5, elasticity=0.3):
    """Create hourglass: upper chamber, narrow gap, lower chamber."""
    parts = []
    half_w = width / 2
    half_gap = gap_width / 2

    # Upper chamber walls
    for sign in [-1, 1]:
        x = center_x + sign * half_w
        body, shape = create_static_segment((x, top_y), (x, gap_y + 20), friction, elasticity)
        sim.add_static(body, shape)
        parts.append((body, shape))

    # Funnel walls leading to gap
    for sign in [-1, 1]:
        x_top = center_x + sign * half_w
        x_bot = center_x + sign * half_gap
        body, shape = create_static_segment((x_top, gap_y + 20), (x_bot, gap_y), friction, elasticity)
        sim.add_static(body, shape)
        parts.append((body, shape))

    # Lower chamber walls
    for sign in [-1, 1]:
        x_top = center_x + sign * half_gap
        x_bot = center_x + sign * half_w
        body, shape = create_static_segment((x_top, gap_y), (x_bot, gap_y - 20), friction, elasticity)
        sim.add_static(body, shape)
        parts.append((body, shape))

    for sign in [-1, 1]:
        x = center_x + sign * half_w
        body, shape = create_static_segment((x, gap_y - 20), (x, bottom_y), friction, elasticity)
        sim.add_static(body, shape)
        parts.append((body, shape))

    # Lower floor
    body, shape = create_static_segment(
        (center_x - half_w, bottom_y), (center_x + half_w, bottom_y), friction, elasticity
    )
    sim.add_static(body, shape)
    parts.append((body, shape))
    return parts


def create_hoop(sim, center_x, center_y, gap_width=40, arm_length=30,
                friction=0.3, elasticity=0.5):
    """Create a basketball-style hoop (two short segments with a gap)."""
    parts = []
    half = gap_width / 2
    # Left rim
    body, shape = create_static_segment(
        (center_x - half - arm_length, center_y), (center_x - half, center_y),
        friction, elasticity
    )
    sim.add_static(body, shape)
    parts.append((body, shape))
    # Right rim
    body, shape = create_static_segment(
        (center_x + half, center_y), (center_x + half + arm_length, center_y),
        friction, elasticity
    )
    sim.add_static(body, shape)
    parts.append((body, shape))
    return parts
