"""Factory functions for creating physics constraints/joints."""

import pymunk
from typing import Tuple, List

from src.physics.objects import create_circle


def create_pin_joint(body_a, body_b, anchor_a=(0, 0), anchor_b=(0, 0)):
    """Create a pin joint (fixed distance) between two bodies."""
    return pymunk.PinJoint(body_a, body_b, anchor_a, anchor_b)


def create_pivot_joint(body_a, body_b, pivot):
    """Create a pivot joint (shared rotation point) between two bodies."""
    return pymunk.PivotJoint(body_a, body_b, pivot)


def create_damped_spring(body_a, body_b, anchor_a, anchor_b,
                         rest_length, stiffness, damping):
    """Create a damped spring between two bodies."""
    return pymunk.DampedSpring(body_a, body_b, anchor_a, anchor_b,
                               rest_length, stiffness, damping)


def create_slide_joint(body_a, body_b, anchor_a, anchor_b, min_dist, max_dist):
    """Create a slide joint (distance range) between two bodies."""
    return pymunk.SlideJoint(body_a, body_b, anchor_a, anchor_b, min_dist, max_dist)


def create_chain(sim, start_pos, end_pos, num_links, link_mass=0.5,
                 link_radius=5, friction=0.5, elasticity=0.3, anchor_start=True):
    """Create a chain of linked circular bodies connected by PinJoints.

    Returns list of (body, shape) tuples. Joints are added to sim directly.
    If anchor_start, first link is pinned to a static body at start_pos.
    """
    dx = (end_pos[0] - start_pos[0]) / max(num_links - 1, 1)
    dy = (end_pos[1] - start_pos[1]) / max(num_links - 1, 1)
    link_dist = (dx ** 2 + dy ** 2) ** 0.5

    links = []
    for i in range(num_links):
        x = start_pos[0] + dx * i
        y = start_pos[1] + dy * i
        body, shape = create_circle((x, y), link_radius, link_mass, friction, elasticity)
        body.object_id = len(sim.bodies)
        body.shape_type = 'circle'
        body.material = {
            'mass': round(link_mass, 4),
            'friction': round(friction, 4),
            'elasticity': round(elasticity, 4),
        }
        body.size_info = {'radius': link_radius}
        sim.add_body(body, shape)
        links.append((body, shape))

    # Pin first link to static anchor
    if anchor_start and links:
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        static_body.position = start_pos
        sim.space.add(static_body)
        joint = pymunk.PinJoint(static_body, links[0][0], (0, 0), (0, 0))
        sim.add_constraint(joint)

    # Connect consecutive links
    for i in range(len(links) - 1):
        joint = pymunk.PinJoint(links[i][0], links[i + 1][0], (0, 0), (0, 0))
        sim.add_constraint(joint)

    return links
