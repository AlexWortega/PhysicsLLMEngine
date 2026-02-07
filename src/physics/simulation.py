"""
Deterministic 2D rigid body physics simulation using Pymunk.

Critical for determinism:
- Fixed timestep (DT = 1/60.0)
- NO threaded mode (space.threaded = False by default)
- Objects tracked in creation order
"""

import pymunk


class PhysicsSimulation:
    """
    A deterministic physics simulation wrapper around Pymunk.

    The simulation uses a fixed timestep and maintains objects in
    creation order to ensure reproducibility.
    """

    DT = 1/60.0  # Fixed timestep - CRITICAL for determinism

    def __init__(self, gravity: tuple = (0, -981)):
        """
        Initialize the physics simulation.

        Args:
            gravity: Gravity vector as (x, y). Default is (0, -981) for
                     downward gravity at ~10 m/s^2 with pixels-to-meters
                     scaling of ~100.
        """
        self.space = pymunk.Space()
        self.space.gravity = gravity
        # DO NOT use space.threaded = True - breaks determinism
        self.frame = 0
        self.bodies = []  # Track bodies in creation order
        self.static_bodies = []  # Track static bodies separately
        self.static_shapes = []  # Track static shapes for serialization
        self.constraints = []  # Track constraints (joints/springs)

    def add_body(self, body: pymunk.Body, shape: pymunk.Shape) -> None:
        """
        Add a dynamic body with its shape to the simulation.

        Args:
            body: The Pymunk body to add
            shape: The shape attached to the body
        """
        self.space.add(body, shape)
        self.bodies.append(body)

    def add_static(self, body: pymunk.Body, shape: pymunk.Shape) -> None:
        """
        Add a static body (like ground or walls) to the simulation.

        Args:
            body: The static Pymunk body
            shape: The shape attached to the body
        """
        self.space.add(body, shape)
        self.static_bodies.append(body)
        self.static_shapes.append(shape)

    def add_constraint(self, constraint) -> None:
        """Add a constraint (joint/spring) to the simulation."""
        self.space.add(constraint)
        self.constraints.append(constraint)

    def step(self) -> None:
        """
        Advance the simulation by one fixed timestep.
        """
        self.space.step(self.DT)
        self.frame += 1

    def get_state(self) -> dict:
        """
        Get the current state of all dynamic bodies.

        Returns:
            dict with:
                - frame: Current frame number
                - objects: List of object states with position, velocity,
                          angle, angular_velocity, and metadata
        """
        objects = []
        for body in self.bodies:
            obj_state = {
                "id": getattr(body, 'object_id', len(objects)),
                "type": getattr(body, 'shape_type', 'unknown'),
                "position": {"x": round(body.position.x, 4), "y": round(body.position.y, 4)},
                "velocity": {"x": round(body.velocity.x, 4), "y": round(body.velocity.y, 4)},
                "angle": round(body.angle, 6),
                "angular_velocity": round(body.angular_velocity, 6),
                "material": getattr(body, 'material', {})
            }
            objects.append(obj_state)

        return {
            "frame": self.frame,
            "objects": objects
        }
