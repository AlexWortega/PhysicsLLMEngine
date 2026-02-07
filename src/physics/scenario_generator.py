"""
Diverse scenario generator for physics simulations.

35 scenario types across 6 categories producing qualitatively different
physics behaviors for training physics-predicting LLMs.

Usage:
    from src.physics.scenario_generator import generate_scenario, SCENARIO_TYPES

    sim, metadata = generate_scenario(seed=42, scenario_type="projectile", difficulty=3)
    sim, metadata = generate_scenario(seed=42)  # random scenario
"""

import math
import random
from typing import Dict, Any, Optional, Tuple, List

import pymunk

from src.physics.simulation import PhysicsSimulation
from src.physics.objects import (
    create_circle, create_rectangle, create_static_segment, create_static_circle,
)
from src.physics.static_geometry import (
    create_u_box, create_ramp, create_funnel, create_peg_grid,
    create_platform, create_bumper, create_hourglass_chamber, create_hoop,
)
from src.physics.constraints import (
    create_pin_joint, create_pivot_joint, create_chain,
)
from src.physics.scenario_registry import (
    register_scenario, get_scenario, list_scenarios, _REGISTRY,
)

SCENE_WIDTH = 800
SCENE_HEIGHT = 600
GROUND_Y = 50
WALL_MARGIN = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_body(sim, body, shape, shape_type, mass, friction, elasticity, size_info):
    """Add a dynamic body to sim with standard metadata."""
    body.object_id = len(sim.bodies)
    body.shape_type = shape_type
    body.material = {
        'mass': round(mass, 4),
        'friction': round(friction, 4),
        'elasticity': round(elasticity, 4),
    }
    body.size_info = size_info
    sim.add_body(body, shape)


def _random_material(rng, mass_range=(0.5, 10.0), friction_range=(0.1, 1.0),
                     elasticity_range=(0.1, 0.95)):
    """Generate random material properties."""
    return (
        rng.uniform(*mass_range),
        rng.uniform(*friction_range),
        min(rng.uniform(*elasticity_range), 0.99),
    )


def _serialize_static(sim):
    """Serialize static geometry for metadata."""
    result = []
    for body, shape in zip(sim.static_bodies, sim.static_shapes):
        if isinstance(shape, pymunk.Segment):
            a = shape.a
            b = shape.b
            pos = body.position
            result.append({
                "type": "segment",
                "p1": {"x": round(a.x + pos.x, 4), "y": round(a.y + pos.y, 4)},
                "p2": {"x": round(b.x + pos.x, 4), "y": round(b.y + pos.y, 4)},
                "friction": shape.friction, "elasticity": shape.elasticity,
            })
        elif isinstance(shape, pymunk.Circle):
            result.append({
                "type": "circle",
                "center": {"x": round(body.position.x, 4), "y": round(body.position.y, 4)},
                "radius": shape.radius,
                "friction": shape.friction, "elasticity": shape.elasticity,
            })
    return result


def _serialize_constraints(sim):
    """Serialize constraints for metadata."""
    result = []
    body_id_map = {b: getattr(b, 'object_id', None) for b in sim.bodies}
    for c in sim.constraints:
        a_id = body_id_map.get(c.a, "static")
        b_id = body_id_map.get(c.b, "static")
        entry = {"type": type(c).__name__, "body_a": a_id, "body_b": b_id}
        if isinstance(c, pymunk.PinJoint):
            entry["anchor_a"] = [round(c.anchor_a.x, 4), round(c.anchor_a.y, 4)]
            entry["anchor_b"] = [round(c.anchor_b.x, 4), round(c.anchor_b.y, 4)]
        elif isinstance(c, pymunk.PivotJoint):
            entry["anchor_a"] = [round(c.anchor_a.x, 4), round(c.anchor_a.y, 4)]
            entry["anchor_b"] = [round(c.anchor_b.x, 4), round(c.anchor_b.y, 4)]
        result.append(entry)
    return result


def _build_metadata(sim, scenario_type, category, difficulty, description):
    """Build standard metadata dict."""
    return {
        "scenario_type": scenario_type,
        "scenario_category": category,
        "difficulty": difficulty,
        "description": description,
        "object_count": len(sim.bodies),
        "static_geometry": _serialize_static(sim),
        "constraints": _serialize_constraints(sim),
    }


# ===================================================================
# CATEGORY 1: Collision & Ballistics
# ===================================================================

@register_scenario("projectile", category="collision",
    description="Objects launched with initial velocity at various angles")
def _scenario_projectile(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    num_proj = difficulty
    num_targets = difficulty * 2

    # Target blocks on right side
    for i in range(num_targets):
        x = 600 + rng.uniform(-20, 20)
        y = GROUND_Y + 15 + i * 30
        mass, friction, elasticity = _random_material(rng, mass_range=(1.0, 5.0))
        w, h = rng.uniform(30, 50), rng.uniform(20, 30)
        body, shape = create_rectangle((x, y), w, h, mass, friction, elasticity)
        _add_body(sim, body, shape, 'rectangle', mass, friction, elasticity, {'width': w, 'height': h})

    # Projectiles on left with initial velocity
    for i in range(num_proj):
        angle = rng.uniform(25, 65) * (math.pi / 180)
        speed = rng.uniform(300, 600)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        r = rng.uniform(10, 20)
        mass, friction, elasticity = _random_material(rng, mass_range=(1.0, 8.0))
        body, shape = create_circle((80, GROUND_Y + 40 + i * 35), r, mass, friction, elasticity)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    angle_deg = int(45)
    desc = f"Projectile: {num_proj} balls launched at ~{angle_deg} degrees toward {num_targets} target blocks."
    return sim, _build_metadata(sim, "projectile", "collision", difficulty, desc)


@register_scenario("billiards", category="collision",
    description="Cue ball strikes triangle-racked stationary balls")
def _scenario_billiards(rng, difficulty):
    sim = PhysicsSimulation(gravity=(0, 0))  # Flat table, no gravity
    # Walls
    create_u_box(sim, ground_friction=0.05, ground_elasticity=0.9,
                 wall_friction=0.05, wall_elasticity=0.9)
    # Top wall
    body, shape = create_static_segment((WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        0.05, 0.9)
    sim.add_static(body, shape)

    r = 12
    friction = 0.05
    elasticity = 0.92
    mass = 1.0

    # Triangle rack of target balls at right-center
    rows = difficulty + 1  # 2-6 rows
    cx, cy = 500, 300
    spacing = r * 2.2
    for row in range(rows):
        for col in range(row + 1):
            x = cx + row * spacing * math.cos(math.pi / 6)
            y = cy + (col - row / 2) * spacing
            body, shape = create_circle((x, y), r, mass, friction, elasticity)
            _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    # Cue ball on left
    cue_speed = rng.uniform(400, 800)
    angle_offset = rng.uniform(-5, 5) * (math.pi / 180)
    body, shape = create_circle((150, 300 + rng.uniform(-10, 10)), r, mass, friction, elasticity)
    body.velocity = (cue_speed * math.cos(angle_offset), cue_speed * math.sin(angle_offset))
    _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    total_targets = sum(range(1, rows + 1))
    desc = f"Billiards: cue ball strikes a triangle of {total_targets} balls on a smooth table."
    return sim, _build_metadata(sim, "billiards", "collision", difficulty, desc)


@register_scenario("bowling", category="collision",
    description="Heavy ball rolls toward arranged pins")
def _scenario_bowling(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    pin_rows = min(difficulty + 1, 4)  # 2-4 rows
    pin_w, pin_h = 8, 30
    pin_mass = 0.5
    pin_friction = 0.6
    pin_elasticity = 0.3

    cx, cy = 550, GROUND_Y + pin_h / 2
    spacing_x = 25
    spacing_y = 20
    for row in range(pin_rows):
        for col in range(row + 1):
            x = cx + row * spacing_x
            y_offset = (col - row / 2) * spacing_y
            body, shape = create_rectangle((x, cy + y_offset), pin_w, pin_h,
                                           pin_mass, pin_friction, pin_elasticity)
            _add_body(sim, body, shape, 'rectangle', pin_mass, pin_friction, pin_elasticity,
                     {'width': pin_w, 'height': pin_h})

    # Bowling ball
    ball_r = 20
    ball_mass = 8.0
    ball_speed = 200 + difficulty * 60
    body, shape = create_circle((100, GROUND_Y + ball_r + 2), ball_r,
                                 ball_mass, 0.8, 0.2)
    body.velocity = (ball_speed, 0)
    _add_body(sim, body, shape, 'circle', ball_mass, 0.8, 0.2, {'radius': ball_r})

    n_pins = sum(range(1, pin_rows + 1))
    desc = f"Bowling: heavy ball rolling at {n_pins} pins arranged in {pin_rows} rows."
    return sim, _build_metadata(sim, "bowling", "collision", difficulty, desc)


@register_scenario("head_on", category="collision",
    description="Two groups of objects approaching each other")
def _scenario_head_on(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    group_size = difficulty * 2
    speed = 150 + difficulty * 80

    # Left group moving right
    for i in range(group_size):
        r = rng.uniform(10, 25)
        mass, friction, elasticity = _random_material(rng)
        x = rng.uniform(80, 200)
        y = GROUND_Y + 50 + rng.uniform(0, 300)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        body.velocity = (speed + rng.uniform(-30, 30), rng.uniform(-50, 50))
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    # Right group moving left
    for i in range(group_size):
        r = rng.uniform(10, 25)
        mass, friction, elasticity = _random_material(rng)
        x = rng.uniform(600, 720)
        y = GROUND_Y + 50 + rng.uniform(0, 300)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        body.velocity = (-speed + rng.uniform(-30, 30), rng.uniform(-50, 50))
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Head-on: two groups of {group_size} objects approaching each other."
    return sim, _build_metadata(sim, "head_on", "collision", difficulty, desc)


@register_scenario("explosion", category="collision",
    description="Objects flying outward from center with radial velocities")
def _scenario_explosion(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    n = 5 + difficulty * 5  # 10-30
    cx, cy = 400, 300
    speed_base = 200 + difficulty * 120

    for i in range(n):
        angle = 2 * math.pi * i / n + rng.uniform(-0.1, 0.1)
        speed = speed_base + rng.uniform(-60, 60)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        r = rng.uniform(8, 20)
        mass, friction, elasticity = _random_material(rng)
        offset = rng.uniform(0, 15)
        x = cx + offset * math.cos(angle)
        y = cy + offset * math.sin(angle)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Explosion: {n} objects flying outward from center."
    return sim, _build_metadata(sim, "explosion", "collision", difficulty, desc)


# ===================================================================
# CATEGORY 2: Stacking & Structural
# ===================================================================

@register_scenario("tower", category="stacking",
    description="Blocks stacked vertically, may topple")
def _scenario_tower(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    n_blocks = 3 + difficulty * 3  # 6-18
    block_w = max(60 - difficulty * 5, 25)  # Narrower at higher difficulty
    block_h = 20
    cx = 400

    for i in range(n_blocks):
        x = cx + rng.uniform(-2, 2)  # Slight imperfection
        y = GROUND_Y + block_h / 2 + i * (block_h + 1)
        mass = rng.uniform(1.0, 3.0)
        body, shape = create_rectangle((x, y), block_w, block_h, mass, 0.7, 0.2)
        _add_body(sim, body, shape, 'rectangle', mass, 0.7, 0.2,
                 {'width': block_w, 'height': block_h})

    # Optional toppler ball at high difficulty
    if difficulty >= 3:
        r = 15
        mass = 5.0
        body, shape = create_circle((cx + rng.uniform(-100, -50), SCENE_HEIGHT - 100),
                                     r, mass, 0.5, 0.3)
        body.velocity = (rng.uniform(50, 150), rng.uniform(-200, -100))
        _add_body(sim, body, shape, 'circle', mass, 0.5, 0.3, {'radius': r})

    desc = f"Tower: {n_blocks} blocks stacked vertically{', with toppler ball' if difficulty >= 3 else ''}."
    return sim, _build_metadata(sim, "tower", "stacking", difficulty, desc)


@register_scenario("pyramid", category="stacking",
    description="Triangular stack of objects")
def _scenario_pyramid(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    base_width = 2 + difficulty  # 3-7
    block_size = 25
    cx = 400
    base_y = GROUND_Y + block_size / 2

    for row in range(base_width):
        blocks_in_row = base_width - row
        row_width = blocks_in_row * (block_size + 2)
        start_x = cx - row_width / 2 + block_size / 2
        for col in range(blocks_in_row):
            x = start_x + col * (block_size + 2)
            y = base_y + row * (block_size + 1)
            mass = rng.uniform(1.0, 3.0)
            body, shape = create_rectangle((x, y), block_size, block_size, mass, 0.7, 0.2)
            _add_body(sim, body, shape, 'rectangle', mass, 0.7, 0.2,
                     {'width': block_size, 'height': block_size})

    total = sum(range(1, base_width + 1))
    desc = f"Pyramid: {total} blocks in triangular stack with base {base_width}."
    return sim, _build_metadata(sim, "pyramid", "stacking", difficulty, desc)


@register_scenario("dominos", category="stacking",
    description="Row of tall thin rectangles, first one pushed")
def _scenario_dominos(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    n = 5 + difficulty * 4  # 9-25
    domino_w = 6
    domino_h = 30
    spacing = domino_h * 0.7
    start_x = 100

    for i in range(n):
        x = start_x + i * spacing
        y = GROUND_Y + domino_h / 2 + 1
        mass = 1.0
        body, shape = create_rectangle((x, y), domino_w, domino_h, mass, 0.6, 0.2)
        _add_body(sim, body, shape, 'rectangle', mass, 0.6, 0.2,
                 {'width': domino_w, 'height': domino_h})

    # Push ball hits first domino
    r = 10
    push_speed = 150
    body, shape = create_circle((start_x - 50, GROUND_Y + domino_h * 0.7),
                                 r, 3.0, 0.5, 0.3)
    body.velocity = (push_speed, 0)
    _add_body(sim, body, shape, 'circle', 3.0, 0.5, 0.3, {'radius': r})

    desc = f"Dominos: {n} thin blocks in a row, pushed by a ball."
    return sim, _build_metadata(sim, "dominos", "stacking", difficulty, desc)


@register_scenario("jenga", category="stacking",
    description="Alternating layers of blocks, some missing")
def _scenario_jenga(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    layers = 3 + difficulty * 2  # 5-13
    blocks_per_layer = 3
    block_long = 60
    block_short = 18
    block_h = 12
    cx = 400
    missing = difficulty - 1  # 0-4 missing blocks

    removed = set()
    if missing > 0:
        candidates = [(l, b) for l in range(layers) for b in range(blocks_per_layer)
                       if l > 0 and l < layers - 1]
        rng.shuffle(candidates)
        for i in range(min(missing, len(candidates))):
            removed.add(candidates[i])

    for layer in range(layers):
        for b in range(blocks_per_layer):
            if (layer, b) in removed:
                continue
            if layer % 2 == 0:
                w, h = block_long / blocks_per_layer - 1, block_h
                x = cx - block_long / 2 + (b + 0.5) * (block_long / blocks_per_layer)
            else:
                w, h = block_h, block_long / blocks_per_layer - 1
                x = cx + (b - 1) * block_short
            y = GROUND_Y + block_h / 2 + layer * (block_h + 0.5)
            mass = 1.0
            body, shape = create_rectangle((x, y), w, h, mass, 0.6, 0.15)
            _add_body(sim, body, shape, 'rectangle', mass, 0.6, 0.15, {'width': w, 'height': h})

    n = len(sim.bodies)
    desc = f"Jenga: {n} blocks in {layers} alternating layers{f', {len(removed)} missing' if removed else ''}."
    return sim, _build_metadata(sim, "jenga", "stacking", difficulty, desc)


@register_scenario("bridge", category="stacking",
    description="Blocks forming arch between two pillars")
def _scenario_bridge(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    pillar_h = 80 + difficulty * 20
    pillar_w = 30
    gap = 100 + difficulty * 30

    # Left pillar (stacked blocks)
    left_x = 300 - gap / 2
    for i in range(pillar_h // 25):
        y = GROUND_Y + 12.5 + i * 25
        mass = 5.0
        body, shape = create_rectangle((left_x, y), pillar_w, 25, mass, 0.8, 0.15)
        _add_body(sim, body, shape, 'rectangle', mass, 0.8, 0.15, {'width': pillar_w, 'height': 25})

    # Right pillar
    right_x = 300 + gap / 2
    for i in range(pillar_h // 25):
        y = GROUND_Y + 12.5 + i * 25
        mass = 5.0
        body, shape = create_rectangle((right_x, y), pillar_w, 25, mass, 0.8, 0.15)
        _add_body(sim, body, shape, 'rectangle', mass, 0.8, 0.15, {'width': pillar_w, 'height': 25})

    # Bridge deck
    deck_y = GROUND_Y + pillar_h + 5
    n_deck = max(3, gap // 35)
    deck_w = gap / n_deck + 5
    for i in range(n_deck):
        x = left_x + (i + 0.5) * (gap / n_deck)
        mass = 2.0
        body, shape = create_rectangle((x, deck_y), deck_w, 12, mass, 0.7, 0.2)
        _add_body(sim, body, shape, 'rectangle', mass, 0.7, 0.2, {'width': deck_w, 'height': 12})

    # Drop weight on bridge at higher difficulty
    if difficulty >= 3:
        r = 15
        mass = 8.0
        body, shape = create_circle((300, deck_y + 150), r, mass, 0.5, 0.3)
        _add_body(sim, body, shape, 'circle', mass, 0.5, 0.3, {'radius': r})

    desc = f"Bridge: blocks forming a bridge across a {gap:.0f}-pixel gap on two pillars."
    return sim, _build_metadata(sim, "bridge", "stacking", difficulty, desc)


# ===================================================================
# CATEGORY 3: Ramps & Terrain
# ===================================================================

@register_scenario("ramp_roll", category="ramp",
    description="Objects rolling down an inclined plane")
def _scenario_ramp_roll(rng, difficulty):
    sim = PhysicsSimulation()
    # Ground only (no side walls to let objects roll off naturally)
    body, shape = create_static_segment((WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body, shape)

    angle = 15 + difficulty * 6  # 21-45 degrees
    ramp_len = 400
    start_x, start_y = 100, 400
    end_x = start_x + ramp_len * math.cos(angle * math.pi / 180)
    end_y = start_y - ramp_len * math.sin(angle * math.pi / 180)
    create_ramp(sim, (start_x, start_y), (end_x, end_y), friction=0.4, elasticity=0.3)

    # Lip at bottom to redirect
    create_ramp(sim, (end_x, end_y), (end_x + 50, end_y + 10), friction=0.4, elasticity=0.3)

    n = 1 + difficulty  # 2-6
    for i in range(n):
        r = rng.uniform(10, 22)
        mass, friction, elasticity = _random_material(rng, mass_range=(0.5, 5.0))
        offset = i * (r * 2 + 5)
        t = offset / ramp_len
        x = start_x + t * (end_x - start_x) + rng.uniform(-3, 3)
        y = start_y + t * (end_y - start_y) + r + 5
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Ramp roll: {n} objects on a {angle}-degree incline."
    return sim, _build_metadata(sim, "ramp_roll", "ramp", difficulty, desc)


@register_scenario("ski_jump", category="ramp",
    description="Ramp launches objects into the air")
def _scenario_ski_jump(rng, difficulty):
    sim = PhysicsSimulation()
    body, shape = create_static_segment((WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body, shape)

    # Downward slope
    create_ramp(sim, (50, 500), (300, 200), friction=0.3, elasticity=0.3)
    # Upward launch curve
    create_ramp(sim, (300, 200), (400, 250), friction=0.3, elasticity=0.3)

    n = difficulty  # 1-5
    for i in range(n):
        r = rng.uniform(10, 18)
        mass = rng.uniform(1.0, 4.0)
        body, shape = create_circle((80 + i * 30, 510 + r), r, mass, 0.4, 0.3)
        _add_body(sim, body, shape, 'circle', mass, 0.4, 0.3, {'radius': r})

    desc = f"Ski jump: {n} balls rolling down a slope and launching off a ramp."
    return sim, _build_metadata(sim, "ski_jump", "ramp", difficulty, desc)


@register_scenario("funnel", category="ramp",
    description="V-shaped walls directing objects through narrow gap")
def _scenario_funnel(rng, difficulty):
    sim = PhysicsSimulation()
    body, shape = create_static_segment((WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body, shape)
    # Side walls
    for x in [WALL_MARGIN, SCENE_WIDTH - WALL_MARGIN]:
        body, shape = create_static_segment((x, GROUND_Y), (x, SCENE_HEIGHT - WALL_MARGIN), 0.5, 0.3)
        sim.add_static(body, shape)

    gap_width = max(60 - difficulty * 8, 20)  # 52 down to 20
    create_funnel(sim, center_x=400, top_y=450, bottom_y=250,
                  top_width=500, bottom_gap=gap_width)

    n = 5 + difficulty * 5  # 10-30
    for i in range(n):
        r = rng.uniform(6, min(gap_width / 2 - 2, 15))
        mass, friction, elasticity = _random_material(rng, mass_range=(0.5, 3.0))
        x = 400 + rng.uniform(-200, 200)
        y = rng.uniform(460, 550)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Funnel: {n} balls falling through a V-shaped funnel with {gap_width:.0f}-pixel gap."
    return sim, _build_metadata(sim, "funnel", "ramp", difficulty, desc)


@register_scenario("plinko", category="ramp",
    description="Peg grid redirects falling objects randomly")
def _scenario_plinko(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    rows = 3 + difficulty  # 4-8
    cols = 3 + difficulty  # 4-8
    spacing = min(60, 500 / cols)
    x_start = 400 - (cols - 1) * spacing / 2
    y_start = 500

    create_peg_grid(sim, x_start, y_start, rows, cols, spacing,
                    peg_radius=5, friction=0.3, elasticity=0.6)

    n_balls = difficulty  # 1-5
    for i in range(n_balls):
        r = 8
        mass = 1.0
        x = 400 + rng.uniform(-spacing / 2, spacing / 2)
        y = y_start + 40 + i * 25
        body, shape = create_circle((x, y), r, mass, 0.3, 0.5)
        _add_body(sim, body, shape, 'circle', mass, 0.3, 0.5, {'radius': r})

    desc = f"Plinko: {n_balls} balls dropping through {rows}x{cols} peg grid."
    return sim, _build_metadata(sim, "plinko", "ramp", difficulty, desc)


@register_scenario("marble_run", category="ramp",
    description="Zigzag ramps with drops between them")
def _scenario_marble_run(rng, difficulty):
    sim = PhysicsSimulation()
    body, shape = create_static_segment((WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body, shape)

    n_ramps = 2 + difficulty  # 3-7
    ramp_len = 250
    y_start = 550
    y_step = (y_start - 100) / n_ramps

    for i in range(n_ramps):
        y = y_start - i * y_step
        if i % 2 == 0:
            create_ramp(sim, (100, y), (100 + ramp_len, y - y_step * 0.6), friction=0.3, elasticity=0.3)
        else:
            create_ramp(sim, (700, y), (700 - ramp_len, y - y_step * 0.6), friction=0.3, elasticity=0.3)

    n_balls = difficulty  # 1-5
    for i in range(n_balls):
        r = rng.uniform(8, 14)
        mass = rng.uniform(0.5, 3.0)
        x = 110 + i * 20
        body, shape = create_circle((x, y_start + r + 5), r, mass, 0.3, 0.4)
        _add_body(sim, body, shape, 'circle', mass, 0.3, 0.4, {'radius': r})

    desc = f"Marble run: {n_balls} balls on {n_ramps} zigzag ramps."
    return sim, _build_metadata(sim, "marble_run", "ramp", difficulty, desc)


# ===================================================================
# CATEGORY 4: Pendulums & Constraints
# ===================================================================

@register_scenario("newtons_cradle", category="constraint",
    description="Balls hung by pins, end ball pulled and released")
def _scenario_newtons_cradle(rng, difficulty):
    sim = PhysicsSimulation()
    body_s, shape_s = create_static_segment((WALL_MARGIN, GROUND_Y),
                                             (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body_s, shape_s)

    n_balls = 3 + difficulty  # 4-8
    r = 15
    mass = 2.0
    string_len = 200
    anchor_y = 450
    cx = 400
    spacing = r * 2 + 1

    for i in range(n_balls):
        x = cx + (i - (n_balls - 1) / 2) * spacing
        y = anchor_y - string_len
        body, shape = create_circle((x, y), r, mass, 0.1, 0.95)
        _add_body(sim, body, shape, 'circle', mass, 0.1, 0.95, {'radius': r})

        # Pin to anchor
        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = (x, anchor_y)
        sim.space.add(anchor)
        joint = pymunk.PinJoint(anchor, body, (0, 0), (0, 0))
        sim.add_constraint(joint)

    # Pull rightmost ball to the side
    if sim.bodies:
        pull_angle = rng.uniform(30, 60) * (math.pi / 180)
        last_body = sim.bodies[-1]
        anchor_x = last_body.position.x
        last_body.position = (
            anchor_x + string_len * math.sin(pull_angle),
            anchor_y - string_len * math.cos(pull_angle),
        )

    desc = f"Newton's cradle: {n_balls} balls suspended by pins, rightmost pulled and released."
    return sim, _build_metadata(sim, "newtons_cradle", "constraint", difficulty, desc)


@register_scenario("pendulum", category="constraint",
    description="Single or multiple pendulums swinging")
def _scenario_pendulum(rng, difficulty):
    sim = PhysicsSimulation()
    body_s, shape_s = create_static_segment((WALL_MARGIN, GROUND_Y),
                                             (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body_s, shape_s)

    n_pend = difficulty  # 1-5
    use_double = difficulty >= 4

    for i in range(n_pend):
        anchor_x = 150 + i * 130
        anchor_y = 500
        length = rng.uniform(120, 250)
        angle = rng.uniform(30, 80) * (math.pi / 180) * rng.choice([-1, 1])
        r = rng.uniform(10, 20)
        mass = rng.uniform(1.0, 5.0)

        x = anchor_x + length * math.sin(angle)
        y = anchor_y - length * math.cos(angle)
        body, shape = create_circle((x, y), r, mass, 0.2, 0.3)
        _add_body(sim, body, shape, 'circle', mass, 0.2, 0.3, {'radius': r})

        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = (anchor_x, anchor_y)
        sim.space.add(anchor)
        joint = pymunk.PinJoint(anchor, body, (0, 0), (0, 0))
        sim.add_constraint(joint)

        # Double pendulum at high difficulty
        if use_double and i == 0:
            length2 = rng.uniform(80, 150)
            angle2 = rng.uniform(-60, 60) * (math.pi / 180)
            r2 = rng.uniform(8, 15)
            mass2 = rng.uniform(0.5, 3.0)
            x2 = x + length2 * math.sin(angle2)
            y2 = y - length2 * math.cos(angle2)
            body2, shape2 = create_circle((x2, y2), r2, mass2, 0.2, 0.3)
            _add_body(sim, body2, shape2, 'circle', mass2, 0.2, 0.3, {'radius': r2})
            joint2 = pymunk.PinJoint(body, body2, (0, 0), (0, 0))
            sim.add_constraint(joint2)

    desc = f"Pendulum: {n_pend} pendulum{'s' if n_pend > 1 else ''}{' (one double)' if use_double else ''} released from angle."
    return sim, _build_metadata(sim, "pendulum", "constraint", difficulty, desc)


@register_scenario("wrecking_ball", category="constraint",
    description="Heavy ball on pin joint swings into a tower")
def _scenario_wrecking_ball(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Tower on right
    n_blocks = 3 + difficulty * 2  # 5-13
    block_w = 35
    block_h = 20
    tower_x = 550

    for i in range(n_blocks):
        x = tower_x + rng.uniform(-2, 2)
        y = GROUND_Y + block_h / 2 + i * (block_h + 1)
        mass = 1.5
        body, shape = create_rectangle((x, y), block_w, block_h, mass, 0.6, 0.2)
        _add_body(sim, body, shape, 'rectangle', mass, 0.6, 0.2,
                 {'width': block_w, 'height': block_h})

    # Wrecking ball on pin joint
    anchor_x = 300
    anchor_y = 500
    string_len = 300
    ball_r = 20 + difficulty * 3
    ball_mass = 10.0 + difficulty * 3

    # Start from far left (pulled position)
    pull_angle = rng.uniform(50, 80) * (math.pi / 180)
    bx = anchor_x - string_len * math.sin(pull_angle)
    by = anchor_y - string_len * math.cos(pull_angle)

    body, shape = create_circle((bx, by), ball_r, ball_mass, 0.5, 0.3)
    _add_body(sim, body, shape, 'circle', ball_mass, 0.5, 0.3, {'radius': ball_r})

    anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
    anchor.position = (anchor_x, anchor_y)
    sim.space.add(anchor)
    joint = pymunk.PinJoint(anchor, body, (0, 0), (0, 0))
    sim.add_constraint(joint)

    desc = f"Wrecking ball: heavy ball ({ball_mass:.0f}kg) swings into tower of {n_blocks} blocks."
    return sim, _build_metadata(sim, "wrecking_ball", "constraint", difficulty, desc)


@register_scenario("chain", category="constraint",
    description="Connected segments dangling, then released")
def _scenario_chain(rng, difficulty):
    sim = PhysicsSimulation()
    body_s, shape_s = create_static_segment((WALL_MARGIN, GROUND_Y),
                                             (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body_s, shape_s)

    n_links = 5 + difficulty * 3  # 8-20
    start = (400, 500)
    # Chain starts horizontal, will fall under gravity
    end = (400 + n_links * 15, 500)
    link_mass = rng.uniform(0.3, 1.0)

    create_chain(sim, start, end, n_links, link_mass=link_mass,
                 link_radius=6, friction=0.4, elasticity=0.3, anchor_start=True)

    desc = f"Chain: {n_links} linked balls anchored at top, falling under gravity."
    return sim, _build_metadata(sim, "chain", "constraint", difficulty, desc)


@register_scenario("seesaw", category="constraint",
    description="Plank on pivot with objects on both sides")
def _scenario_seesaw(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Fulcrum point
    pivot_x, pivot_y = 400, GROUND_Y + 60

    # Plank
    plank_w = 300 + difficulty * 30
    plank_h = 10
    plank_mass = 3.0
    plank_body, plank_shape = create_rectangle((pivot_x, pivot_y), plank_w, plank_h,
                                                plank_mass, 0.7, 0.2)
    _add_body(sim, plank_body, plank_shape, 'rectangle', plank_mass, 0.7, 0.2,
             {'width': plank_w, 'height': plank_h})

    # Pivot joint
    anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
    anchor.position = (pivot_x, pivot_y)
    sim.space.add(anchor)
    joint = pymunk.PivotJoint(anchor, plank_body, (pivot_x, pivot_y))
    sim.add_constraint(joint)

    # Objects on both sides
    n_left = difficulty
    n_right = difficulty
    for i in range(n_left):
        r = rng.uniform(10, 20)
        mass = rng.uniform(1.0, 5.0)
        x = pivot_x - plank_w / 2 + 30 + i * 30
        y = pivot_y + plank_h / 2 + r + 50 + i * 30  # Drop from above
        body, shape = create_circle((x, y), r, mass, 0.5, 0.3)
        _add_body(sim, body, shape, 'circle', mass, 0.5, 0.3, {'radius': r})

    for i in range(n_right):
        r = rng.uniform(10, 20)
        mass = rng.uniform(2.0, 8.0)  # Heavier on right for asymmetry
        x = pivot_x + plank_w / 2 - 30 - i * 30
        y = pivot_y + plank_h / 2 + r + 80 + i * 40
        body, shape = create_circle((x, y), r, mass, 0.5, 0.3)
        _add_body(sim, body, shape, 'circle', mass, 0.5, 0.3, {'radius': r})

    desc = f"Seesaw: plank on pivot with {n_left} objects left and {n_right} right."
    return sim, _build_metadata(sim, "seesaw", "constraint", difficulty, desc)


# ===================================================================
# CATEGORY 5: Mini-game Physics
# ===================================================================

@register_scenario("angry_birds", category="minigame",
    description="Projectile launched at multi-layer block structure")
def _scenario_angry_birds(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Structure on right side
    layers = 2 + difficulty  # 3-7
    blocks_per_layer = 2 + difficulty  # 3-7
    block_w = 25
    block_h = 20
    struct_x = 550

    for layer in range(layers):
        for b in range(blocks_per_layer):
            x = struct_x + (b - blocks_per_layer / 2) * (block_w + 3)
            y = GROUND_Y + block_h / 2 + layer * (block_h + 2)
            mass = rng.uniform(0.5, 2.0)
            body, shape = create_rectangle((x, y), block_w, block_h, mass, 0.5, 0.2)
            _add_body(sim, body, shape, 'rectangle', mass, 0.5, 0.2,
                     {'width': block_w, 'height': block_h})

    # Projectiles from left
    n_proj = min(difficulty, 3)
    for i in range(n_proj):
        r = 12
        mass = 3.0 + i
        angle = rng.uniform(30, 55) * (math.pi / 180)
        speed = rng.uniform(400, 700)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        body, shape = create_circle((80 + i * 30, GROUND_Y + 30 + i * 20), r, mass, 0.5, 0.3)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, 0.5, 0.3, {'radius': r})

    n_blocks = layers * blocks_per_layer
    desc = f"Angry Birds: {n_proj} projectiles aimed at structure of {n_blocks} blocks ({layers} layers)."
    return sim, _build_metadata(sim, "angry_birds", "minigame", difficulty, desc)


@register_scenario("pinball", category="minigame",
    description="Ball bouncing between static bumpers")
def _scenario_pinball(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    n_bumpers = 3 + difficulty * 2  # 5-13
    for i in range(n_bumpers):
        bx = rng.uniform(100, 700)
        by = rng.uniform(150, 500)
        br = rng.uniform(15, 30)
        create_bumper(sim, (bx, by), br, elasticity=0.9, friction=0.1)

    # Flippers at bottom (angled segments)
    create_ramp(sim, (250, 80), (350, 120), friction=0.3, elasticity=0.8)
    create_ramp(sim, (550, 80), (450, 120), friction=0.3, elasticity=0.8)

    # Ball launched upward
    r = 10
    mass = 1.0
    speed = 500 + difficulty * 100
    body, shape = create_circle((400, 80), r, mass, 0.1, 0.85)
    body.velocity = (rng.uniform(-50, 50), speed)
    _add_body(sim, body, shape, 'circle', mass, 0.1, 0.85, {'radius': r})

    desc = f"Pinball: ball launched among {n_bumpers} bumpers."
    return sim, _build_metadata(sim, "pinball", "minigame", difficulty, desc)


@register_scenario("basketball", category="minigame",
    description="Ball arcing toward a hoop")
def _scenario_basketball(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    hoop_x = 550 + difficulty * 20
    hoop_y = 300 + difficulty * 20
    gap = max(50 - difficulty * 3, 35)
    create_hoop(sim, hoop_x, hoop_y, gap_width=gap, arm_length=25)

    # Backboard
    create_ramp(sim, (hoop_x + gap / 2 + 25, hoop_y),
                (hoop_x + gap / 2 + 25, hoop_y + 60), friction=0.5, elasticity=0.6)

    n_balls = difficulty  # 1-5
    for i in range(n_balls):
        r = 12
        mass = 1.5
        # Calculate launch angle for parabolic arc
        dx = hoop_x - 100
        dy = hoop_y - (GROUND_Y + 50)
        angle = rng.uniform(50, 70) * (math.pi / 180)
        speed = rng.uniform(350, 550)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        body, shape = create_circle((80 + i * 20, GROUND_Y + 30 + i * 15), r, mass, 0.7, 0.6)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, 0.7, 0.6, {'radius': r})

    desc = f"Basketball: {n_balls} ball{'s' if n_balls > 1 else ''} arcing toward a hoop."
    return sim, _build_metadata(sim, "basketball", "minigame", difficulty, desc)


@register_scenario("breakout", category="minigame",
    description="Ball launched at grid of blocks")
def _scenario_breakout(rng, difficulty):
    sim = PhysicsSimulation()
    # Full box (all 4 walls)
    create_u_box(sim)
    body, shape = create_static_segment((WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        0.3, 0.8)
    sim.add_static(body, shape)

    # Paddle at bottom
    create_platform(sim, 400, GROUND_Y + 30, 100, friction=0.5, elasticity=0.8)

    # Block grid at top
    rows = 2 + difficulty  # 3-7
    cols = 3 + difficulty  # 4-8
    block_w = min(60, (SCENE_WIDTH - 100) / cols - 4)
    block_h = 15
    grid_top = SCENE_HEIGHT - 100

    for row in range(rows):
        for col in range(cols):
            x = 60 + col * (block_w + 4) + block_w / 2
            y = grid_top - row * (block_h + 4)
            mass = 0.5
            body, shape = create_rectangle((x, y), block_w, block_h, mass, 0.3, 0.5)
            _add_body(sim, body, shape, 'rectangle', mass, 0.3, 0.5,
                     {'width': block_w, 'height': block_h})

    # Ball
    r = 8
    mass = 1.0
    body, shape = create_circle((400, GROUND_Y + 60), r, mass, 0.1, 0.9)
    body.velocity = (rng.uniform(-100, 100), 400 + difficulty * 50)
    _add_body(sim, body, shape, 'circle', mass, 0.1, 0.9, {'radius': r})

    n_blocks = rows * cols
    desc = f"Breakout: ball launched at {rows}x{cols} grid of {n_blocks} blocks."
    return sim, _build_metadata(sim, "breakout", "minigame", difficulty, desc)


@register_scenario("pong", category="minigame",
    description="Ball bouncing between two static paddles")
def _scenario_pong(rng, difficulty):
    sim = PhysicsSimulation(gravity=(0, 0))  # No gravity
    # Top and bottom walls
    body, shape = create_static_segment((0, WALL_MARGIN), (SCENE_WIDTH, WALL_MARGIN), 0.3, 0.9)
    sim.add_static(body, shape)
    body, shape = create_static_segment((0, SCENE_HEIGHT - WALL_MARGIN),
                                        (SCENE_WIDTH, SCENE_HEIGHT - WALL_MARGIN), 0.3, 0.9)
    sim.add_static(body, shape)

    # Paddles
    paddle_h = max(100 - difficulty * 10, 40)
    paddle_y = 300 + rng.uniform(-50, 50)
    # Left paddle
    body, shape = create_static_segment((30, paddle_y - paddle_h / 2),
                                        (30, paddle_y + paddle_h / 2), 0.3, 0.9)
    sim.add_static(body, shape)
    # Right paddle
    paddle_y2 = 300 + rng.uniform(-50, 50)
    body, shape = create_static_segment((770, paddle_y2 - paddle_h / 2),
                                        (770, paddle_y2 + paddle_h / 2), 0.3, 0.9)
    sim.add_static(body, shape)

    # Ball(s)
    n_balls = min(difficulty, 3)
    for i in range(n_balls):
        r = 8
        mass = 1.0
        speed = 300 + difficulty * 80
        angle = rng.uniform(-45, 45) * (math.pi / 180)
        vx = speed * math.cos(angle) * rng.choice([-1, 1])
        vy = speed * math.sin(angle)
        x = 400 + rng.uniform(-50, 50)
        y = 300 + rng.uniform(-100, 100)
        body, shape = create_circle((x, y), r, mass, 0.1, 0.95)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, 0.1, 0.95, {'radius': r})

    desc = f"Pong: {n_balls} ball{'s' if n_balls > 1 else ''} bouncing between two paddles."
    return sim, _build_metadata(sim, "pong", "minigame", difficulty, desc)


# ===================================================================
# CATEGORY 6: Complex & Chaotic
# ===================================================================

@register_scenario("avalanche", category="complex",
    description="Pile of objects on steep tilted surface, released")
def _scenario_avalanche(rng, difficulty):
    sim = PhysicsSimulation()
    body, shape = create_static_segment((WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, GROUND_Y), 0.8, 0.5)
    sim.add_static(body, shape)
    # Right wall to catch debris
    body, shape = create_static_segment((SCENE_WIDTH - WALL_MARGIN, GROUND_Y),
                                        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT), 0.5, 0.3)
    sim.add_static(body, shape)

    # Steep slope
    slope_angle = 30 + difficulty * 5  # 35-55 degrees
    create_ramp(sim, (50, 500), (500, 100), friction=0.2, elasticity=0.3)

    n = 10 + difficulty * 8  # 18-50
    for i in range(n):
        t = rng.uniform(0.1, 0.9)
        rx = 50 + t * 450
        ry = 500 - t * 400
        # Place slightly above ramp
        r = rng.uniform(6, 18)
        mass, friction, elasticity = _random_material(rng, mass_range=(0.5, 5.0))
        # Offset perpendicular to slope
        perp_offset = rng.uniform(r + 2, r + 30)
        nx = -0.624  # Normal to slope (approx)
        ny = 0.781
        x = rx + nx * perp_offset
        y = ry + ny * perp_offset
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Avalanche: {n} objects piled on a steep slope, sliding down."
    return sim, _build_metadata(sim, "avalanche", "complex", difficulty, desc)


@register_scenario("conveyor", category="complex",
    description="Heavy pusher body moves objects off edge")
def _scenario_conveyor(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Platform
    create_platform(sim, 400, GROUND_Y + 100, 500, friction=0.5, elasticity=0.3)

    n = 3 + difficulty * 3  # 6-18
    platform_y = GROUND_Y + 100

    # Objects sitting on platform
    for i in range(n):
        r = rng.uniform(8, 18)
        mass, friction, elasticity = _random_material(rng, mass_range=(0.3, 3.0))
        x = 250 + i * (500 / n)
        y = platform_y + r + 3
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    # Heavy pusher on left
    pusher_w = 30
    pusher_h = 60
    pusher_mass = 50.0
    body, shape = create_rectangle((130, platform_y + pusher_h / 2 + 3),
                                    pusher_w, pusher_h, pusher_mass, 0.8, 0.2)
    body.velocity = (100 + difficulty * 40, 0)
    _add_body(sim, body, shape, 'rectangle', pusher_mass, 0.8, 0.2,
             {'width': pusher_w, 'height': pusher_h})

    desc = f"Conveyor: heavy pusher pushes {n} objects along a platform."
    return sim, _build_metadata(sim, "conveyor", "complex", difficulty, desc)


@register_scenario("hourglass", category="complex",
    description="Objects falling through narrow gap between chambers")
def _scenario_hourglass(rng, difficulty):
    sim = PhysicsSimulation()

    gap_width = max(40 - difficulty * 5, 18)  # 35 down to 18
    create_hourglass_chamber(sim, center_x=400, width=300,
                             top_y=550, bottom_y=50, gap_y=300,
                             gap_width=gap_width)

    n = 10 + difficulty * 6  # 16-40
    for i in range(n):
        r = rng.uniform(4, min(gap_width / 2 - 2, 12))
        mass, friction, elasticity = _random_material(rng, mass_range=(0.3, 2.0))
        x = 400 + rng.uniform(-120, 120)
        y = rng.uniform(320, 530)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Hourglass: {n} balls in upper chamber falling through {gap_width:.0f}-pixel gap."
    return sim, _build_metadata(sim, "hourglass", "complex", difficulty, desc)


@register_scenario("wind", category="complex",
    description="Objects under lateral wind (horizontal gravity component)")
def _scenario_wind(rng, difficulty):
    wind_strength = 100 + difficulty * 80  # 180-500
    wind_dir = rng.choice([-1, 1])
    gx = wind_strength * wind_dir
    sim = PhysicsSimulation(gravity=(gx, -981))
    create_u_box(sim)

    n = 5 + difficulty * 4  # 9-25
    for i in range(n):
        shape_type = rng.choice(['circle', 'rectangle'])
        mass, friction, elasticity = _random_material(rng)
        x = rng.uniform(100, 700)
        y = rng.uniform(200, 550)

        if shape_type == 'circle':
            r = rng.uniform(8, 25)
            body, shape = create_circle((x, y), r, mass, friction, elasticity)
            _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})
        else:
            w = rng.uniform(15, 50)
            h = rng.uniform(15, 50)
            body, shape = create_rectangle((x, y), w, h, mass, friction, elasticity)
            _add_body(sim, body, shape, 'rectangle', mass, friction, elasticity, {'width': w, 'height': h})

    direction = "left" if wind_dir < 0 else "right"
    desc = f"Wind: {n} objects under lateral wind blowing {direction} (strength {wind_strength:.0f})."
    return sim, _build_metadata(sim, "wind", "complex", difficulty, desc)


@register_scenario("orbit", category="complex",
    description="Zero gravity with objects given tangential velocities")
def _scenario_orbit(rng, difficulty):
    sim = PhysicsSimulation(gravity=(0, 0))

    # Walls to contain
    create_u_box(sim, wall_friction=0.1, wall_elasticity=0.8,
                 ground_friction=0.1, ground_elasticity=0.8)
    body, shape = create_static_segment((WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
                                        0.1, 0.8)
    sim.add_static(body, shape)

    # Central large body (static attractor — approximated as a heavy dynamic body)
    center_r = 25
    center_mass = 50.0
    body, shape = create_circle((400, 300), center_r, center_mass, 0.3, 0.5)
    _add_body(sim, body, shape, 'circle', center_mass, 0.3, 0.5, {'radius': center_r})

    # Orbiting objects with tangential velocity
    n = 3 + difficulty * 2  # 5-13
    for i in range(n):
        angle = 2 * math.pi * i / n + rng.uniform(-0.2, 0.2)
        orbit_r = rng.uniform(80, 250)
        x = 400 + orbit_r * math.cos(angle)
        y = 300 + orbit_r * math.sin(angle)
        speed = rng.uniform(100, 300)
        # Tangential velocity (perpendicular to radius)
        vx = -speed * math.sin(angle)
        vy = speed * math.cos(angle)
        r = rng.uniform(6, 15)
        mass = rng.uniform(0.5, 3.0)
        body, shape = create_circle((x, y), r, mass, 0.2, 0.7)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, 0.2, 0.7, {'radius': r})

    desc = f"Orbit: {n} objects with tangential velocities in zero gravity around a central mass."
    return sim, _build_metadata(sim, "orbit", "complex", difficulty, desc)


# ===================================================================
# NEW SCENARIOS: Edge cases & Games
# ===================================================================

@register_scenario("planetary_rotation", category="complex",
    description="N-body gravitational orbits with custom gravitational force — heldout task")
def _scenario_planetary_rotation(rng, difficulty):
    """Planetary system with gravitational attraction via velocity callbacks.

    Each step, Newtonian gravity F = G*m1*m2/r^2 is applied between all pairs.
    The central 'star' is heavy; planets orbit with Keplerian-ish velocities.
    This is a heldout evaluation task (not in training).
    """
    sim = PhysicsSimulation(gravity=(0, 0))
    # Bounding box
    create_u_box(sim, wall_friction=0.1, wall_elasticity=0.5,
                 ground_friction=0.1, ground_elasticity=0.5)
    body_top, shape_top = create_static_segment(
        (WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN), 0.1, 0.5)
    sim.add_static(body_top, shape_top)

    G = 5000.0 * (1 + difficulty * 0.5)  # Gravitational constant

    # Central star
    star_r = 18 + difficulty * 3
    star_mass = 80.0 + difficulty * 20.0
    star_body, star_shape = create_circle((400, 300), star_r, star_mass, 0.3, 0.3)
    _add_body(sim, star_body, star_shape, 'circle', star_mass, 0.3, 0.3, {'radius': star_r})

    # Planets
    n_planets = 2 + difficulty  # 3 to 7
    planet_bodies = []
    for i in range(n_planets):
        angle = 2 * math.pi * i / n_planets + rng.uniform(-0.3, 0.3)
        orbit_r = rng.uniform(60, 250)
        x = 400 + orbit_r * math.cos(angle)
        y = 300 + orbit_r * math.sin(angle)
        # Keplerian velocity: v = sqrt(G*M/r)
        v_orbital = math.sqrt(G * star_mass / max(orbit_r, 30))
        # Add slight eccentricity
        v_factor = rng.uniform(0.8, 1.2)
        vx = -v_orbital * v_factor * math.sin(angle)
        vy = v_orbital * v_factor * math.cos(angle)
        r = rng.uniform(5, 12)
        mass = rng.uniform(0.5, 5.0)
        body, shape = create_circle((x, y), r, mass, 0.2, 0.5)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', mass, 0.2, 0.5, {'radius': r})
        planet_bodies.append(body)

    # Optional moons (higher difficulty)
    n_moons = max(0, difficulty - 2)
    for i in range(n_moons):
        parent = rng.choice(planet_bodies)
        moon_angle = rng.uniform(0, 2 * math.pi)
        moon_orbit = rng.uniform(20, 40)
        mx = parent.position.x + moon_orbit * math.cos(moon_angle)
        my = parent.position.y + moon_orbit * math.sin(moon_angle)
        mv = math.sqrt(G * parent.mass / max(moon_orbit, 10)) * 0.3
        mvx = parent.velocity.x - mv * math.sin(moon_angle)
        mvy = parent.velocity.y + mv * math.cos(moon_angle)
        mr = rng.uniform(3, 6)
        mm = rng.uniform(0.1, 0.5)
        body, shape = create_circle((mx, my), mr, mm, 0.1, 0.5)
        body.velocity = (mvx, mvy)
        _add_body(sim, body, shape, 'circle', mm, 0.1, 0.5, {'radius': mr})

    desc = (f"Planetary rotation: {n_planets} planets orbiting a star (mass={star_mass:.0f}) "
            f"with {n_moons} moons, G={G:.0f}.")
    return sim, _build_metadata(sim, "planetary_rotation", "complex", difficulty, desc)


@register_scenario("catapult", category="collision",
    description="Lever-based catapult launches projectile at target structure")
def _scenario_catapult(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Fulcrum (static triangle approx as circle)
    fulcrum_x = 150
    fulcrum_y = GROUND_Y + 15
    body_f, shape_f = create_static_circle((fulcrum_x, fulcrum_y), 15, 0.8, 0.3)
    sim.add_static(body_f, shape_f)

    # Arm (long rectangle resting on fulcrum)
    arm_w = 160 + difficulty * 20
    arm_h = 8
    arm_mass = 3.0
    arm_body, arm_shape = create_rectangle(
        (fulcrum_x, fulcrum_y + arm_h / 2 + 15), arm_w, arm_h,
        arm_mass, 0.5, 0.3)
    _add_body(sim, arm_body, arm_shape, 'rectangle', arm_mass, 0.5, 0.3,
              {'width': arm_w, 'height': arm_h})
    # Pin the arm to the fulcrum
    pivot = create_pivot_joint(arm_body, sim.space.static_body,
                               (fulcrum_x, fulcrum_y + 15))
    sim.add_constraint(pivot)

    # Counterweight (heavy, falls on short end)
    cw_mass = 15.0 + difficulty * 5
    cw_r = 12 + difficulty * 2
    cw_body, cw_shape = create_circle(
        (fulcrum_x - arm_w / 3, fulcrum_y + 60), cw_r, cw_mass, 0.5, 0.2)
    _add_body(sim, cw_body, cw_shape, 'circle', cw_mass, 0.5, 0.2, {'radius': cw_r})

    # Projectile (light, on long end)
    proj_r = rng.uniform(8, 14)
    proj_mass = rng.uniform(0.5, 2.0)
    proj_body, proj_shape = create_circle(
        (fulcrum_x + arm_w / 3, fulcrum_y + 25), proj_r, proj_mass, 0.3, 0.7)
    _add_body(sim, proj_body, proj_shape, 'circle', proj_mass, 0.3, 0.7, {'radius': proj_r})

    # Target structure on far right
    n_targets = 3 + difficulty * 2
    for i in range(n_targets):
        tx = 600 + rng.uniform(-30, 30)
        ty = GROUND_Y + 15 + i * 25
        tw, th = rng.uniform(20, 40), rng.uniform(15, 25)
        tm = rng.uniform(1.0, 4.0)
        body, shape = create_rectangle((tx, ty), tw, th, tm, 0.6, 0.4)
        _add_body(sim, body, shape, 'rectangle', tm, 0.6, 0.4, {'width': tw, 'height': th})

    desc = (f"Catapult: counterweight (mass={cw_mass:.0f}) drops to launch "
            f"projectile at {n_targets} target blocks.")
    return sim, _build_metadata(sim, "catapult", "collision", difficulty, desc)


@register_scenario("zero_g_pool", category="minigame",
    description="Billiards in zero gravity with 4-wall bouncing")
def _scenario_zero_g_pool(rng, difficulty):
    sim = PhysicsSimulation(gravity=(0, 0))
    # 4-wall enclosure
    create_u_box(sim, ground_friction=0.02, ground_elasticity=0.95,
                 wall_friction=0.02, wall_elasticity=0.95)
    body_top, shape_top = create_static_segment(
        (WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN), 0.02, 0.95)
    sim.add_static(body_top, shape_top)

    n = 5 + difficulty * 3  # 8-20
    r = 12
    elasticity = 0.95
    friction = 0.02
    mass = 1.0
    for i in range(n):
        x = rng.uniform(50, 750)
        y = rng.uniform(80, 520)
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        speed = rng.uniform(50, 300)
        angle = rng.uniform(0, 2 * math.pi)
        body.velocity = (speed * math.cos(angle), speed * math.sin(angle))
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Zero-G pool: {n} balls bouncing in enclosed box, no gravity."
    return sim, _build_metadata(sim, "zero_g_pool", "minigame", difficulty, desc)


@register_scenario("domino_spiral", category="stacking",
    description="Dominoes arranged in spiral pattern for chain reaction")
def _scenario_domino_spiral(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    n = 15 + difficulty * 10  # 25-65
    cx, cy = 400, 300
    spacing = 25
    domino_w, domino_h = 8, 30
    mass = 1.5
    friction = 0.6
    elasticity = 0.3

    for i in range(n):
        t = i * 0.25  # spiral parameter
        r = 50 + spacing * t / (2 * math.pi)
        angle = t
        x = cx + r * math.cos(angle)
        y = max(GROUND_Y + domino_h / 2 + 5, cy + r * math.sin(angle) * 0.3)
        body, shape = create_rectangle((x, y), domino_w, domino_h, mass, friction, elasticity)
        # Tilt dominoes slightly outward
        body.angle = angle + math.pi / 2
        _add_body(sim, body, shape, 'rectangle', mass, friction, elasticity,
                  {'width': domino_w, 'height': domino_h})

    # Trigger ball
    trigger_r = 15
    trigger_mass = 5.0
    first_x = cx + 50
    trigger_body, trigger_shape = create_circle(
        (first_x - 30, GROUND_Y + 50), trigger_r, trigger_mass, 0.4, 0.5)
    trigger_body.velocity = (200, 0)
    _add_body(sim, trigger_body, trigger_shape, 'circle', trigger_mass, 0.4, 0.5,
              {'radius': trigger_r})

    desc = f"Domino spiral: {n} dominoes in spiral with trigger ball."
    return sim, _build_metadata(sim, "domino_spiral", "stacking", difficulty, desc)


@register_scenario("trampoline", category="ramp",
    description="Objects bouncing on high-elasticity trampoline surface")
def _scenario_trampoline(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim, ground_elasticity=0.3)

    # Trampoline surface (high elasticity segment)
    tramp_y = GROUND_Y + 30
    tramp_left = 200
    tramp_right = 600
    body_t, shape_t = create_static_segment(
        (tramp_left, tramp_y), (tramp_right, tramp_y),
        friction=0.3, elasticity=0.92)
    sim.add_static(body_t, shape_t)

    # Objects dropped from various heights
    n = 3 + difficulty * 2  # 5-13
    for i in range(n):
        x = rng.uniform(tramp_left + 20, tramp_right - 20)
        y = rng.uniform(300, 560)
        r = rng.uniform(8, 20)
        mass, friction, elasticity = _random_material(
            rng, elasticity_range=(0.7, 0.95))
        body, shape = create_circle((x, y), r, mass, friction, elasticity)
        _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})

    desc = f"Trampoline: {n} balls dropped onto high-bounce surface."
    return sim, _build_metadata(sim, "trampoline", "ramp", difficulty, desc)


@register_scenario("pachinko", category="minigame",
    description="Ball dropped through dense peg field (Galton board style)")
def _scenario_pachinko(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Dense peg grid
    rows = 6 + difficulty * 2  # 8-16
    cols = 8 + difficulty * 2  # 10-18
    peg_r = 5
    x_spacing = (SCENE_WIDTH - 100) / (cols + 1)
    y_spacing = (SCENE_HEIGHT - 200) / (rows + 1)

    for row in range(rows):
        offset = x_spacing / 2 if row % 2 == 1 else 0
        for col in range(cols):
            px = 50 + x_spacing * (col + 1) + offset
            py = SCENE_HEIGHT - 80 - y_spacing * (row + 1)
            if py > GROUND_Y + 30:
                body_p, shape_p = create_static_circle((px, py), peg_r, 0.3, 0.6)
                sim.add_static(body_p, shape_p)

    # Drop balls from top
    n_balls = 2 + difficulty  # 3-7
    for i in range(n_balls):
        x = 400 + rng.uniform(-50, 50)
        y = SCENE_HEIGHT - 40 - i * 15
        r = rng.uniform(6, 10)
        mass = rng.uniform(0.5, 2.0)
        body, shape = create_circle((x, y), r, mass, 0.3, 0.6)
        _add_body(sim, body, shape, 'circle', mass, 0.3, 0.6, {'radius': r})

    desc = f"Pachinko: {n_balls} balls through {rows}x{cols} peg grid."
    return sim, _build_metadata(sim, "pachinko", "minigame", difficulty, desc)


@register_scenario("elastic_collision", category="collision",
    description="Head-on collision between equal or unequal mass objects — edge case for conservation laws")
def _scenario_elastic_collision(rng, difficulty):
    """Edge case: perfectly clean elastic collisions to test conservation."""
    sim = PhysicsSimulation(gravity=(0, 0))
    create_u_box(sim, ground_friction=0.0, ground_elasticity=0.99,
                 wall_friction=0.0, wall_elasticity=0.99)
    body_top, shape_top = create_static_segment(
        (WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN),
        (SCENE_WIDTH - WALL_MARGIN, SCENE_HEIGHT - WALL_MARGIN), 0.0, 0.99)
    sim.add_static(body_top, shape_top)

    n_pairs = 1 + difficulty  # 2-6 pairs
    for i in range(n_pairs):
        y = 100 + i * (400 / max(n_pairs, 1))
        # Left object
        m1 = rng.uniform(1.0, 10.0)
        r1 = 10 + m1
        v1 = rng.uniform(100, 400)
        body1, shape1 = create_circle((100, y), r1, m1, 0.0, 0.99)
        body1.velocity = (v1, 0)
        _add_body(sim, body1, shape1, 'circle', m1, 0.0, 0.99, {'radius': r1})
        # Right object
        mass_ratio = rng.choice([1.0, 0.5, 2.0, 0.1, 10.0])
        m2 = m1 * mass_ratio
        r2 = 10 + m2
        v2 = -rng.uniform(50, 200)
        body2, shape2 = create_circle((700, y), min(r2, 30), m2, 0.0, 0.99)
        body2.velocity = (v2, 0)
        _add_body(sim, body2, shape2, 'circle', m2, 0.0, 0.99, {'radius': min(r2, 30)})

    desc = f"Elastic collision: {n_pairs} pairs, head-on, near-perfect elasticity."
    return sim, _build_metadata(sim, "elastic_collision", "collision", difficulty, desc)


@register_scenario("stack_collapse", category="stacking",
    description="Tall unstable stack that collapses — tests long-range cascading dynamics")
def _scenario_stack_collapse(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Build a deliberately unstable tall stack
    n_layers = 5 + difficulty * 3  # 8-20
    blocks_per_layer = 2 + difficulty  # 3-7
    for layer in range(n_layers):
        for b in range(blocks_per_layer):
            w = rng.uniform(30, 60)
            h = rng.uniform(12, 20)
            # Slight offset to create instability
            offset = rng.uniform(-8, 8) * (layer / n_layers)
            x = 400 + (b - blocks_per_layer / 2) * (w + 2) + offset
            y = GROUND_Y + 10 + layer * (h + 1)
            mass = rng.uniform(1.0, 4.0)
            body, shape = create_rectangle(
                (x, y), w, h, mass, 0.5, 0.3)
            _add_body(sim, body, shape, 'rectangle', mass, 0.5, 0.3,
                      {'width': w, 'height': h})

    # Wrecking ball from the side
    wb_r = 15 + difficulty * 3
    wb_mass = 20.0 + difficulty * 5
    wb_body, wb_shape = create_circle((50, 200), wb_r, wb_mass, 0.3, 0.4)
    wb_body.velocity = (500 + difficulty * 50, 100)
    _add_body(sim, wb_body, wb_shape, 'circle', wb_mass, 0.3, 0.4, {'radius': wb_r})

    total_blocks = n_layers * blocks_per_layer
    desc = (f"Stack collapse: {total_blocks} blocks in {n_layers} layers, "
            f"hit by wrecking ball (mass={wb_mass:.0f}).")
    return sim, _build_metadata(sim, "stack_collapse", "stacking", difficulty, desc)


@register_scenario("pendulum_wave", category="constraint",
    description="Multiple pendulums with slightly different lengths creating wave patterns")
def _scenario_pendulum_wave(rng, difficulty):
    sim = PhysicsSimulation()
    create_u_box(sim)

    # Ceiling beam
    beam_y = SCENE_HEIGHT - 40
    body_beam, shape_beam = create_static_segment(
        (100, beam_y), (700, beam_y), 0.5, 0.3)
    sim.add_static(body_beam, shape_beam)

    n = 8 + difficulty * 3  # 11-23
    base_length = 150
    length_increment = 3.0  # Each pendulum slightly longer

    for i in range(n):
        anchor_x = 100 + (600 / (n + 1)) * (i + 1)
        length = base_length + i * length_increment
        bob_x = anchor_x
        bob_y = beam_y - length
        r = rng.uniform(8, 14)
        mass = rng.uniform(1.0, 3.0)
        body, shape = create_circle((bob_x, bob_y), r, mass, 0.2, 0.6)
        _add_body(sim, body, shape, 'circle', mass, 0.2, 0.6, {'radius': r})
        # Pin joint to ceiling
        joint = create_pin_joint(body, sim.space.static_body,
                                 (0, 0), (anchor_x, beam_y))
        sim.add_constraint(joint)
        # Initial displacement (all released from same angle)
        swing_angle = 0.5  # radians
        body.position = (anchor_x + length * math.sin(swing_angle),
                         beam_y - length * math.cos(swing_angle))

    desc = f"Pendulum wave: {n} pendulums with incrementing lengths, released in sync."
    return sim, _build_metadata(sim, "pendulum_wave", "constraint", difficulty, desc)


@register_scenario("gravity_well", category="complex",
    description="Lateral gravity gradient pulling objects toward a point — edge case")
def _scenario_gravity_well(rng, difficulty):
    """Non-uniform gravity field approximated by lateral wind + standard gravity."""
    # Use angled gravity to simulate gravitational well
    gx = rng.choice([-200, 200]) * (1 + difficulty * 0.3)
    sim = PhysicsSimulation(gravity=(gx, -981))
    create_u_box(sim)

    n = 6 + difficulty * 4  # 10-26
    for i in range(n):
        shape_type = rng.choice(['circle', 'rectangle'])
        x = rng.uniform(80, 720)
        y = rng.uniform(100, 550)
        mass, friction, elasticity = _random_material(rng)
        if shape_type == 'circle':
            r = rng.uniform(8, 22)
            body, shape = create_circle((x, y), r, mass, friction, elasticity)
            _add_body(sim, body, shape, 'circle', mass, friction, elasticity, {'radius': r})
        else:
            w = rng.uniform(15, 45)
            h = rng.uniform(15, 45)
            body, shape = create_rectangle((x, y), w, h, mass, friction, elasticity)
            _add_body(sim, body, shape, 'rectangle', mass, friction, elasticity,
                      {'width': w, 'height': h})

    direction = "left" if gx < 0 else "right"
    desc = f"Gravity well: {n} objects in angled gravity field ({direction}, gx={gx:.0f})."
    return sim, _build_metadata(sim, "gravity_well", "complex", difficulty, desc)


@register_scenario("particle_explosion", category="complex",
    description="50-100 small particles explode radially from center")
def _scenario_particle_explosion(rng, difficulty):
    """Many small particles explode outward."""
    sim = PhysicsSimulation()
    create_u_box(sim)
    
    n = 50 + int(difficulty * 50)  # 50-100 particles
    center_x, center_y = 400, 400
    
    for i in range(n):
        angle = rng.uniform(0, 6.28)
        speed = rng.uniform(100, 400) * (1 + difficulty * 0.5)
        r = rng.uniform(3, 8)  # small particles
        
        x = center_x + rng.uniform(-20, 20)
        y = center_y + rng.uniform(-20, 20)
        
        vx = speed * (angle % 3.14)
        vy = speed * (angle / 3.14 - 1)
        
        body, shape = create_circle((x, y), r, mass=1.0, friction=0.3, elasticity=0.6)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', 1.0, 0.3, 0.6, {'radius': r})
    
    desc = f"Particle explosion: {n} particles explode radially from center."
    return sim, _build_metadata(sim, "particle_explosion", "complex", difficulty, desc)


@register_scenario("chain_reaction", category="complex",
    description="100+ dominos creating massive chain reaction")
def _scenario_chain_reaction(rng, difficulty):
    """Massive domino chain reaction."""
    sim = PhysicsSimulation()
    create_u_box(sim)
    
    n = 100 + int(difficulty * 100)  # 100-200 dominos
    spacing = 25
    
    # Spiral pattern
    for i in range(n):
        angle = i * 0.15
        radius = 100 + i * 2
        x = 400 + radius * (angle % 1.57)
        y = 400 + radius * (angle / 1.57 - 0.5)
        
        body, shape = create_rectangle((x, y), 8, 30, mass=0.5, friction=0.4, elasticity=0.2)
        _add_body(sim, body, shape, 'rectangle', 0.5, 0.4, 0.2, {'width': 8, 'height': 30})
    
    # Trigger ball
    ball_body, ball_shape = create_circle((50, 400), 15, mass=5.0, friction=0.1, elasticity=0.8)
    ball_body.velocity = (300, 0)
    _add_body(sim, ball_body, ball_shape, 'circle', 5.0, 0.1, 0.8, {'radius': 15})
    
    desc = f"Chain reaction: {n} dominos in spiral + trigger ball."
    return sim, _build_metadata(sim, "chain_reaction", "complex", difficulty, desc)


@register_scenario("fluid_sim", category="complex",
    description="Fluid simulation using many small particles")
def _scenario_fluid_sim(rng, difficulty):
    """Pseudo-fluid using particle system."""
    sim = PhysicsSimulation()
    create_u_box(sim)
    
    n = 60 + int(difficulty * 60)  # 60-120 particles
    
    # Start in container at top
    for i in range(n):
        x = rng.uniform(250, 550)
        y = rng.uniform(100, 300)
        r = rng.uniform(4, 7)
        
        body, shape = create_circle((x, y), r, mass=0.8, friction=0.1, elasticity=0.3)
        body.velocity = (rng.uniform(-50, 50), 0)
        _add_body(sim, body, shape, 'circle', 0.8, 0.1, 0.3, {'radius': r})
    
    # Add barriers to make it flow (static objects)
    for bx in [200, 400, 600]:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (bx, 450)
        shape = pymunk.Poly.create_box(body, (80, 20))
        shape.friction = 0.5
        shape.elasticity = 0.2
        sim.add_static(body, shape)
    
    desc = f"Fluid sim: {n} particles flowing through barriers."
    return sim, _build_metadata(sim, "fluid_sim", "complex", difficulty, desc)


@register_scenario("solar_system", category="complex",
    description="Planets orbiting around central sun with gravity")
def _scenario_solar_system(rng, difficulty):
    """Orbital mechanics — planets around sun."""
    # Use orbit scenario as base (already exists)
    sim = PhysicsSimulation()
    create_u_box(sim)
    
    # Central "sun" (static)
    sun_body, sun_shape = create_circle((400, 400), 40, mass=1000, friction=0, elasticity=0.9)
    sun_body.body_type = pymunk.Body.STATIC
    _add_body(sim, sun_body, sun_shape, 'circle', 1000, 0, 0.9, {'radius': 40})
    
    n_planets = 3 + difficulty * 2  # 5-9 planets
    
    for i in range(n_planets):
        angle = i * 6.28 / n_planets
        radius = 80 + i * 40
        x = 400 + radius * (angle % 3.14)
        y = 400 + radius * (angle / 1.57 - 1)
        
        # Orbital velocity
        speed = 150 / (radius ** 0.5)
        vx = -speed * (angle / 1.57 - 1)
        vy = speed * (angle % 1.57)
        
        r = rng.uniform(8, 15)
        body, shape = create_circle((x, y), r, mass=1.0, friction=0, elasticity=0.9)
        body.velocity = (vx, vy)
        _add_body(sim, body, shape, 'circle', 1.0, 0, 0.9, {'radius': r})
    
    desc = f"Solar system: {n_planets} planets orbiting central sun."
    return sim, _build_metadata(sim, "solar_system", "complex", difficulty, desc)


# ===================================================================
# Entry Point
# ===================================================================

# Compute after all registrations
SCENARIO_TYPES = list_scenarios()


def generate_scenario(
    seed: int,
    scenario_type: Optional[str] = None,
    difficulty: Optional[int] = None,
    gravity: Optional[Tuple[float, float]] = None,
) -> Tuple[PhysicsSimulation, Dict[str, Any]]:
    """
    Generate a deterministic physics scenario.

    Args:
        seed: Random seed for reproducibility
        scenario_type: Specific scenario name, or None for random
        difficulty: 1-5 (None = random based on seed)
        gravity: Override gravity (None = scenario default)

    Returns:
        (PhysicsSimulation, metadata_dict)
    """
    rng = random.Random(seed)

    if scenario_type is None:
        scenario_type = rng.choice(SCENARIO_TYPES)

    if difficulty is None:
        info = get_scenario(scenario_type)
        lo, hi = info["difficulty_range"]
        difficulty = rng.randint(lo, hi)

    info = get_scenario(scenario_type)
    sim, metadata = info["fn"](rng, difficulty)

    if gravity is not None:
        sim.space.gravity = gravity
        # Update metadata to reflect override
        metadata["gravity_override"] = {"x": gravity[0], "y": gravity[1]}

    metadata["seed"] = seed
    return sim, metadata
