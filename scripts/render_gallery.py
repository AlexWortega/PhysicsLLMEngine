#!/usr/bin/env python3
"""Render a gallery of all 30 scenario types as GIF animations."""

import math
import os
import sys
sys.path.insert(0, "/home/alexw")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import pymunk

from src.physics.scenario_generator import generate_scenario, SCENARIO_TYPES
from src.physics.scenario_registry import get_scenario


COLORS = {
    "circle": "#3498db",
    "rectangle": "#e74c3c",
    "static_segment": "#7f8c8d",
    "static_circle": "#95a5a6",
    "constraint": "#f39c12",
}

CATEGORY_COLORS = {
    "collision": "#e74c3c",
    "stacking": "#3498db",
    "ramp": "#2ecc71",
    "constraint": "#f39c12",
    "minigame": "#9b59b6",
    "complex": "#1abc9c",
}


def render_scenario_gif(scenario_type, output_path, seed=42, difficulty=3,
                        n_frames=120, fps=20, size=(400, 300)):
    """Render a single scenario as an animated GIF."""
    sim, meta = generate_scenario(seed=seed, scenario_type=scenario_type, difficulty=difficulty)

    # Pre-simulate all frames
    frames_data = []
    frames_data.append(sim.get_state())
    for _ in range(n_frames):
        sim.step()
        frames_data.append(sim.get_state())

    fig, ax = plt.subplots(1, 1, figsize=(size[0] / 80, size[1] / 80), dpi=80)

    info = get_scenario(scenario_type)
    cat = info["category"]
    cat_color = CATEGORY_COLORS.get(cat, "#333")

    ax.set_xlim(-10, 810)
    ax.set_ylim(-10, 610)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    title = ax.set_title(
        f"{scenario_type.replace('_', ' ').title()}",
        color=cat_color, fontsize=11, fontweight="bold", pad=4
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw static geometry
    static_artists = []
    for body, shape in zip(sim.static_bodies, sim.static_shapes):
        if isinstance(shape, pymunk.Segment):
            a = shape.a
            b = shape.b
            pos = body.position
            x1, y1 = a.x + pos.x, a.y + pos.y
            x2, y2 = b.x + pos.x, b.y + pos.y
            line, = ax.plot([x1, x2], [y1, y2], color="#555555", linewidth=2, zorder=1)
            static_artists.append(line)
        elif isinstance(shape, pymunk.Circle):
            circ = plt.Circle((body.position.x, body.position.y), shape.radius,
                              color="#555555", zorder=1)
            ax.add_patch(circ)
            static_artists.append(circ)

    # Draw constraint lines (pin joints)
    constraint_lines = []
    for c in sim.constraints:
        if isinstance(c, pymunk.PinJoint):
            line, = ax.plot([], [], color=COLORS["constraint"], linewidth=1,
                           alpha=0.6, zorder=2)
            constraint_lines.append((c, line))

    # Dynamic objects
    obj_artists = []
    for body in sim.bodies:
        shape_type = getattr(body, "shape_type", "circle")
        color = COLORS.get(shape_type, "#3498db")
        size_info = getattr(body, "size_info", {})
        if shape_type == "circle":
            r = size_info.get("radius", 10)
            circ = plt.Circle((0, 0), r, color=color, alpha=0.85, zorder=3)
            ax.add_patch(circ)
            obj_artists.append(("circle", circ, r))
        else:
            w = size_info.get("width", 30)
            h = size_info.get("height", 20)
            rect = patches.Rectangle((0, 0), w, h, color=color, alpha=0.85, zorder=3)
            ax.add_patch(rect)
            obj_artists.append(("rect", rect, w, h))

    frame_text = ax.text(10, 585, "", color="#aaa", fontsize=7, zorder=10)

    def update(frame_idx):
        state = frames_data[frame_idx]
        for i, obj in enumerate(state["objects"]):
            if i >= len(obj_artists):
                break
            px = obj["position"]["x"]
            py = obj["position"]["y"]
            angle = obj.get("angle", 0)
            kind = obj_artists[i][0]
            if kind == "circle":
                _, circ_patch, r = obj_artists[i]
                circ_patch.center = (px, py)
            else:
                _, rect_patch, w, h = obj_artists[i]
                rect_patch.set_xy((px - w / 2, py - h / 2))

        # Update constraint lines
        for c, line in constraint_lines:
            if isinstance(c, pymunk.PinJoint):
                a_pos = c.a.local_to_world(c.anchor_a)
                b_pos = c.b.local_to_world(c.anchor_b)
                line.set_data([a_pos.x, b_pos.x], [a_pos.y, b_pos.y])

        frame_text.set_text(f"f:{frame_idx}")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False, interval=1000 // fps)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_gallery_image(gif_dir, output_path, cols=6):
    """Combine first frames of each GIF into a single gallery image."""
    gifs = sorted([f for f in os.listdir(gif_dir) if f.endswith(".gif")])
    if not gifs:
        return

    rows = math.ceil(len(gifs) / cols)
    images = []
    for g in gifs:
        img = Image.open(os.path.join(gif_dir, g))
        img.seek(30)  # Get a frame mid-animation for more interesting view
        images.append(img.copy())

    w, h = images[0].size
    gallery = Image.new("RGB", (cols * w, rows * h), (15, 15, 35))
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        gallery.paste(img.convert("RGB"), (col * w, row * h))

    gallery.save(output_path)
    print(f"Gallery image saved: {output_path} ({cols}x{rows}, {gallery.size[0]}x{gallery.size[1]}px)")


def main():
    out_dir = "/home/alexw/evaluation_results/scenario_gallery"
    os.makedirs(out_dir, exist_ok=True)

    categories_order = ["collision", "stacking", "ramp", "constraint", "minigame", "complex"]
    ordered = []
    for cat in categories_order:
        from src.physics.scenario_registry import list_scenarios as ls
        ordered.extend(ls(category=cat))

    print(f"Rendering {len(ordered)} scenarios...")
    for i, st in enumerate(ordered):
        info = get_scenario(st)
        gif_path = os.path.join(out_dir, f"{i + 1:02d}_{st}.gif")
        print(f"  [{i + 1:02d}/{len(ordered)}] {st} ({info['category']})...", end=" ", flush=True)
        render_scenario_gif(st, gif_path, seed=42, difficulty=3, n_frames=120, fps=20)
        print("done")

    # Make combined gallery image
    gallery_path = os.path.join(out_dir, "gallery.png")
    make_gallery_image(out_dir, gallery_path, cols=6)

    print(f"\nAll files in {out_dir}/")


if __name__ == "__main__":
    main()
