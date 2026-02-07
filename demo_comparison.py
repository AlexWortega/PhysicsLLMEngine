#!/usr/bin/env python3
"""
Demo: Ground Truth vs Mock Prediction Comparison GIF
Green = Ground Truth (Pymunk simulation)
Blue = Prediction (demo: GT + random noise to simulate LLM error)
"""
import sys
import pymunk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, str(__file__).rsplit('/', 1)[0])
from src.physics.scenario_generator import generate_scenario

def render_comparison_gif(scenario_type: str, difficulty: int, seed: int, out_path: str):
    """Render side-by-side GT vs Prediction GIF."""
    
    # Generate scenario
    sim, metadata = generate_scenario(seed=seed, scenario_type=scenario_type, difficulty=difficulty)
    
    # Run GT simulation
    frames_gt = []
    for _ in range(60):  # 1 second @ 60 FPS
        sim.step()
        frame_data = []
        for body in sim.space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                frame_data.append({
                    'x': body.position.x,
                    'y': body.position.y,
                    'angle': body.angle,
                    'shapes': []
                })
                for shape in body.shapes:
                    if isinstance(shape, pymunk.Circle):
                        frame_data[-1]['shapes'].append(('circle', shape.radius))
                    elif isinstance(shape, pymunk.Poly):
                        verts = [body.local_to_world(v) for v in shape.get_vertices()]
                        frame_data[-1]['shapes'].append(('poly', verts))
        frames_gt.append(frame_data)
    
    # Create "prediction" (GT + noise)
    frames_pred = []
    for frame in frames_gt:
        noisy_frame = []
        for obj in frame:
            # Add small random error (simulating LLM inaccuracy)
            noise_x = np.random.normal(0, 5)
            noise_y = np.random.normal(0, 5)
            noisy_frame.append({
                'x': obj['x'] + noise_x,
                'y': obj['y'] + noise_y,
                'angle': obj['angle'] + np.random.normal(0, 0.1),
                'shapes': obj['shapes']
            })
        frames_pred.append(noisy_frame)
    
    # Render side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlim(0, 800)
    ax1.set_ylim(0, 800)
    ax1.set_aspect('equal')
    ax1.set_title('Ground Truth (Pymunk)', color='green', fontsize=14, fontweight='bold')
    ax1.set_facecolor('#1a1a1a')
    
    ax2.set_xlim(0, 800)
    ax2.set_ylim(0, 800)
    ax2.set_aspect('equal')
    ax2.set_title('LLM Prediction', color='cyan', fontsize=14, fontweight='bold')
    ax2.set_facecolor('#1a1a1a')
    
    fig.patch.set_facecolor('#0a0a0a')
    
    def animate(i):
        ax1.clear()
        ax2.clear()
        
        # Reset styling
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 800)
            ax.set_ylim(0, 800)
            ax.set_aspect('equal')
            ax.set_facecolor('#1a1a1a')
            ax.axis('off')
        
        ax1.text(400, 750, 'Ground Truth (Pymunk)', ha='center', color='lime', 
                fontsize=12, fontweight='bold')
        ax2.text(400, 750, 'LLM Prediction', ha='center', color='cyan', 
                fontsize=12, fontweight='bold')
        
        # Draw GT (green)
        for obj in frames_gt[i]:
            for shape_type, shape_data in obj['shapes']:
                if shape_type == 'circle':
                    circle = patches.Circle((obj['x'], obj['y']), shape_data, 
                                          color='lime', alpha=0.7, linewidth=2)
                    ax1.add_patch(circle)
                elif shape_type == 'poly':
                    poly = patches.Polygon(shape_data, closed=True, 
                                          color='lime', alpha=0.7, linewidth=2)
                    ax1.add_patch(poly)
        
        # Draw Prediction (blue)
        for obj in frames_pred[i]:
            for shape_type, shape_data in obj['shapes']:
                if shape_type == 'circle':
                    circle = patches.Circle((obj['x'], obj['y']), shape_data, 
                                          color='cyan', alpha=0.7, linewidth=2)
                    ax2.add_patch(circle)
                elif shape_type == 'poly':
                    # Need to recalculate vertices with noisy position/angle
                    poly = patches.Polygon(shape_data, closed=True, 
                                          color='cyan', alpha=0.7, linewidth=2)
                    ax2.add_patch(poly)
    
    anim = FuncAnimation(fig, animate, frames=len(frames_gt), interval=1000/15, repeat=True)
    writer = PillowWriter(fps=15)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "particle_explosion"
    difficulty = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    out_path = f"/tmp/PhysicsLLMEngine/comparison_{scenario}.gif"
    render_comparison_gif(scenario, difficulty, seed, out_path)
