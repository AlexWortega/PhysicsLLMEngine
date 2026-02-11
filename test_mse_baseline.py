#!/usr/bin/env python3
"""
MSE Baseline Tests - Naive Physics Predictors vs Ground Truth

Tests 3 baseline prediction strategies:
1. Constant position (objects don't move)
2. Linear extrapolation (last velocity × dt)
3. Gravity-only (ignore collisions, just apply gravity)

Compares against Pymunk ground truth to establish MSE baselines.
When LLM is trained, it should beat these baselines significantly.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.scenario_generator import generate_scenario


def calculate_mse(pred_positions, gt_positions):
    """Calculate Mean Squared Error between predicted and ground truth positions."""
    pred = np.array(pred_positions)
    gt = np.array(gt_positions)
    return np.mean((pred - gt) ** 2)


def extract_positions(sim):
    """Extract (x, y) positions from all dynamic bodies."""
    positions = []
    for body in sim.bodies:
        positions.append([body.position.x, body.position.y])
    return positions


def baseline_constant(context_frames):
    """Naive baseline: predict same position as last context frame."""
    return context_frames[-1]


def baseline_linear_extrapolation(context_frames, velocities, dt=1/60.0):
    """Naive baseline: last_pos + velocity * dt (no collisions, no gravity)."""
    last_pos = np.array(context_frames[-1])
    vel = np.array(velocities)
    return (last_pos + vel * dt).tolist()


def baseline_gravity_only(context_frames, velocities, gravity_y=-981, dt=1/60.0):
    """Naive baseline: apply gravity, ignore collisions."""
    last_pos = np.array(context_frames[-1])
    vel = np.array(velocities)
    
    # Update velocity with gravity
    vel[:, 1] += gravity_y * dt
    
    # Update position
    new_pos = last_pos + vel * dt
    return new_pos.tolist()


def evaluate_scenario(scenario_type, difficulty, seed, context_frames=10, predict_frames=5):
    """
    Generate scene, run GT simulation, compare against baselines.
    
    Returns MSE for each baseline strategy.
    """
    
    # Generate scenario
    sim, metadata = generate_scenario(seed=seed, scenario_type=scenario_type, difficulty=difficulty)
    
    # Run simulation and capture frames
    all_positions = []
    all_velocities = []
    
    for frame_idx in range(context_frames + predict_frames):
        sim.step()
        positions = extract_positions(sim)
        velocities = [[b.velocity.x, b.velocity.y] for b in sim.bodies]
        
        all_positions.append(positions)
        all_velocities.append(velocities)
    
    # Split into context and prediction
    context_positions = all_positions[:context_frames]
    gt_predictions = all_positions[context_frames:]
    
    last_velocities = all_velocities[context_frames - 1]
    
    # Evaluate baselines
    results = {
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "seed": seed,
        "num_objects": len(sim.bodies),
        "context_frames": context_frames,
        "predict_frames": predict_frames,
        "baselines": {}
    }
    
    for i in range(predict_frames):
        # Baseline 1: Constant position
        pred_constant = baseline_constant(context_positions)
        mse_constant = calculate_mse(pred_constant, gt_predictions[i])
        
        # Baseline 2: Linear extrapolation
        pred_linear = baseline_linear_extrapolation(context_positions, last_velocities)
        mse_linear = calculate_mse(pred_linear, gt_predictions[i])
        
        # Baseline 3: Gravity only
        pred_gravity = baseline_gravity_only(context_positions, last_velocities)
        mse_gravity = calculate_mse(pred_gravity, gt_predictions[i])
        
        results["baselines"][f"frame_{i+1}"] = {
            "constant_mse": float(mse_constant),
            "linear_mse": float(mse_linear),
            "gravity_only_mse": float(mse_gravity)
        }
    
    # Calculate average MSE across all predict frames
    avg_constant = np.mean([results["baselines"][f"frame_{i+1}"]["constant_mse"] for i in range(predict_frames)])
    avg_linear = np.mean([results["baselines"][f"frame_{i+1}"]["linear_mse"] for i in range(predict_frames)])
    avg_gravity = np.mean([results["baselines"][f"frame_{i+1}"]["gravity_only_mse"] for i in range(predict_frames)])
    
    results["average_mse"] = {
        "constant": float(avg_constant),
        "linear": float(avg_linear),
        "gravity_only": float(avg_gravity)
    }
    
    return results


def main():
    print("\n🧪 PhysicsLLMEngine MSE Baseline Evaluation")
    print("=" * 80)
    print()
    print("Testing naive prediction strategies vs Ground Truth (Pymunk simulation)")
    print("Baselines: Constant Position | Linear Extrapolation | Gravity Only")
    print()
    
    # Test on held-out scenarios
    held_out_scenarios = [
        ("pong", 3),
        ("bowling", 3),
        ("ramp_roll", 3),
        ("angry_birds", 2),  # Lower difficulty for speed
        ("hourglass", 2),
        ("newtons_cradle", 3)
    ]
    
    all_results = []
    
    for scenario_type, difficulty in held_out_scenarios:
        seed = np.random.randint(10000000, 99999999)
        
        print(f"📊 Evaluating: {scenario_type} (difficulty {difficulty}, seed {seed})")
        
        try:
            results = evaluate_scenario(
                scenario_type=scenario_type,
                difficulty=difficulty,
                seed=seed,
                context_frames=10,
                predict_frames=5
            )
            
            all_results.append(results)
            
            avg = results["average_mse"]
            print(f"  Objects: {results['num_objects']}")
            print(f"  Constant MSE:       {avg['constant']:.2f}")
            print(f"  Linear MSE:         {avg['linear']:.2f}")
            print(f"  Gravity-Only MSE:   {avg['gravity_only']:.2f}")
            
            # Best baseline
            best_baseline = min(avg, key=avg.get)
            best_mse = avg[best_baseline]
            print(f"  🏆 Best baseline: {best_baseline} ({best_mse:.2f})")
            print()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}\n")
            all_results.append({"error": str(e), "scenario_type": scenario_type})
    
    # Save results
    output_file = Path(__file__).parent / "mse_baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("=" * 80)
    print("📊 BASELINE MSE SUMMARY")
    print("=" * 80)
    print()
    
    valid_results = [r for r in all_results if "average_mse" in r]
    
    if valid_results:
        avg_constant = np.mean([r["average_mse"]["constant"] for r in valid_results])
        avg_linear = np.mean([r["average_mse"]["linear"] for r in valid_results])
        avg_gravity = np.mean([r["average_mse"]["gravity_only"] for r in valid_results])
        
        print(f"Average MSE across {len(valid_results)} scenarios:")
        print(f"  Constant Position:     {avg_constant:.2f}")
        print(f"  Linear Extrapolation:  {avg_linear:.2f}")
        print(f"  Gravity Only:          {avg_gravity:.2f}")
        print()
        print(f"🎯 **Target for LLM:** Beat {min(avg_constant, avg_linear, avg_gravity):.2f} MSE")
        print()
        print("💡 Expected LLM performance:")
        print("  - Good model: MSE < 50 (sub-pixel accuracy)")
        print("  - Great model: MSE < 10 (near-perfect)")
        print("  - SOTA model: MSE < 1 (human-level)")
        print()
    
    print(f"✅ Results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
