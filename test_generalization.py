#!/usr/bin/env python3
"""
Comprehensive Generalization Tests for PhysicsLLMEngine

Tests 5 types of generalization:
1. Zero-shot scenarios (held-out types)
2. Out-of-distribution difficulty (extreme values)
3. Parameter extrapolation (mass, friction, etc.)
4. Compositional generalization (combined physics)
5. Physical law consistency (energy, momentum)
"""

import sys
import json
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.scenario_generator import generate_scenario, SCENARIO_TYPES
from src.physics.simulation import PhysicsSimulation


# ==============================================================================
# Test 1: Zero-Shot Scenarios
# ==============================================================================

def test_zero_shot_scenarios():
    """Test on held-out scenario types never seen during training."""
    
    held_out_types = [
        "pong",
        "bowling", 
        "ramp_roll",
        "angry_birds",
        "hourglass",
        "newtons_cradle"
    ]
    
    print("=" * 80)
    print("TEST 1: Zero-Shot Scenarios (Held-Out Types)")
    print("=" * 80)
    
    results = {}
    
    for scenario_type in held_out_types:
        print(f"\n📊 Testing: {scenario_type}")
        
        # Generate 3 scenes with different difficulties
        scenes = []
        for difficulty in [2, 3, 4]:
            seed = random.randint(10000000, 99999999)
            try:
                sim, metadata = generate_scenario(
                    seed=seed,
                    scenario_type=scenario_type,
                    difficulty=difficulty
                )
                
                # Run simulation for 50 frames
                frames = []
                for _ in range(50):
                    sim.step()
                    frames.append(sim.get_state())
                
                scenes.append({
                    "seed": seed,
                    "difficulty": difficulty,
                    "metadata": metadata,
                    "num_objects": len(sim.bodies),
                    "frames_captured": len(frames)
                })
                
                print(f"  ✅ Difficulty {difficulty}: {len(sim.bodies)} objects, {len(frames)} frames")
                
            except Exception as e:
                print(f"  ❌ Difficulty {difficulty}: {e}")
                scenes.append({"error": str(e)})
        
        results[scenario_type] = scenes
    
    return results


# ==============================================================================
# Test 2: Out-of-Distribution Difficulty
# ==============================================================================

def test_ood_difficulty():
    """Test extreme parameter ranges not seen in training."""
    
    print("\n" + "=" * 80)
    print("TEST 2: Out-of-Distribution Difficulty")
    print("=" * 80)
    
    # Training uses difficulty 1-5
    # Test difficulty 0 (very easy) and 6-10 (very hard)
    
    test_scenarios = ["billiards", "tower", "pendulum"]
    ood_difficulties = [0, 6, 7, 8, 9, 10]
    
    results = {}
    
    for scenario_type in test_scenarios:
        print(f"\n📊 Testing: {scenario_type}")
        results[scenario_type] = {}
        
        for difficulty in ood_difficulties:
            seed = random.randint(10000000, 99999999)
            try:
                sim, metadata = generate_scenario(
                    seed=seed,
                    scenario_type=scenario_type,
                    difficulty=difficulty
                )
                
                # Check scene properties
                num_objects = len(sim.bodies)
                
                # Run 30 frames
                for _ in range(30):
                    sim.step()
                
                results[scenario_type][difficulty] = {
                    "success": True,
                    "num_objects": num_objects,
                    "metadata": metadata
                }
                
                print(f"  ✅ Difficulty {difficulty}: {num_objects} objects")
                
            except Exception as e:
                print(f"  ❌ Difficulty {difficulty}: {e}")
                results[scenario_type][difficulty] = {"error": str(e)}
    
    return results


# ==============================================================================
# Test 3: Parameter Extrapolation
# ==============================================================================

def test_parameter_extrapolation():
    """Test physical parameters outside training distribution."""
    
    print("\n" + "=" * 80)
    print("TEST 3: Parameter Extrapolation")
    print("=" * 80)
    
    from src.physics.simulation import PhysicsSimulation
    from src.physics.objects import create_circle, create_rectangle
    
    results = {
        "extreme_gravity": [],
        "extreme_friction": [],
        "extreme_mass": []
    }
    
    # Test 1: Extreme gravity
    print("\n🌍 Testing extreme gravity values...")
    for gravity_y in [-5000, -2000, -100, 0, 500, 2000]:
        try:
            sim = PhysicsSimulation(gravity=(0, gravity_y))
            
            # Add test objects
            body1, shape1 = create_circle((300, 500), 20, mass=1.0, friction=0.5, elasticity=0.8)
            body2, shape2 = create_rectangle((500, 300), 50, 50, mass=2.0, friction=0.5, elasticity=0.6)
            
            sim.add_body(body1, shape1)
            sim.add_body(body2, shape2)
            
            # Simulate 50 frames
            for _ in range(50):
                sim.step()
            
            print(f"  ✅ Gravity Y={gravity_y}: simulation stable")
            results["extreme_gravity"].append({"gravity_y": gravity_y, "success": True})
            
        except Exception as e:
            print(f"  ❌ Gravity Y={gravity_y}: {e}")
            results["extreme_gravity"].append({"gravity_y": gravity_y, "error": str(e)})
    
    # Test 2: Extreme friction
    print("\n🧊 Testing extreme friction values...")
    for friction in [0.0, 0.001, 0.1, 0.5, 0.9, 0.99]:
        try:
            sim = PhysicsSimulation()
            body, shape = create_circle((400, 400), 30, mass=1.0, friction=friction, elasticity=0.5)
            body.velocity = (200, 0)
            sim.add_body(body, shape)
            
            for _ in range(50):
                sim.step()
            
            print(f"  ✅ Friction={friction}: simulation stable")
            results["extreme_friction"].append({"friction": friction, "success": True})
            
        except Exception as e:
            print(f"  ❌ Friction={friction}: {e}")
            results["extreme_friction"].append({"friction": friction, "error": str(e)})
    
    # Test 3: Extreme mass ratios
    print("\n⚖️  Testing extreme mass ratios...")
    for mass_ratio in [1, 10, 100, 1000, 10000]:
        try:
            sim = PhysicsSimulation()
            
            # Light object
            body1, shape1 = create_circle((200, 400), 15, mass=1.0, friction=0.3, elasticity=0.8)
            body1.velocity = (300, 0)
            sim.add_body(body1, shape1)
            
            # Heavy object
            body2, shape2 = create_circle((600, 400), 30, mass=mass_ratio, friction=0.3, elasticity=0.8)
            sim.add_body(body2, shape2)
            
            for _ in range(50):
                sim.step()
            
            print(f"  ✅ Mass ratio 1:{mass_ratio}: collision stable")
            results["extreme_mass"].append({"mass_ratio": mass_ratio, "success": True})
            
        except Exception as e:
            print(f"  ❌ Mass ratio 1:{mass_ratio}: {e}")
            results["extreme_mass"].append({"mass_ratio": mass_ratio, "error": str(e)})
    
    return results


# ==============================================================================
# Test 4: Compositional Generalization
# ==============================================================================

def test_compositional_generalization():
    """Test novel combinations of physics primitives."""
    
    print("\n" + "=" * 80)
    print("TEST 4: Compositional Generalization")
    print("=" * 80)
    
    # Create hybrid scenarios not in training
    
    results = {}
    
    # Hybrid 1: Pendulum + Projectile
    print("\n🔗 Testing: Pendulum + Projectile collision")
    try:
        sim1, _ = generate_scenario(seed=12345, scenario_type="pendulum", difficulty=3)
        sim2, _ = generate_scenario(seed=67890, scenario_type="projectile", difficulty=2)
        
        # Merge simulations (add projectile bodies to pendulum scene)
        for body in sim2.bodies[:3]:  # Add 3 projectiles
            sim1.space.add(body)
            sim1.bodies.append(body)
        
        for _ in range(50):
            sim1.step()
        
        print(f"  ✅ Hybrid scenario: {len(sim1.bodies)} total objects")
        results["pendulum_projectile"] = {"success": True, "num_objects": len(sim1.bodies)}
        
    except Exception as e:
        print(f"  ❌ Hybrid failed: {e}")
        results["pendulum_projectile"] = {"error": str(e)}
    
    # Hybrid 2: Tower + Bowling
    print("\n🎳 Testing: Tower + Bowling ball")
    try:
        sim_tower, _ = generate_scenario(seed=11111, scenario_type="tower", difficulty=4)
        sim_bowling, _ = generate_scenario(seed=22222, scenario_type="bowling", difficulty=2)
        
        # Add bowling ball to tower scene
        for body in sim_bowling.bodies[:1]:  # Just the ball
            body.velocity = (400, 0)  # Fast horizontal
            sim_tower.space.add(body)
            sim_tower.bodies.append(body)
        
        for _ in range(50):
            sim_tower.step()
        
        print(f"  ✅ Hybrid scenario: tower destruction by bowling ball")
        results["tower_bowling"] = {"success": True}
        
    except Exception as e:
        print(f"  ❌ Hybrid failed: {e}")
        results["tower_bowling"] = {"error": str(e)}
    
    return results


# ==============================================================================
# Test 5: Physical Law Consistency
# ==============================================================================

def test_physical_laws():
    """Test if simulations obey fundamental physics laws."""
    
    print("\n" + "=" * 80)
    print("TEST 5: Physical Law Consistency")
    print("=" * 80)
    
    from src.physics.objects import create_circle, create_rectangle
    
    results = {
        "energy_conservation": [],
        "momentum_conservation": []
    }
    
    # Test energy conservation in elastic collision
    print("\n⚡ Testing energy conservation (elastic collision)...")
    try:
        sim = PhysicsSimulation(gravity=(0, 0))  # No gravity
        
        # Two objects, elastic collision
        body1, shape1 = create_circle((200, 400), 20, mass=1.0, friction=0.0, elasticity=0.99)
        body1.velocity = (300, 0)
        
        body2, shape2 = create_circle((600, 400), 20, mass=1.0, friction=0.0, elasticity=0.99)
        body2.velocity = (0, 0)
        
        sim.add_body(body1, shape1)
        sim.add_body(body2, shape2)
        
        # Calculate initial kinetic energy
        ke_initial = 0.5 * body1.mass * (body1.velocity.length ** 2)
        
        # Simulate collision
        for _ in range(100):
            sim.step()
        
        # Calculate final kinetic energy
        ke_final = sum(0.5 * b.mass * (b.velocity.length ** 2) for b in sim.bodies)
        
        energy_loss_pct = abs(ke_final - ke_initial) / ke_initial * 100
        
        print(f"  Initial KE: {ke_initial:.2f}")
        print(f"  Final KE: {ke_final:.2f}")
        print(f"  Energy loss: {energy_loss_pct:.2f}%")
        
        if energy_loss_pct < 10:
            print(f"  ✅ Energy conserved (loss < 10%)")
            results["energy_conservation"].append({"success": True, "loss_pct": energy_loss_pct})
        else:
            print(f"  ⚠️  High energy loss: {energy_loss_pct:.2f}%")
            results["energy_conservation"].append({"success": False, "loss_pct": energy_loss_pct})
            
    except Exception as e:
        print(f"  ❌ Energy test failed: {e}")
        results["energy_conservation"].append({"error": str(e)})
    
    # Test momentum conservation
    print("\n💥 Testing momentum conservation...")
    try:
        sim = PhysicsSimulation(gravity=(0, 0))
        
        body1, shape1 = create_circle((200, 400), 25, mass=2.0, friction=0.0, elasticity=0.5)
        body1.velocity = (200, 0)
        
        body2, shape2 = create_circle((600, 400), 15, mass=1.0, friction=0.0, elasticity=0.5)
        body2.velocity = (-100, 0)
        
        sim.add_body(body1, shape1)
        sim.add_body(body2, shape2)
        
        # Initial momentum
        p_initial = sum(b.mass * b.velocity.x for b in sim.bodies)
        
        for _ in range(100):
            sim.step()
        
        # Final momentum
        p_final = sum(b.mass * b.velocity.x for b in sim.bodies)
        
        momentum_change_pct = abs(p_final - p_initial) / abs(p_initial) * 100 if p_initial != 0 else 0
        
        print(f"  Initial momentum: {p_initial:.2f}")
        print(f"  Final momentum: {p_final:.2f}")
        print(f"  Change: {momentum_change_pct:.2f}%")
        
        if momentum_change_pct < 5:
            print(f"  ✅ Momentum conserved (change < 5%)")
            results["momentum_conservation"].append({"success": True, "change_pct": momentum_change_pct})
        else:
            print(f"  ⚠️  Momentum change: {momentum_change_pct:.2f}%")
            results["momentum_conservation"].append({"success": False, "change_pct": momentum_change_pct})
            
    except Exception as e:
        print(f"  ❌ Momentum test failed: {e}")
        results["momentum_conservation"].append({"error": str(e)})
    
    return results


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    print("\n🧪 PhysicsLLMEngine Generalization Test Suite")
    print("=" * 80)
    print()
    
    all_results = {}
    
    # Run all tests
    all_results["zero_shot"] = test_zero_shot_scenarios()
    all_results["ood_difficulty"] = test_ood_difficulty()
    all_results["parameter_extrapolation"] = test_parameter_extrapolation()
    all_results["compositional"] = test_compositional_generalization()
    all_results["physical_laws"] = test_physical_laws()
    
    # Save results
    output_file = Path(__file__).parent / "generalization_test_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print(f"✅ All tests complete! Results saved to: {output_file}")
    print("=" * 80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"  Zero-shot scenarios: {len(all_results['zero_shot'])} tested")
    print(f"  OOD difficulties: {sum(len(v) for v in all_results['ood_difficulty'].values())} tested")
    print(f"  Parameter extrapolations: {sum(len(v) for v in all_results['parameter_extrapolation'].values())} tested")
    print(f"  Compositional tests: {len(all_results['compositional'])} tested")
    print(f"  Physical law tests: 2 laws tested")
    print()

if __name__ == "__main__":
    main()
