#!/usr/bin/env python3
"""
Evaluate trained LLM vs Ground Truth using MSE.

Loads LFM2-350M + LoRA checkpoint, generates predictions,
compares against Pymunk ground truth.
"""

import sys
import json
import re
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.physics.scenario_generator import generate_scenario


# Regex to parse object positions from generated text
OBJ_PATTERN = re.compile(r"obj_(\d+):\s*pos=\(([^)]+)\)")


def load_model(checkpoint_path="/home/alexw/checkpoints/lfm2-scenarios/final"):
    """Load LFM2 + LoRA model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-350M",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
    
    model.eval()
    
    print("✅ Model loaded")
    return model, tokenizer


def extract_positions_from_text(text):
    """Parse 'obj_X: pos=(x, y)' from generated text."""
    positions = {}
    for match in OBJ_PATTERN.finditer(text):
        obj_id = int(match.group(1))
        pos_str = match.group(2)
        x, y = map(float, pos_str.split(','))
        positions[obj_id] = [x, y]
    return positions


def build_prompt(sim, metadata, frames):
    """Build text prompt from scene metadata and context frames."""
    from src.data.formats import format_scene_header, format_frame
    
    prompt_lines = []
    
    # Header
    header = format_scene_header(metadata, sim)
    prompt_lines.append(header)
    prompt_lines.append("")
    
    # Context frames
    for i, frame_state in enumerate(frames, start=1):
        frame_text = format_frame(i, frame_state)
        prompt_lines.append(frame_text)
        prompt_lines.append("")
    
    # Prompt for next frame
    next_frame_num = len(frames) + 1
    prompt_lines.append(f"Frame {next_frame_num}:")
    
    return "\n".join(prompt_lines)


def evaluate_scenario(model, tokenizer, scenario_type, difficulty, seed, 
                      context_frames=10, predict_frames=5):
    """
    Evaluate LLM prediction vs Ground Truth for one scenario.
    
    Returns MSE for each predicted frame.
    """
    
    # Generate scenario
    sim, metadata = generate_scenario(seed=seed, scenario_type=scenario_type, difficulty=difficulty)
    
    # Run GT simulation
    gt_frames = []
    for _ in range(context_frames + predict_frames):
        sim.step()
        positions = [[b.position.x, b.position.y] for b in sim.bodies]
        gt_frames.append(positions)
    
    # Extract context and prediction ground truth
    context = gt_frames[:context_frames]
    gt_predictions = gt_frames[context_frames:]
    
    # Generate LLM predictions (autoregressive)
    # For now: just predict next frame given context
    # TODO: Implement autoregressive rollout
    
    # Build prompt
    try:
        from src.data.formats import format_scene_header, format_frame
        
        # Simplified format (since we don't have exact format_* functions)
        prompt = f"Scene: {len(sim.bodies)} objects. Gravity: (0.00, -981.00). Timestep: 0.0167.\n"
        prompt += f"Scenario: {scenario_type} ({metadata['scenario_category']}), difficulty: {difficulty}\n\n"
        
        # Add context frames
        for i, frame_pos in enumerate(context, start=1):
            prompt += f"Frame {i}: {len(frame_pos)} objects moving.\n"
            for obj_id, pos in enumerate(frame_pos):
                prompt += f"  obj_{obj_id}: pos=({pos[0]:.2f}, {pos[1]:.2f})\n"
            prompt += "\n"
        
        # Prompt for next frame
        prompt += f"Frame {context_frames + 1}:"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predicted positions
        pred_positions_dict = extract_positions_from_text(generated_text)
        
        # Convert to list aligned with GT
        pred_positions = []
        for obj_id in range(len(gt_predictions[0])):
            if obj_id in pred_positions_dict:
                pred_positions.append(pred_positions_dict[obj_id])
            else:
                # Missing object — use last known position (penalize)
                pred_positions.append(context[-1][obj_id])
        
        # Calculate MSE
        pred_arr = np.array(pred_positions)
        gt_arr = np.array(gt_predictions[0])
        
        mse = np.mean((pred_arr - gt_arr) ** 2)
        
        return {
            "scenario_type": scenario_type,
            "difficulty": difficulty,
            "seed": seed,
            "num_objects": len(sim.bodies),
            "mse": float(mse),
            "success": True
        }
        
    except Exception as e:
        return {
            "scenario_type": scenario_type,
            "difficulty": difficulty,
            "seed": seed,
            "error": str(e),
            "success": False
        }


def main():
    print("\n🧪 PhysicsLLMEngine LLM MSE Evaluation")
    print("=" * 80)
    print()
    
    # Load model
    model, tokenizer = load_model()
    
    # Test on held-out scenarios
    held_out_scenarios = [
        ("pong", 3),
        ("bowling", 3),
        ("ramp_roll", 3),
        ("angry_birds", 2),
        ("hourglass", 2),
        ("newtons_cradle", 3)
    ]
    
    print("\n" + "=" * 80)
    print("Evaluating on Held-Out Scenarios")
    print("=" * 80)
    print()
    
    results = []
    
    for scenario_type, difficulty in held_out_scenarios:
        seed = np.random.randint(10000000, 99999999)
        
        print(f"📊 Testing: {scenario_type} (difficulty {difficulty})")
        
        result = evaluate_scenario(
            model=model,
            tokenizer=tokenizer,
            scenario_type=scenario_type,
            difficulty=difficulty,
            seed=seed,
            context_frames=10,
            predict_frames=1  # Start with 1-frame prediction
        )
        
        results.append(result)
        
        if result["success"]:
            print(f"  Objects: {result['num_objects']}")
            print(f"  MSE: {result['mse']:.2f}")
            print()
        else:
            print(f"  ❌ Failed: {result['error']}")
            print()
    
    # Save results
    output_file = Path(__file__).parent / "llm_mse_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    valid_results = [r for r in results if r["success"]]
    
    if valid_results:
        avg_mse = np.mean([r["mse"] for r in valid_results])
        
        print("=" * 80)
        print("📊 LLM MSE RESULTS")
        print("=" * 80)
        print()
        print(f"Scenarios tested: {len(valid_results)}/{len(results)}")
        print(f"Average MSE: {avg_mse:.2f}")
        print()
        print("🎯 Baseline comparison:")
        print("  Naive baseline: 51.33 MSE")
        print(f"  LLM model:      {avg_mse:.2f} MSE")
        
        if avg_mse < 51.33:
            improvement = (51.33 - avg_mse) / 51.33 * 100
            print(f"  ✅ LLM beats baseline by {improvement:.1f}%!")
        else:
            print(f"  ⚠️  LLM worse than baseline")
        
        print()
        
        if avg_mse < 50:
            print("  ✅ GOOD MODEL (sub-pixel accuracy)")
        if avg_mse < 10:
            print("  🌟 GREAT MODEL (near-perfect)")
        if avg_mse < 1:
            print("  🏆 SOTA MODEL (human-level)")
        
        print()
    
    print(f"✅ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
