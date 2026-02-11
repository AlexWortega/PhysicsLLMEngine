#!/usr/bin/env python3
"""
Visualize MSE baseline results - how error accumulates over frames.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('mse_baseline_results.json') as f:
    results = json.load(f)

# Extract MSE per frame for each scenario
scenarios = []
for r in results:
    if 'baselines' not in r:
        continue
    
    scenario_name = f"{r['scenario_type']} (d={r['difficulty']})"
    frames = []
    constant_mse = []
    linear_mse = []
    gravity_mse = []
    
    for frame_key in sorted(r['baselines'].keys(), key=lambda x: int(x.split('_')[1])):
        frame_num = int(frame_key.split('_')[1])
        frames.append(frame_num)
        constant_mse.append(r['baselines'][frame_key]['constant_mse'])
        linear_mse.append(r['baselines'][frame_key]['linear_mse'])
        gravity_mse.append(r['baselines'][frame_key]['gravity_only_mse'])
    
    scenarios.append({
        'name': scenario_name,
        'frames': frames,
        'constant': constant_mse,
        'linear': linear_mse,
        'gravity': gravity_mse
    })

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, scenario in enumerate(scenarios):
    ax = axes[idx]
    
    ax.plot(scenario['frames'], scenario['constant'], 'o-', label='Constant', color='red', alpha=0.7)
    ax.plot(scenario['frames'], scenario['linear'], 's-', label='Linear', color='blue', alpha=0.7)
    ax.plot(scenario['frames'], scenario['gravity'], '^-', label='Gravity Only', color='green', alpha=0.7)
    
    ax.set_title(scenario['name'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Prediction Frame')
    ax.set_ylabel('MSE (pixels²)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mse_baselines_visualization.png', dpi=150)
print("✅ Saved: mse_baselines_visualization.png")

# Summary plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

avg_constant = np.mean([s['constant'] for s in scenarios], axis=0)
avg_linear = np.mean([s['linear'] for s in scenarios], axis=0)
avg_gravity = np.mean([s['gravity'] for s in scenarios], axis=0)

frames = scenarios[0]['frames']

ax2.plot(frames, avg_constant, 'o-', label='Constant Position', color='red', linewidth=2, markersize=8)
ax2.plot(frames, avg_linear, 's-', label='Linear Extrapolation', color='blue', linewidth=2, markersize=8)
ax2.plot(frames, avg_gravity, '^-', label='Gravity Only', color='green', linewidth=2, markersize=8)

ax2.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Good Model Target (MSE<50)')
ax2.axhline(y=10, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Great Model Target (MSE<10)')

ax2.set_title('Average MSE Across All Held-Out Scenarios', fontsize=14, fontweight='bold')
ax2.set_xlabel('Prediction Frame', fontsize=12)
ax2.set_ylabel('Mean Squared Error (pixels²)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mse_baselines_average.png', dpi=150)
print("✅ Saved: mse_baselines_average.png")

# Print summary table
print("\n" + "=" * 80)
print("📊 MSE SUMMARY TABLE")
print("=" * 80)
print()
print("| Scenario | Objects | Constant | Linear | Gravity | Best |")
print("|----------|---------|----------|--------|---------|------|")

for r in results:
    if 'average_mse' not in r:
        continue
    
    name = r['scenario_type']
    objs = r['num_objects']
    const = r['average_mse']['constant']
    lin = r['average_mse']['linear']
    grav = r['average_mse']['gravity_only']
    
    best = min(const, lin, grav)
    best_name = 'const' if best == const else ('linear' if best == lin else 'gravity')
    
    print(f"| {name:12} | {objs:7} | {const:8.2f} | {lin:6.2f} | {grav:7.2f} | {best_name} ({best:.2f}) |")

print()
print("=" * 80)
