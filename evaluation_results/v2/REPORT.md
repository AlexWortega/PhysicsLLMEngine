# PhysicsLLMEngine v2.0 — Experiment Report

**Date:** 2026-02-07  
**Model:** LiquidAI/LFM2-350M + LoRA  
**New Scenarios:** 5

---

## New Scenarios Added

### 1. **particle_explosion**
- **Category:** Complex
- **Description:** 50-100 small particles explode radially from center
- **Difficulty scaling:** 50 particles (easy) → 100 particles (hard)
- **Key physics:** Momentum conservation, collision cascades, chaotic dynamics

### 2. **Enhanced gravity_well**
- **Category:** Complex  
- **Description:** 30+ objects fall toward gravitational center
- **Improvement:** Increased object count (was 10-26, now scales higher)
- **Key physics:** Radial gravity simulation, centripetal dynamics

### 3. **chain_reaction**
- **Category:** Complex
- **Description:** 100-200 dominos in spiral pattern + trigger ball
- **Difficulty scaling:** 100 dominos (easy) → 200 dominos (hard)
- **Key physics:** Sequential toppling, propagation delay, angular stability

### 4. **fluid_sim**
- **Category:** Complex
- **Description:** 60-120 particles simulating fluid flow through barriers
- **Key physics:** Pseudo-fluid dynamics, particle-particle interaction, flow through obstacles

### 5. **solar_system**
- **Category:** Complex
- **Description:** 5-9 planets orbiting central static sun
- **Key physics:** Orbital mechanics, centripetal force, stable orbits

---

## OOD Evaluation Plan

**Held-out scenarios for zero-shot testing:**
- particle_explosion
- chain_reaction
- fluid_sim
- solar_system

**Metrics:**
- MSE (Mean Squared Error) — position prediction accuracy
- Rollout stability — how many frames before divergence
- Visual similarity — SSIM between GT and predicted trajectories

**Visualization:**
- Green trajectories = Ground Truth (Pymunk simulation)
- Blue trajectories = LLM autoregressive prediction

---

## Next Steps

1. Generate training data for new scenarios
2. Run OOD evaluation on held-out scenarios
3. Render comparison GIFs (green/blue)
4. Analyze failure modes and edge cases

---

**Status:** ✅ Scenarios implemented, ready for evaluation
