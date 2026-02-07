"""
Physics-informed auxiliary loss for fine-tuning.

Implements energy and momentum conservation penalties as auxiliary loss
to improve physics prediction accuracy during fine-tuning.

Research Pattern 3: Physics auxiliary loss with weights 0.01-0.1 to
avoid destabilizing the main language modeling loss.
"""

import re
from typing import Optional, Tuple, List
import torch
import torch.nn as nn


class PhysicsAuxiliaryLoss(nn.Module):
    """
    Physics-informed auxiliary loss for energy/momentum conservation.

    Penalizes violations of physical conservation laws:
    - Energy: penalizes energy GAIN (entropy should not decrease)
    - Momentum: penalizes changes in total momentum (should be conserved)

    Args:
        energy_weight: Weight for energy conservation loss (default 0.01)
        momentum_weight: Weight for momentum conservation loss (default 0.01)
    """

    def __init__(
        self,
        energy_weight: float = 0.01,
        momentum_weight: float = 0.01,
    ):
        super().__init__()
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight

    def compute_kinetic_energy(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute kinetic energy for each batch item.

        KE = 0.5 * m * |v|^2 summed over all objects.

        Args:
            velocities: Tensor of shape (batch, num_objects, 2) for 2D velocities
            masses: Tensor of shape (batch, num_objects) for object masses

        Returns:
            Tensor of shape (batch,) with total KE per batch item
        """
        # |v|^2 = vx^2 + vy^2
        v_squared = (velocities ** 2).sum(dim=-1)  # (batch, num_objects)

        # KE = 0.5 * m * v^2
        ke_per_object = 0.5 * masses * v_squared  # (batch, num_objects)

        # Sum over objects
        total_ke = ke_per_object.sum(dim=-1)  # (batch,)

        return total_ke

    def compute_momentum(
        self,
        velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total momentum vector for each batch item.

        p = m * v summed over all objects.

        Args:
            velocities: Tensor of shape (batch, num_objects, 2)
            masses: Tensor of shape (batch, num_objects)

        Returns:
            Tensor of shape (batch, 2) with total momentum vector per batch item
        """
        # Expand masses for broadcasting: (batch, num_objects) -> (batch, num_objects, 1)
        masses_expanded = masses.unsqueeze(-1)

        # p = m * v for each object
        momentum_per_object = masses_expanded * velocities  # (batch, num_objects, 2)

        # Sum over objects
        total_momentum = momentum_per_object.sum(dim=1)  # (batch, 2)

        return total_momentum

    def forward(
        self,
        pred_velocities: torch.Tensor,
        true_velocities: torch.Tensor,
        masses: torch.Tensor,
        prev_ke: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute physics auxiliary loss.

        Penalizes:
        1. Energy GAIN (pred_ke > true_ke): systems should dissipate, not gain energy
        2. Momentum change: total momentum should be conserved

        Args:
            pred_velocities: Predicted velocities (batch, num_objects, 2)
            true_velocities: Ground truth velocities (batch, num_objects, 2)
            masses: Object masses (batch, num_objects)
            prev_ke: Previous frame kinetic energy for additional constraint (optional)

        Returns:
            Scalar loss value (weighted sum of energy and momentum losses)
        """
        # Compute kinetic energies
        pred_ke = self.compute_kinetic_energy(pred_velocities, masses)
        true_ke = self.compute_kinetic_energy(true_velocities, masses)

        # Energy loss: only penalize energy GAIN (pred_ke > true_ke)
        # Using ReLU: max(0, pred_ke - true_ke)
        energy_gain = torch.relu(pred_ke - true_ke)
        energy_loss = energy_gain.mean()

        # Compute momenta
        pred_momentum = self.compute_momentum(pred_velocities, masses)
        true_momentum = self.compute_momentum(true_velocities, masses)

        # Momentum loss: L2 distance of total momentum vectors
        momentum_diff = pred_momentum - true_momentum  # (batch, 2)
        momentum_loss = (momentum_diff ** 2).sum(dim=-1).mean()

        # Weighted sum
        total_loss = (
            self.energy_weight * energy_loss +
            self.momentum_weight * momentum_loss
        )

        return total_loss


def extract_physics_from_text(
    text: str,
    num_objects: int,
) -> Optional[torch.Tensor]:
    """
    Extract velocity values from physics frame text.

    Parses text format like:
    "obj_0: pos=(1.2345, 6.7890), vel=(0.1234, -0.5678)"

    Args:
        text: Frame text containing object velocities
        num_objects: Expected number of objects

    Returns:
        Tensor of shape (num_objects, 2) with velocities, or None if parsing fails
    """
    # Pattern to match velocity components
    # Matches: vel=(X.XXXX, Y.YYYY) or vel=(-X.XXXX, -Y.YYYY)
    vel_pattern = r"vel=\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)"

    matches = re.findall(vel_pattern, text)

    if len(matches) < num_objects:
        return None

    try:
        velocities = []
        for i in range(num_objects):
            vx = float(matches[i][0])
            vy = float(matches[i][1])
            velocities.append([vx, vy])

        return torch.tensor(velocities, dtype=torch.float32)
    except (ValueError, IndexError):
        return None


def extract_masses_from_header(
    header_text: str,
    num_objects: int,
    default_mass: float = 1.0,
) -> torch.Tensor:
    """
    Extract object masses from scene header text.

    If masses are not found, returns uniform mass for all objects.

    Args:
        header_text: Scene header text
        num_objects: Number of objects
        default_mass: Default mass if not found (default 1.0)

    Returns:
        Tensor of shape (num_objects,) with masses
    """
    # Pattern to match mass values (if present in header)
    # Format depends on scene header structure
    mass_pattern = r"mass[=:]?\s*(-?[\d.]+)"

    matches = re.findall(mass_pattern, header_text, re.IGNORECASE)

    if len(matches) >= num_objects:
        try:
            masses = [float(matches[i]) for i in range(num_objects)]
            return torch.tensor(masses, dtype=torch.float32)
        except ValueError:
            pass

    # Return uniform masses
    return torch.full((num_objects,), default_mass, dtype=torch.float32)


def compute_physics_loss(
    pred_text: str,
    true_text: str,
    masses: torch.Tensor,
    energy_weight: float = 0.01,
    momentum_weight: float = 0.01,
) -> float:
    """
    Compute physics loss from text predictions.

    Bridges text output to numerical physics loss computation.

    Args:
        pred_text: Predicted frame text
        true_text: Ground truth frame text
        masses: Object masses tensor of shape (num_objects,)
        energy_weight: Weight for energy loss
        momentum_weight: Weight for momentum loss

    Returns:
        Loss value as float, or 0.0 if extraction fails
    """
    num_objects = masses.shape[0]

    # Extract velocities from text
    pred_velocities = extract_physics_from_text(pred_text, num_objects)
    true_velocities = extract_physics_from_text(true_text, num_objects)

    if pred_velocities is None or true_velocities is None:
        return 0.0

    # Add batch dimension
    pred_velocities = pred_velocities.unsqueeze(0)  # (1, num_objects, 2)
    true_velocities = true_velocities.unsqueeze(0)
    masses_batch = masses.unsqueeze(0)  # (1, num_objects)

    # Compute loss
    loss_fn = PhysicsAuxiliaryLoss(
        energy_weight=energy_weight,
        momentum_weight=momentum_weight,
    )

    loss = loss_fn(pred_velocities, true_velocities, masses_batch)

    return loss.item()
