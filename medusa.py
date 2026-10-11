"""Neural network architecture used by the MEDUSA estimator.

The estimator, training objective, and hyperparameters live in ``models.py``.
This module contains only the reusable PyTorch network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MEDUSA(nn.Module):
    """Predict factual and counterfactual outcomes plus treatment propensities.

    Covariates are encoded into a latent representation. A propensity head
    predicts the treatment arm, while a shared outcome network is evaluated
    for the observed treatment and for every possible arm. Treatment arms are
    represented by their raw binary bit vectors.

    The forward pass returns the factual outcome logit, all treatment outcome
    logits, and propensity logits, in that order.
    """

    def __init__(self, input_size, latent_dim, num_treatments, treatment_bits):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_treatments = num_treatments
        self.treatment_bits = treatment_bits

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.propensity_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_treatments),
        )
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + treatment_bits, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with Xavier-uniform weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _indices_to_bits(self, indices):
        powers = 2 ** torch.arange(self.treatment_bits, device=indices.device)
        return ((indices.view(-1, 1).long() // powers.view(1, -1)) % 2).float()

    def forward(self, x, treatment_bits):
        """Return factual logits, all-arm logits, and propensity logits.

        Args:
            x: Covariate tensor with shape ``(batch_size, input_size)``.
            treatment_bits: Observed treatment codes with shape
                ``(batch_size, treatment_bits)``.

        Returns:
            A tuple containing the factual outcome logits with shape
            ``(batch_size, 1)``, all treatment outcome logits with shape
            ``(batch_size, num_treatments)``, and treatment propensity logits
            with shape ``(batch_size, num_treatments)``.
        """
        representation = self.encoder(x)
        treatment_bits = treatment_bits.float()
        if treatment_bits.ndim == 1:
            treatment_bits = treatment_bits.view(1, -1)

        factual_input = torch.cat([representation, treatment_bits], dim=1)
        factual_logit = self.shared_net(factual_input)

        treatment_indices = torch.arange(self.num_treatments, device=x.device)
        all_treatment_bits = self._indices_to_bits(treatment_indices)
        representation_by_treatment = representation.unsqueeze(1).expand(
            -1, self.num_treatments, -1
        )
        treatment_bits_by_row = all_treatment_bits.unsqueeze(0).expand(
            x.size(0), -1, -1
        )
        all_inputs = torch.cat(
            [representation_by_treatment, treatment_bits_by_row], dim=-1
        )
        all_logits = self.shared_net(all_inputs).squeeze(-1)
        propensity_logits = self.propensity_head(representation)

        return factual_logit, all_logits, propensity_logits
