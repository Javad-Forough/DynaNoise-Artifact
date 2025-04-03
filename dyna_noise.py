import torch
import torch.nn.functional as F
import math
import numpy as np

class DynaNoise:
    def __init__(self, base_variance=0.1, lambda_scale=1.0, temperature=1.0):
        """
        :param base_variance: Base noise variance (sigma^2_0) for dynamic noise.
        :param lambda_scale:  Lambda (λ) factor that scales noise with sensitivity.
        :param temperature:   T for the optional softmax temperature smoothing.
        """
        self.base_variance = base_variance
        self.lambda_scale = lambda_scale
        self.temperature = temperature

    
    def sensitivity_score(self, logits):
        """
        Compute R(q) from the paper.
        R(q) = 1 - H(p) / log(k), 
        where p is the softmax probability and k is num_classes.
        """
        with torch.no_grad():
            # Convert logits -> probabilities
            probs = F.softmax(logits, dim=1)
            # Shannon entropy
            # sum over classes: -p_i log(p_i)
            eps = 1e-9
            entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)  # (batch,)
            # max entropy is log(k), where k = logits.shape[1]
            k = logits.shape[1]
            max_entropy = math.log(k + eps)
            R_q = 1.0 - (entropy / max_entropy)  # shape (batch,)
        return R_q  # shape (batch,)

    def inject_noise(self, logits):
        """
        Dynamically inject noise based on R(q).
        We compute sigma^2(q) = sigma^2_0 * (1 + λ * R(q)).
        Then sample from N(0, sigma^2(q)).
        """
        # R(q) for each sample in the batch
        R_q = self.sensitivity_score(logits)  # shape (batch,)
        # Compute per-sample noise variance
        sigma_sq = self.base_variance * (1.0 + self.lambda_scale * R_q)  # shape (batch,)
        # For each sample in the batch, draw noise from N(0, sigma_sq)
        # We'll produce an output shape (batch, num_classes)
        noise_list = []
        for i in range(logits.size(0)):
            # Each sample's std dev
            std_i = sigma_sq[i].sqrt().item()
            noise_i = torch.randn_like(logits[i]) * std_i  # shape (num_classes,)
            noise_list.append(noise_i.unsqueeze(0))
        noise_batch = torch.cat(noise_list, dim=0)  # shape (batch, num_classes)
        return logits + noise_batch

    def smooth_output(self, noisy_logits):
        """
        Probabilistic smoothing step.
        single pass with softmax temperature
            out = softmax(logits / T)
        """
        # Just apply temperature smoothing
        out_probs = F.softmax(noisy_logits / self.temperature, dim=1)
        return out_probs

    def forward(self, logits):
        """
        The main interface for DynaNoise:
         1) inject dynamic noise
         2) apply smoothing
        Return final probabilities.
        """
        noisy = self.inject_noise(logits)
        smoothed = self.smooth_output(noisy)
        return smoothed
