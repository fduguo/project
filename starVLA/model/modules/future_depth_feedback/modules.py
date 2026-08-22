from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.future_vit import CrossAttnFutureHead


class FutureDepthPredictor(nn.Module):
    """Predict structured future-depth tokens from detached action queries."""

    def __init__(
        self,
        hidden_size: int,
        depth_feature_dim: int,
        num_views: int,
        target_grid_size: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.depth_feature_dim = int(depth_feature_dim)
        self.num_views = int(num_views)
        self.target_grid_size = int(target_grid_size)
        if min(self.hidden_size, self.depth_feature_dim, self.num_views, self.target_grid_size) <= 0:
            raise ValueError("Future-depth dimensions must all be positive.")

        self.tokens_per_view = self.target_grid_size**2
        self.target_tokens = self.num_views * self.tokens_per_view
        self.decoder = CrossAttnFutureHead(
            hidden_size=self.hidden_size,
            target_tokens=self.target_tokens,
            num_layers_attn=int(num_layers),
            num_heads=int(num_heads),
            ffn_mult=int(ffn_mult),
            use_positional_embedding=True,
            num_views=self.num_views,
            tokens_per_view=self.tokens_per_view,
            use_self_attention=True,
            use_output_layernorm=True,
        )
        self.to_depth_feature = nn.Linear(self.hidden_size, self.depth_feature_dim)

    def forward(
        self,
        action_queries: torch.Tensor,
        *,
        detach_input: bool = True,
    ) -> torch.Tensor:
        if action_queries.ndim != 3 or action_queries.shape[-1] != self.hidden_size:
            raise ValueError(
                "Expected action_queries shaped [B, H, hidden_size], got "
                f"{tuple(action_queries.shape)}."
            )
        # Legacy callers keep the original detached behavior. Progressive JEPA
        # callers pass detach_input=False after applying an explicit gradient
        # scaler to make the VLM gradient strength configurable.
        predictor_input = action_queries.detach() if detach_input else action_queries
        decoded = self.decoder(predictor_input)
        return self.to_depth_feature(decoded)


class FutureDepthFeedback(nn.Module):
    """Read predicted future depth and add a gated residual to action queries."""

    def __init__(
        self,
        hidden_size: int,
        depth_feature_dim: int,
        num_heads: int = 8,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.depth_feature_dim = int(depth_feature_dim)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.depth_norm = nn.LayerNorm(self.depth_feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=int(num_heads),
            kdim=self.depth_feature_dim,
            vdim=self.depth_feature_dim,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def gate_value(self) -> torch.Tensor:
        return torch.tanh(self.gate)

    def forward(
        self,
        action_queries: torch.Tensor,
        predicted_depth: torch.Tensor,
    ) -> torch.Tensor:
        if action_queries.ndim != 3 or action_queries.shape[-1] != self.hidden_size:
            raise ValueError(
                "Expected action_queries shaped [B, H, hidden_size], got "
                f"{tuple(action_queries.shape)}."
            )
        if predicted_depth.ndim != 3 or predicted_depth.shape[-1] != self.depth_feature_dim:
            raise ValueError(
                "Expected predicted_depth shaped [B, N, depth_feature_dim], got "
                f"{tuple(predicted_depth.shape)}."
            )
        if action_queries.shape[0] != predicted_depth.shape[0]:
            raise ValueError("Action and future-depth batches must have the same size.")

        # The residual identity remains connected to the VLM. Only the branch query
        # is detached, so action loss can train the branch without leaking an extra
        # gradient path into the VLM.
        branch_query = self.query_norm(action_queries.detach())
        depth_kv = self.depth_norm(predicted_depth)
        delta = self.cross_attention(
            branch_query,
            depth_kv,
            depth_kv,
            need_weights=False,
        )[0]
        return action_queries + self.gate_value.to(dtype=delta.dtype) * delta


class ConditionalFutureDepthFeedback(nn.Module):
    """Task-conditioned feature routing and action-step feedback gating.

    The module keeps the old feedback parameter names (`query_norm`,
    `depth_norm`, `cross_attention`, and `gate`) so an existing global-gate
    checkpoint can initialize the shared path with `strict=False`.

    Routing has two levels:
      * sample-conditioned FiLM selects useful depth-feature channels;
      * action-token-conditioned gates decide when/how strongly to inject.

    All router inputs are detached. The ordinary residual identity remains
    connected to the VLM, while no extra router gradient path enters it.
    """

    def __init__(
        self,
        hidden_size: int,
        depth_feature_dim: int,
        num_heads: int = 8,
        gate_init: float = 0.0,
        router_hidden_dim: int = 128,
        gate_max: float = 1.0,
        film_scale: float = 0.1,
        near_zero_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.depth_feature_dim = int(depth_feature_dim)
        self.router_hidden_dim = int(router_hidden_dim)
        self.gate_max = float(gate_max)
        self.film_scale = float(film_scale)
        self.near_zero_threshold = float(near_zero_threshold)
        if min(self.hidden_size, self.depth_feature_dim, self.router_hidden_dim) <= 0:
            raise ValueError("Conditional routing dimensions must be positive.")
        if self.gate_max <= 0.0:
            raise ValueError("gate_max must be positive.")
        if self.film_scale < 0.0:
            raise ValueError("film_scale must be non-negative.")

        # These four names intentionally match FutureDepthFeedback.
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.depth_norm = nn.LayerNorm(self.depth_feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=int(num_heads),
            kdim=self.depth_feature_dim,
            vdim=self.depth_feature_dim,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        self.context_norm = nn.LayerNorm(self.hidden_size)
        self.context_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.router_hidden_dim),
            nn.SiLU(),
        )
        self.film_gamma = nn.Linear(self.router_hidden_dim, self.depth_feature_dim)
        self.film_beta = nn.Linear(self.router_hidden_dim, self.depth_feature_dim)
        self.step_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.router_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.router_hidden_dim, 1),
        )

        # Identity-preserving initialization:
        # gamma=1, beta=0, and conditional gate offset=0.
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        nn.init.zeros_(self.step_gate[-1].weight)
        nn.init.zeros_(self.step_gate[-1].bias)

        self._last_gate: torch.Tensor | None = None
        self._last_gamma: torch.Tensor | None = None
        self._last_beta: torch.Tensor | None = None
        self._last_delta: torch.Tensor | None = None
        self._last_injected: torch.Tensor | None = None
        self._last_query_reference: torch.Tensor | None = None

    @property
    def gate_value(self) -> torch.Tensor:
        """Legacy-compatible global gate value used by old logging code."""

        return self.gate_max * torch.tanh(self.gate)

    def _route_depth(
        self,
        predicted_depth: torch.Tensor,
        sample_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_features = self.context_router(sample_context)
        gamma = 1.0 + self.film_scale * torch.tanh(self.film_gamma(context_features))
        beta = self.film_scale * torch.tanh(self.film_beta(context_features))
        normalized_depth = self.depth_norm(predicted_depth)
        routed_depth = (
            gamma.unsqueeze(1).to(dtype=normalized_depth.dtype) * normalized_depth
            + beta.unsqueeze(1).to(dtype=normalized_depth.dtype)
        )
        return routed_depth, gamma, beta

    def _conditional_gate(
        self,
        branch_query: torch.Tensor,
        sample_context: torch.Tensor,
    ) -> torch.Tensor:
        expanded_context = sample_context.unsqueeze(1).expand(
            -1,
            branch_query.shape[1],
            -1,
        )
        conditional_offset = self.step_gate(
            torch.cat([branch_query, expanded_context], dim=-1)
        )
        raw_gate = self.gate.to(dtype=conditional_offset.dtype) + conditional_offset
        return self.gate_max * torch.tanh(raw_gate)

    def forward(
        self,
        action_queries: torch.Tensor,
        predicted_depth: torch.Tensor,
    ) -> torch.Tensor:
        if action_queries.ndim != 3 or action_queries.shape[-1] != self.hidden_size:
            raise ValueError(
                "Expected action_queries shaped [B, H, hidden_size], got "
                f"{tuple(action_queries.shape)}."
            )
        if predicted_depth.ndim != 3 or predicted_depth.shape[-1] != self.depth_feature_dim:
            raise ValueError(
                "Expected predicted_depth shaped [B, N, depth_feature_dim], got "
                f"{tuple(predicted_depth.shape)}."
            )
        if action_queries.shape[0] != predicted_depth.shape[0]:
            raise ValueError("Action and future-depth batches must have the same size.")

        detached_queries = action_queries.detach()
        branch_query = self.query_norm(detached_queries)
        sample_context = self.context_norm(detached_queries.mean(dim=1))
        routed_depth, gamma, beta = self._route_depth(predicted_depth, sample_context)
        delta = self.cross_attention(
            branch_query,
            routed_depth,
            routed_depth,
            need_weights=False,
        )[0]
        gate = self._conditional_gate(branch_query, sample_context).to(dtype=delta.dtype)
        injected = gate * delta

        # Keep graph-connected values for optional router regularization. Logging
        # reads detached copies through routing_metrics().
        self._last_gate = gate
        self._last_gamma = gamma
        self._last_beta = beta
        self._last_delta = delta
        self._last_injected = injected
        self._last_query_reference = detached_queries
        return action_queries + injected

    def routing_regularization(
        self,
        *,
        gate_l1_weight: float = 0.0,
        film_weight: float = 0.0,
    ) -> torch.Tensor:
        if self._last_gate is None:
            return self.gate.new_zeros(())
        gate_penalty = self._last_gate.float().abs().mean()
        film_penalty = (
            (self._last_gamma.float() - 1.0).square().mean()
            + self._last_beta.float().square().mean()
        )
        return float(gate_l1_weight) * gate_penalty + float(film_weight) * film_penalty

    def routing_parameter_count(self) -> int:
        routing_modules = (
            self.context_norm,
            self.context_router,
            self.film_gamma,
            self.film_beta,
            self.step_gate,
        )
        return sum(
            parameter.numel()
            for module in routing_modules
            for parameter in module.parameters()
        )

    @torch.no_grad()
    def routing_metrics(self) -> dict[str, torch.Tensor]:
        if self._last_gate is None:
            zero = self.gate.detach().new_zeros(())
            return {
                "future_depth_gate_mean": zero,
                "future_depth_gate_abs_mean": zero,
                "future_depth_gate_std": zero,
                "future_depth_gate_min": zero,
                "future_depth_gate_max": zero,
                "future_depth_gate_near_zero_fraction": zero,
                "future_depth_gate_positive_fraction": zero,
                "future_depth_film_gamma_abs_deviation": zero,
                "future_depth_film_beta_abs_mean": zero,
                "future_depth_feedback_delta_abs_mean": zero,
                "future_depth_effective_feedback_ratio": zero,
                "future_depth_global_gate_raw": zero,
                "future_depth_step_gate_output_weight_abs_mean": zero,
                "future_depth_film_gamma_weight_abs_mean": zero,
                "future_depth_film_beta_weight_abs_mean": zero,
            }

        gate = self._last_gate.detach().float()
        gamma = self._last_gamma.detach().float()
        beta = self._last_beta.detach().float()
        delta = self._last_delta.detach().float()
        injected = self._last_injected.detach().float()
        query = self._last_query_reference.detach().float()
        metrics = {
            "future_depth_gate_mean": gate.mean(),
            "future_depth_gate_abs_mean": gate.abs().mean(),
            "future_depth_gate_std": gate.std(unbiased=False),
            "future_depth_gate_min": gate.min(),
            "future_depth_gate_max": gate.max(),
            "future_depth_gate_near_zero_fraction": (
                gate.abs() < self.near_zero_threshold
            ).float().mean(),
            "future_depth_gate_positive_fraction": (gate > 0.0).float().mean(),
            "future_depth_film_gamma_abs_deviation": (gamma - 1.0).abs().mean(),
            "future_depth_film_beta_abs_mean": beta.abs().mean(),
            "future_depth_feedback_delta_abs_mean": delta.abs().mean(),
            "future_depth_effective_feedback_ratio": (
                injected.abs().mean() / query.abs().mean().clamp_min(1e-6)
            ),
            "future_depth_global_gate_raw": self.gate.detach().float(),
            "future_depth_step_gate_output_weight_abs_mean": (
                self.step_gate[-1].weight.detach().float().abs().mean()
            ),
            "future_depth_film_gamma_weight_abs_mean": (
                self.film_gamma.weight.detach().float().abs().mean()
            ),
            "future_depth_film_beta_weight_abs_mean": (
                self.film_beta.weight.detach().float().abs().mean()
            ),
        }
        for step_index in range(gate.shape[1]):
            metrics[f"future_depth_gate_step_{step_index}_mean"] = gate[:, step_index].mean()
            metrics[f"future_depth_gate_step_{step_index}_abs_mean"] = (
                gate[:, step_index].abs().mean()
            )
        return metrics
