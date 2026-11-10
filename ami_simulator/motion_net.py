"""Small causal policies for optical modulation control."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class _CausalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = functional.pad(values, (self.left_padding, 0))
        return functional.relu(self.norm(self.conv(values)))


class TinyCausalGearPolicy(nn.Module):
    """Predict AMI speed-gear logits from a short causal feature sequence.

    ``features`` has shape ``[batch, time, feature_dim]``.  The network is
    intentionally agnostic to the feature extractor: the first version can
    use IMU and event statistics, while a small event encoder can be added
    later without changing the controller interface.
    """

    def __init__(self, feature_dim: int = 24, hidden_dim: int = 32, num_gears: int = 5) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_gears < 2:
            raise ValueError("feature_dim and hidden_dim must be positive; num_gears must be at least two.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_gears = int(num_gears)
        self.temporal = nn.Sequential(
            _CausalConvBlock(self.feature_dim, self.hidden_dim, dilation=1),
            _CausalConvBlock(self.hidden_dim, self.hidden_dim, dilation=2),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_gears),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded = self.temporal(features.transpose(1, 2))
        return self.head(encoded[:, :, -1])

    def probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Return gear probabilities for monitoring or a hysteresis FSM."""
        return torch.softmax(self(features), dim=-1)


class CausalRPMPolicy(nn.Module):
    """Predict a bounded prism RPM from causal IMU/event-quality features."""

    def __init__(self, feature_dim: int = 14, hidden_dim: int = 32, min_rpm: float = 180.0, max_rpm: float = 1440.0) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or not bool(torch.isfinite(torch.tensor([min_rpm, max_rpm])).all()) or min_rpm < 0.0 or max_rpm <= min_rpm:
            raise ValueError("feature_dim, hidden_dim, and RPM bounds must be valid.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.temporal = nn.Sequential(
            _CausalConvBlock(self.feature_dim, self.hidden_dim, dilation=1),
            _CausalConvBlock(self.hidden_dim, self.hidden_dim, dilation=2),
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded = self.temporal(features.transpose(1, 2))
        normalized = torch.sigmoid(self.head(encoded[:, :, -1]).squeeze(-1))
        return self.min_rpm + (self.max_rpm - self.min_rpm) * normalized


class CausalFlowFusionRPMPolicy(nn.Module):
    """Causal LSTM policy fusing IMU, RGB-flow, and event-state features."""

    def __init__(self, feature_dim: int = 17, hidden_dim: int = 64, min_rpm: float = 180.0, max_rpm: float = 1440.0) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or not bool(torch.isfinite(torch.tensor([min_rpm, max_rpm])).all()) or min_rpm < 0.0 or max_rpm <= min_rpm:
            raise ValueError("feature_dim, hidden_dim, and RPM bounds must be valid.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_rpm = float(min_rpm)
        self.max_rpm = float(max_rpm)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.temporal = nn.LSTM(self.feature_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded, _ = self.temporal(self.input_norm(features))
        normalized = torch.sigmoid(self.head(encoded[:, -1]).squeeze(-1))
        return self.min_rpm + (self.max_rpm - self.min_rpm) * normalized


class CausalFlowFusionGearPolicy(nn.Module):
    """Causal LSTM classifier for a finite set of physically simulated gears."""

    def __init__(self, feature_dim: int = 17, hidden_dim: int = 64, num_gears: int = 5) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_gears < 2:
            raise ValueError("feature_dim and hidden_dim must be positive; num_gears must be at least two.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_gears = int(num_gears)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.temporal = nn.LSTM(self.feature_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_gears),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded, _ = self.temporal(self.input_norm(features))
        return self.head(encoded[:, -1])


class CausalFlowFusionQPolicy(CausalFlowFusionGearPolicy):
    """Predict one utility value for each candidate prism-speed action."""

    @property
    def num_actions(self) -> int:
        return self.num_gears


class CausalGatedFlowQPolicy(nn.Module):
    """Residual flow adapter around a causal IMU/event/phase Q policy."""

    def __init__(self, feature_dim: int = 19, hidden_dim: int = 64, num_gears: int = 5, *, base_feature_dim: int = 11, flow_start: int = 9, flow_dim: int = 8) -> None:
        super().__init__()
        if feature_dim != base_feature_dim + flow_dim or flow_start + flow_dim > feature_dim:
            raise ValueError("Gated flow layout must contain base features plus one contiguous flow block.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_gears = int(num_gears)
        self.base_feature_dim = int(base_feature_dim)
        self.flow_start = int(flow_start)
        self.flow_dim = int(flow_dim)
        self.base_policy = CausalFlowFusionQPolicy(self.base_feature_dim, hidden_dim, num_gears)
        self.flow_adapter = nn.Sequential(
            nn.Linear(self.flow_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.base_feature_dim),
        )
        self.gate = nn.Linear(self.feature_dim, self.base_feature_dim)
        nn.init.zeros_(self.flow_adapter[-1].weight)
        nn.init.zeros_(self.flow_adapter[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @property
    def num_actions(self) -> int:
        return self.num_gears

    def _split(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow = features[..., self.flow_start : self.flow_start + self.flow_dim]
        base = torch.cat((features[..., : self.flow_start], features[..., self.flow_start + self.flow_dim :]), dim=-1)
        return base, flow

    def gate_values(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        base, flow = self._split(features)
        fused = base + self.gate_values(features) * self.flow_adapter(flow)
        return self.base_policy(fused)


class CausalAnchorResidualPolicy(nn.Module):
    """Predict residual utility gains and switch confidence around an anchor.

    The model deliberately keeps the anchor action outside the output head.
    A caller uses the residual gains only when they clear a conservative
    threshold; otherwise the fixed anchor remains active.
    """

    def __init__(self, feature_dim: int = 20, hidden_dim: int = 64, residual_actions: int = 2) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim < 4 or residual_actions < 1:
            raise ValueError("feature_dim/hidden_dim/residual_actions must be positive.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_actions = int(residual_actions)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, residual_actions)
        self.switch_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded = self.encoder(features)
        pooled = encoded[:, -1]
        return self.delta_head(pooled), self.switch_head(pooled).squeeze(-1)


class CausalAnchorResidualGRUPolicy(CausalAnchorResidualPolicy):
    """Temporal student for residual optical modulation around 180 RPM."""

    def __init__(self, feature_dim: int = 20, hidden_dim: int = 64, residual_actions: int = 2) -> None:
        nn.Module.__init__(self)
        if feature_dim <= 0 or hidden_dim < 4 or residual_actions < 1:
            raise ValueError("feature_dim/hidden_dim/residual_actions must be positive.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_actions = int(residual_actions)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.temporal = nn.GRU(self.feature_dim, hidden_dim, num_layers=1, batch_first=True)
        self.encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.delta_head = nn.Linear(hidden_dim, residual_actions)
        self.switch_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded, _ = self.temporal(self.input_norm(features))
        pooled = self.encoder(encoded[:, -1])
        return self.delta_head(pooled), self.switch_head(pooled).squeeze(-1)


class CausalAnchorQGRUPolicy(nn.Module):
    """Predict finite-horizon utility for each optical-speed action.

    The anchor action (180 RPM) is kept as an explicit output column.  A
    caller can therefore apply an anchor-safe gate: only leave 180 RPM when
    the predicted advantage and confidence are sufficient.  ``features`` are
    causal state vectors and the output has shape ``[batch, num_actions]``.
    """

    def __init__(self, feature_dim: int = 47, hidden_dim: int = 64, num_actions: int = 3) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim < 4 or num_actions < 2:
            raise ValueError("feature_dim/hidden_dim must be positive and num_actions at least two.")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.temporal = nn.GRU(self.feature_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded, _ = self.temporal(self.input_norm(features))
        return self.q_head(self.encoder(encoded[:, -1]))


class CausalAnchorQualityResidualPolicy(CausalAnchorResidualPolicy):
    """Residual policy with an explicit event-efficiency risk prediction."""

    def __init__(self, feature_dim: int = 20, hidden_dim: int = 64, residual_actions: int = 2) -> None:
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, residual_actions=residual_actions)
        self.efficiency_head = nn.Linear(hidden_dim, residual_actions)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected features shaped [batch, time, {self.feature_dim}].")
        encoded = self.encoder(features)
        pooled = encoded[:, -1]
        return self.delta_head(pooled), self.switch_head(pooled).squeeze(-1), self.efficiency_head(pooled)
