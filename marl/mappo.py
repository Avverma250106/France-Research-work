"""Continuous-action MAPPO with independent actors and a centralized critic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(frozen=True)
class MAPPOConfig:
    hidden_size: int = 128
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    rollout_length: int = 128
    minibatch_size: int = 64
    ppo_epochs: int = 4
    max_gradient_norm: float = 0.5
    reward_mode: str = "individual"
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.actor_learning_rate <= 0 or self.critic_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1].")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1).")
        if self.entropy_coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("Loss coefficients must be non-negative.")
        if self.rollout_length <= 0 or self.minibatch_size <= 0 or self.ppo_epochs <= 0:
            raise ValueError("Rollout length, minibatch size, and PPO epochs must be positive.")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive.")
        if self.reward_mode not in {"individual", "team"}:
            raise ValueError("reward_mode must be 'individual' or 'team'.")


class _Actor(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.log_standard_deviation = nn.Parameter(torch.zeros(action_size))

    def distribution(self, observation: Tensor) -> Normal:
        mean = self.network(observation)
        log_std = self.log_standard_deviation.clamp(-5.0, 2.0)
        return Normal(mean, log_std.exp().expand_as(mean))


class MAPPOPolicy(nn.Module):
    """One local actor per agent plus a centralized per-agent value critic."""

    def __init__(
        self,
        num_agents: int,
        observation_size: int,
        action_size: int = 2,
        hidden_size: int = 128,
    ):
        super().__init__()
        if min(num_agents, observation_size, action_size, hidden_size) <= 0:
            raise ValueError("All MAPPO dimensions must be positive.")
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.action_size = action_size
        self.actors = nn.ModuleList(
            _Actor(observation_size, action_size, hidden_size)
            for _ in range(num_agents)
        )
        self.critic = nn.Sequential(
            nn.Linear(num_agents * observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_agents),
        )

    def value(self, global_state: Tensor) -> Tensor:
        return self.critic(global_state)

    def act(
        self,
        local_observations: np.ndarray | Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return [0, 1] energy fractions, pre-squash actions, and log probabilities."""
        device = next(self.parameters()).device
        observations = torch.as_tensor(
            local_observations, dtype=torch.float32, device=device
        )
        if observations.shape != (self.num_agents, self.observation_size):
            raise ValueError(
                "local_observations must have shape "
                f"({self.num_agents}, {self.observation_size})."
            )
        actions = [
            self.act_local(index, observations[index], deterministic=deterministic)
            for index in range(self.num_agents)
        ]
        fractions, latent_actions, log_probabilities = zip(*actions)
        return (
            np.stack(fractions),
            np.stack(latent_actions),
            np.asarray(log_probabilities, dtype=np.float32),
        )

    def act_local(
        self,
        agent_index: int,
        local_observation: np.ndarray | Tensor,
        *,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run one decentralized actor using only that station's observation."""
        if not 0 <= agent_index < self.num_agents:
            raise IndexError(f"agent_index must be in [0, {self.num_agents}).")
        device = next(self.parameters()).device
        observation = torch.as_tensor(
            local_observation, dtype=torch.float32, device=device
        )
        if observation.shape != (self.observation_size,):
            raise ValueError(
                f"local_observation must have shape ({self.observation_size},)."
            )
        with torch.no_grad():
            distribution = self.actors[agent_index].distribution(observation)
            latent = distribution.mean if deterministic else distribution.sample()
            tanh_action = torch.tanh(latent)
            fraction = (tanh_action + 1.0) * 0.5
            log_prob = squashed_log_probability(distribution, latent, tanh_action)
        return (
            fraction.cpu().numpy(),
            latent.cpu().numpy(),
            float(log_prob.cpu().item()),
        )

    def evaluate_actions(
        self,
        local_observations: Tensor,
        global_state: Tensor,
        latent_actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        log_probabilities = []
        entropies = []
        for index, actor in enumerate(self.actors):
            distribution = actor.distribution(local_observations[:, index, :])
            latent = latent_actions[:, index, :]
            log_probabilities.append(
                squashed_log_probability(distribution, latent, torch.tanh(latent))
            )
            entropies.append(distribution.entropy().sum(dim=-1))
        return (
            torch.stack(log_probabilities, dim=1),
            torch.stack(entropies, dim=1),
            self.value(global_state),
        )


def squashed_log_probability(
    distribution: Normal, latent_action: Tensor, tanh_action: Tensor
) -> Tensor:
    """Log density after tanh and affine mapping from [-1, 1] to [0, 1]."""
    jacobian = 0.5 * (1.0 - tanh_action.square())
    return (
        distribution.log_prob(latent_action)
        - torch.log(jacobian.clamp_min(1e-6))
    ).sum(dim=-1)


def clipped_surrogate_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantages: Tensor,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    ratios = torch.exp(new_log_probability - old_log_probability)
    unclipped = ratios * advantages
    clipped = torch.clamp(
        ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * advantages
    return -torch.minimum(unclipped, clipped).mean(), ratios


def generalized_advantage_estimate(
    rewards: np.ndarray | Tensor,
    values: np.ndarray | Tensor,
    dones: np.ndarray | Tensor,
    last_values: np.ndarray | Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Calculate time-major GAE and value targets for [time, agent] arrays."""
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    value_tensor = torch.as_tensor(values, dtype=torch.float32)
    done_tensor = torch.as_tensor(dones, dtype=torch.float32)
    last_value_tensor = torch.as_tensor(last_values, dtype=torch.float32)
    if reward_tensor.ndim != 2 or value_tensor.shape != reward_tensor.shape:
        raise ValueError("rewards and values must have matching [time, agent] shapes.")
    if done_tensor.shape not in {
        reward_tensor.shape,
        torch.Size((reward_tensor.shape[0],)),
    }:
        raise ValueError("dones must have shape [time] or [time, agent].")
    if last_value_tensor.shape != reward_tensor.shape[1:]:
        raise ValueError("last_values must have shape [agent].")
    if done_tensor.ndim == 1:
        done_tensor = done_tensor[:, None].expand_as(reward_tensor)

    advantages = torch.zeros_like(reward_tensor)
    next_advantage = torch.zeros(reward_tensor.shape[1], dtype=torch.float32)
    next_value = last_value_tensor
    for step_index in reversed(range(reward_tensor.shape[0])):
        nonterminal = 1.0 - done_tensor[step_index]
        delta = (
            reward_tensor[step_index]
            + gamma * next_value * nonterminal
            - value_tensor[step_index]
        )
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[step_index] = next_advantage
        next_value = value_tensor[step_index]
    return advantages, advantages + value_tensor


class RolloutBuffer:
    """Fixed-length time-major on-policy rollout storage."""

    def __init__(self, rollout_length: int, num_agents: int, observation_size: int, action_size: int):
        self.rollout_length = rollout_length
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.action_size = action_size
        self.clear()

    def clear(self):
        self.local_observations: list[np.ndarray] = []
        self.latent_actions: list[np.ndarray] = []
        self.log_probabilities: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.dones: list[float] = []

    def add(
        self,
        local_observations: np.ndarray,
        latent_actions: np.ndarray,
        log_probabilities: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        done: bool,
    ):
        if len(self.rewards) >= self.rollout_length:
            raise RuntimeError("Rollout buffer is full.")
        expected = {
            "local_observations": (self.num_agents, self.observation_size),
            "latent_actions": (self.num_agents, self.action_size),
            "log_probabilities": (self.num_agents,),
            "rewards": (self.num_agents,),
            "values": (self.num_agents,),
        }
        provided = (
            ("local_observations", local_observations),
            ("latent_actions", latent_actions),
            ("log_probabilities", log_probabilities),
            ("rewards", rewards),
            ("values", values),
        )
        for name, value in provided:
            shape = expected[name]
            value = np.asarray(value, dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}.")
            getattr(self, name).append(value.copy())
        self.dones.append(float(done))

    def as_tensors(self, device: torch.device) -> dict[str, Tensor]:
        if not self.rewards:
            raise ValueError("Cannot tensorize an empty rollout.")
        return {
            name: torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)
            for name, values in (
                ("local_observations", self.local_observations),
                ("latent_actions", self.latent_actions),
                ("log_probabilities", self.log_probabilities),
                ("rewards", self.rewards),
                ("values", self.values),
                ("dones", self.dones),
            )
        }


class MAPPO:
    def __init__(
        self,
        num_agents: int,
        observation_size: int,
        action_size: int = 2,
        config: MAPPOConfig | None = None,
    ):
        self.config = config or MAPPOConfig()
        self.device = torch.device(self.config.device)
        self.policy = MAPPOPolicy(
            num_agents,
            observation_size,
            action_size,
            self.config.hidden_size,
        ).to(self.device)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.actor_learning_rate)
            for actor in self.policy.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.action_size = action_size

    def act(
        self, local_observations: np.ndarray, *, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fractions, latent, log_probabilities = self.policy.act(
            local_observations, deterministic=deterministic
        )
        global_state = torch.as_tensor(
            local_observations.reshape(-1),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            values = self.policy.value(global_state).cpu().numpy()
        return fractions, latent, log_probabilities, values

    def act_local(
        self,
        agent_index: int,
        local_observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Decentralized inference: the selected actor sees one local observation."""
        fractions, _, _ = self.policy.act_local(
            agent_index, local_observation, deterministic=deterministic
        )
        return fractions

    def update(
        self,
        buffer: RolloutBuffer,
        *,
        last_values: np.ndarray | None = None,
    ) -> dict[str, float]:
        batch = buffer.as_tensors(self.device)
        if last_values is None:
            last_values = np.zeros(self.num_agents, dtype=np.float32)
        advantages, returns = generalized_advantage_estimate(
            batch["rewards"].cpu(),
            batch["values"].cpu(),
            batch["dones"].cpu(),
            last_values,
            self.config.gamma,
            self.config.gae_lambda,
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        advantages = (
            advantages - advantages.mean(dim=0, keepdim=True)
        ) / (advantages.std(dim=0, keepdim=True, unbiased=False) + 1e-8)
        time_count = batch["rewards"].shape[0]
        global_states = batch["local_observations"].reshape(time_count, -1)
        policy_losses = []
        value_losses = []
        entropies = []
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(time_count, device=self.device)
            for indices in permutation.split(self.config.minibatch_size):
                new_log_probs, entropy, predicted_values = self.policy.evaluate_actions(
                    batch["local_observations"][indices],
                    global_states[indices],
                    batch["latent_actions"][indices],
                )
                for optimizer in self.actor_optimizers:
                    optimizer.zero_grad(set_to_none=True)
                minibatch_actor_losses = []
                for agent_index, optimizer in enumerate(self.actor_optimizers):
                    policy_loss, _ = clipped_surrogate_loss(
                        new_log_probs[:, agent_index],
                        batch["log_probabilities"][indices, agent_index],
                        advantages[indices, agent_index],
                        self.config.clip_epsilon,
                    )
                    actor_loss = (
                        policy_loss
                        - self.config.entropy_coefficient
                        * entropy[:, agent_index].mean()
                    )
                    minibatch_actor_losses.append(actor_loss)
                    policy_losses.append(policy_loss.detach())
                    entropies.append(entropy[:, agent_index].mean().detach())
                torch.stack(minibatch_actor_losses).sum().backward()
                for agent_index, optimizer in enumerate(self.actor_optimizers):
                    nn.utils.clip_grad_norm_(
                        self.policy.actors[agent_index].parameters(),
                        self.config.max_gradient_norm,
                    )
                    optimizer.step()

                value_loss = 0.5 * (predicted_values - returns[indices]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_coefficient * value_loss).backward()
                nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.config.max_gradient_norm
                )
                self.critic_optimizer.step()
                value_losses.append(value_loss.detach())

        buffer.clear()
        return {
            "policy_loss": torch.stack(policy_losses).mean().item(),
            "value_loss": torch.stack(value_losses).mean().item(),
            "entropy": torch.stack(entropies).mean().item(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        training_config: Mapping[str, Any],
        normalization_config: Mapping[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "training_config": dict(training_config),
                "normalization_config": dict(normalization_config or {}),
                "num_agents": self.num_agents,
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "mappo_config": asdict(self.config),
            },
            checkpoint_path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        num_agents: int | None = None,
    ) -> tuple["MAPPO", dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        config_values = dict(checkpoint["mappo_config"])
        if device is not None:
            config_values["device"] = device
        checkpoint_agents = int(checkpoint["num_agents"])
        requested_agents = checkpoint_agents if num_agents is None else num_agents
        if requested_agents != checkpoint_agents:
            raise ValueError(
                "This checkpoint is fixed to "
                f"{checkpoint_agents} agents; its actor set and/or centralized "
                "critic dimensions cannot be reused for a different station count."
            )
        agent = cls(
            requested_agents,
            checkpoint["observation_size"],
            checkpoint["action_size"],
            MAPPOConfig(**config_values),
        )
        agent.policy.load_state_dict(checkpoint["policy"])
        return agent, checkpoint


def observations_to_array(
    observations: Mapping[str, Mapping[str, float]],
    station_ids: Sequence[str],
    observation_fields: Sequence[str],
) -> np.ndarray:
    """Convert the environment's mapping into stable station/field order."""
    return np.asarray(
        [
            [observations[station_id][field] for field in observation_fields]
            for station_id in station_ids
        ],
        dtype=np.float32,
    )


def fractions_to_environment_action(
    fractions: np.ndarray,
    station_ids: Sequence[str],
    station_capacities: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Scale [0, 1] actor fractions to kWh requests; constraints stay in env."""
    fractions = np.asarray(fractions, dtype=np.float32)
    if fractions.shape != (len(station_ids), 2):
        raise ValueError(f"fractions must have shape ({len(station_ids)}, 2).")
    if not np.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("Action fractions must be finite and lie in [0, 1].")
    return {
        station_id: {
            "renewable": float(fractions[index, 0] * station_capacities[station_id]),
            "grid": float(fractions[index, 1] * station_capacities[station_id]),
        }
        for index, station_id in enumerate(station_ids)
    }


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
