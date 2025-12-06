"""Gymnasium environment simulating customer responses to personalized offers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .config import CustomerConfig


@dataclass
class CustomerState:
    loyalty_score: float
    average_spend: float
    last_offer: int
    accepted_last_offer: int
    recency_of_purchase: int


class CustomerOfferEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[CustomerConfig] = None, seed: Optional[int] = None) -> None:
        super().__init__()
        self.config = config or CustomerConfig()
        self.np_random = np.random.default_rng(seed)
        # observation: [loyalty_score, average_spend, last_offer, accepted_last_offer, recency_of_purchase]
        low = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 500.0, 4.0, 1.0, float(self.config.max_recency)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.state: Optional[CustomerState] = None
        self.timestep = 0
        self.initial_loyalty = 0.0      # Track initial loyalty for long-term metrics

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        avg_spend = float(
            np.clip(
                self.np_random.normal(self.config.avg_spend_mean, self.config.avg_spend_std),
                10.0,
                500.0,
            )
        )
        loyalty = float(np.clip(self.np_random.uniform(0.1, 0.6), 0.0, 1.0))
        self.initial_loyalty = loyalty  # Store for final reward calculation
        recency = int(self.np_random.integers(0, 5))
        self.state = CustomerState(
            loyalty_score=loyalty,
            average_spend=avg_spend,
            last_offer=-1,
            accepted_last_offer=0,
            recency_of_purchase=recency,
        )
        self.timestep = 0
        return self._get_obs(), self._get_info(accepted=False, purchase_amount=0.0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."
        assert self.action_space.contains(action), f"Invalid action {action}"

        accepted = self._did_accept_offer(action)
        purchase_amount = self.state.average_spend if accepted else 0.0
        
        # # Calculate immediate reward
        # immediate_revenue = purchase_amount * (1.0 - self.config.offer_cost[action])
        
        # Calculate final reward that includes long-term value
        reward = self._calculate_improved_reward(action, purchase_amount, accepted)
        
        self._update_state_after_action(action, accepted)
        self.timestep += 1
        terminated = self.timestep >= self.config.horizon
        truncated = False
        obs = self._get_obs()
        info = self._get_info(accepted=accepted, purchase_amount=purchase_amount)
        return obs, float(reward), terminated, truncated, info

    # AFTER DQN 3 BALANCED TRAINING, FOR TRAIN_FINAL TRAINING
    def _calculate_improved_reward(self, action: int, purchase_amount: float, accepted: bool):
        """Improved reward with better cost-effectiveness and customer value focus."""
        if not accepted:
            return 0.0
        
        # Base revenue calculation
        base_revenue = purchase_amount * (1.0 - self.config.offer_cost[action])
        
        # Cost-effectiveness penalty for expensive offers
        offer_cost = self.config.offer_cost[action]
        
        # Penalty for using expensive offers inappropriately
        penalty = 0.0
        if action == 3 and self.state.loyalty_score > 0.5:  # BOGO for high loyalty
            penalty = 15.0
        elif action == 4 and self.state.loyalty_score < 0.3:  # Premium for low loyalty
            penalty = 10.0
        elif action == 2 and self.state.loyalty_score > 0.7 and self.state.accepted_last_offer == 1:  # 10% for high loyalty with recent acceptance
            penalty = 5.0  # Encourage trying different offers
        
        # Bonus for appropriate offer usage
        bonus = 0.0
        if action == 0 and self.state.loyalty_score > 0.6:  # No offer for high loyalty
            bonus = 8.0
        elif action == 4 and self.state.loyalty_score > 0.6:  # Premium for high loyalty
            bonus = 12.0
        elif action == 1 and self.state.loyalty_score < 0.4 and self.state.loyalty_score > 0.2:  # 5% for moderate loyalty
            bonus = 5.0
        elif action == 3 and self.state.loyalty_score < 0.3 and self.state.recency_of_purchase > 5:  # BOGO for inactive
            bonus = 8.0
        elif action == 1 and self.state.accepted_last_offer == 0:  # Small discount after rejection
            bonus = 3.0  # Encourage recovery from rejections
        
        # Loyalty building component
        loyalty_bonus = max(0, (self.state.loyalty_score - 0.5) * 10.0)
        
        # Diversity bonus: penalize repeating same offer too often
        if self.state.last_offer == action and self.state.accepted_last_offer == 1:
            penalty += 2.0  # Small penalty for repeating same offer after acceptance
        
        total_reward = base_revenue - penalty + bonus + loyalty_bonus
        return total_reward


    def _did_accept_offer(self, action: int) -> bool:
        """Compute acceptance probability based on offer, loyalty, recency, and average spend."""
        assert self.state is not None
        uplift = self.config.offer_uplift[action]
        loyalty_component = 0.35 * self.state.loyalty_score
        recency_component = np.exp(-self.config.recency_decay * self.state.recency_of_purchase)
        spend_component = 0.08 if self.state.average_spend > 100 else 0.02
        prob = self.config.base_purchase_prob + uplift + loyalty_component + 0.15 * recency_component + spend_component
        prob = float(np.clip(prob, 0.0, 0.95))
        return bool(self.np_random.random() < prob)

    def _update_state_after_action(self, action: int, accepted: bool) -> None:
        assert self.state is not None
        loyalty = self.state.loyalty_score
        recency = self.state.recency_of_purchase

        if accepted:
            loyalty = np.clip(loyalty + self.config.loyalty_gain, 0.0, 1.0)
            recency = 0
        else:
            loyalty = np.clip(loyalty - self.config.loyalty_decay, 0.0, 1.0)
            recency = min(recency + 1, self.config.max_recency)

        self.state = CustomerState(
            loyalty_score=float(loyalty),
            average_spend=self.state.average_spend,
            last_offer=int(action),
            accepted_last_offer=int(accepted),
            recency_of_purchase=int(recency),
        )

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        obs = np.array(
            [
                self.state.loyalty_score,
                self.state.average_spend,
                self.state.last_offer,
                self.state.accepted_last_offer,
                float(self.state.recency_of_purchase),
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self, *, accepted: bool, purchase_amount: float) -> Dict[str, Any]:
        assert self.state is not None
        return {
            "accepted": accepted,
            "purchase_amount": purchase_amount,
            "loyalty_score": self.state.loyalty_score,
            "recency_of_purchase": self.state.recency_of_purchase,
        }
