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
        
        # Calculate immediate reward
        immediate_revenue = purchase_amount * (1.0 - self.config.offer_cost[action])
        
        # Calculate final reward that includes long-term value
        reward = self._calculate_final_reward(action, purchase_amount, accepted, immediate_revenue)
        
        self._update_state_after_action(action, accepted)
        self.timestep += 1
        terminated = self.timestep >= self.config.horizon
        truncated = False
        obs = self._get_obs()
        info = self._get_info(accepted=accepted, purchase_amount=purchase_amount)
        return obs, float(reward), terminated, truncated, info

    def _calculate_final_reward(self, action: int, purchase_amount: float, accepted: bool, immediate_revenue: float):
        """Calculate reward with long-term considerations."""
        if not accepted:
            return 0.0  # No immediate reward for rejected offers
        
        # Base revenue
        reward = immediate_revenue
        
        # Offer efficiency bonus/penalty
        offer_cost = self.config.offer_cost[action]
        
        # Penalty for using expensive offers when cheaper ones would work
        if offer_cost > 0.15:  # BOGO or Premium
            # Heavy penalty if customer has low spend or low loyalty
            if self.state.average_spend < 80 or self.state.loyalty_score < 0.4:
                reward -= offer_cost * 100  # Significant penalty
            elif self.state.average_spend < 120 and action == 3:  # BOGO on low-medium value
                reward -= offer_cost * 50  # Medium penalty
        
        # Bonus for cost-effective offers
        elif action == 0:  # No offer - most profitable
            if self.state.loyalty_score > 0.6:  # High loyalty customers
                reward += 10.0
        elif action == 1:  # 5% discount
            if self.state.average_spend > 50 and self.state.loyalty_score < 0.5:
                reward += 3.0  # Good for moderate customers
        
        # Loyalty building component
        loyalty_bonus = (self.state.loyalty_score - 0.5) * 15.0  # Bonus for maintaining high loyalty
        reward += max(0, loyalty_bonus)
        
        return reward

    # def _calculate_cost_penalty(self, action:int, purchase_amount:float):
    #     """Penalize expensive offers that don't generate sufficient revenue"""
    #     offer_cost = self.config.offer_cost[action]
    #     cost_ratio = offer_cost * purchase_amount   # How expensive is this offer relative to purchase

    #     # if the offer cost is too high relative to purchase amount, apply penalty
    #     if offer_cost > 0.15 and purchase_amount < 80:    # Expensive offers on low-value customers
    #         return offer_cost * purchase_amount * 0.8
    #     elif offer_cost > 0.20 and purchase_amount < 120:   # very expensive offers
    #         return offer_cost * purchase_amount * 1.2
    #     else:
    #         return 0.0

    # def _calculate_loyalty_bonus(self):
    #     """Bonus for building customer loyalty"""
    #     if self.state is not None:
    #         # Small bonus for maintaining or increasing loyalty
    #         loyalty_change = self.config.loyalty_gain if True else -self.config.loyalty_decay
    #         return loyalty_change * 50.0    # Scale up the loyalty impact
    #     return 0.0

    # def _calculate_efficiency_bonus(self, action: int, purchase_amount:float):
    #     """Bonus for using cost-effective offers"""
    #     offer_cost = self.config.offer_cost[action]

    #     # Bonus for no offer (most profitable) on high-value customers
    #     if action == 0 and purchase_amount > 100:
    #         return 10.0
    #     elif action == 0 and purchase_amount > 50:
    #         return 5.0
    #     # Bonus for small discounts(5%) on medium-value customers
    #     elif action == 1 and purchase_amount > 60:
    #         return 3.0
    #     # Bonus for 10% disc when it works well
    #     elif action == 2 and purchase_amount > 70:
    #         return 2.0
    #     else:
    #         return 0.0

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
