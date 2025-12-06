"""Configuration objects for the customer offer simulation environment."""

from dataclasses import dataclass


@dataclass
class CustomerConfig:
    """Environment tunables for simulated customer behavior."""

    horizon: int = 12  # steps per episode
    base_purchase_prob: float = 0.08  # baseline chance of purchase with no offer
    recency_decay: float = 0.35  # controls how quickly interest fades with time since last purchase
    loyalty_gain: float = 0.06  # loyalty increase after an accepted offer
    loyalty_decay: float = 0.02  # loyalty decrease after a rejected offer
    avg_spend_mean: float = 35.0  # typical basket size for a coffee shop
    avg_spend_std: float = 10.0
    max_recency: int = 30  # clamp for observation scaling

    # Offer-specific parameters
    offer_uplift = {
        0: 0.0,   # no offer
        1: 0.05,  # 5% discount
        2: 0.15,  # 10% discount
        3: 0.3,  # BOGO
        4: 0.10,  # premium membership offer
    }
    offer_cost = {
        0: 0.0,
        1: 0.05,
        2: 0.10,
        3: 0.25,
        4: 0.22,
    }
