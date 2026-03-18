"""
Flask API — serves the trained DQN model for the interactive demo.
Run: python api/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.offer_env.q_network import QNetwork
from src.offer_env.config import CustomerConfig

# ── Constants ────────────────────────────────────────────────────────────────
STATE_SIZE  = 5
ACTION_SIZE = 5
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'cafe_recommendation_model.pth')

OFFER_META = [
    {"id": 0, "name": "No Offer",           "icon": "🚫", "cost": 0.00, "uplift": 0.00},
    {"id": 1, "name": "5% Discount",        "icon": "🏷️", "cost": 0.05, "uplift": 0.05},
    {"id": 2, "name": "10% Discount",       "icon": "💰", "cost": 0.10, "uplift": 0.15},
    {"id": 3, "name": "BOGO",               "icon": "🎁", "cost": 0.25, "uplift": 0.30},
    {"id": 4, "name": "Premium Membership", "icon": "⭐", "cost": 0.22, "uplift": 0.10},
]

cfg = CustomerConfig()

# ── Load model ────────────────────────────────────────────────────────────────
network = QNetwork(STATE_SIZE, ACTION_SIZE)
network.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
network.eval()
print(f"[API] Model loaded from {MODEL_PATH}")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from file:// or any origin


def state_to_tensor(s: dict) -> torch.Tensor:
    """Convert JSON state dict → [1, 5] float32 tensor (raw, matches training)."""
    vec = np.array([
        float(s["loyalty"]),
        float(s["spend"]),
        float(s.get("last_offer", -1)),
        float(s.get("accepted", 0)),
        float(s["recency"]),
    ], dtype=np.float32)
    return torch.from_numpy(vec).unsqueeze(0)   # shape [1, 5]


def acceptance_prob(loyalty, spend, recency, offer_id):
    uplift = cfg.offer_uplift[offer_id]
    loyalty_comp  = 0.35 * loyalty
    recency_comp  = np.exp(-cfg.recency_decay * recency)
    spend_comp    = 0.08 if spend > 100 else 0.02
    prob = cfg.base_purchase_prob + uplift + loyalty_comp + 0.15 * recency_comp + spend_comp
    return float(np.clip(prob, 0.0, 0.95))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "cafe_recommendation_model.pth"})


@app.post("/predict")
def predict():
    """
    Body: { loyalty, spend, recency, accepted, last_offer }
    Returns: { action, offer, q_values, accept_probs, exp_revenues }
    """
    data = request.get_json(force=True)

    with torch.no_grad():
        q_tensor = network(state_to_tensor(data))

    q_values = q_tensor.squeeze().tolist()
    action   = int(np.argmax(q_values))

    loyalty = float(data["loyalty"])
    spend   = float(data["spend"])
    recency = float(data["recency"])

    accept_probs  = [acceptance_prob(loyalty, spend, recency, i) for i in range(ACTION_SIZE)]
    exp_revenues  = [spend * (1 - OFFER_META[i]["cost"]) * accept_probs[i] for i in range(ACTION_SIZE)]

    return jsonify({
        "action":       action,
        "offer":        OFFER_META[action],
        "q_values":     q_values,
        "accept_probs": accept_probs,
        "exp_revenues": exp_revenues,
        "source":       "model",
    })


@app.post("/simulate")
def simulate():
    """
    Body: { loyalty, spend, recency, accepted, last_offer, rule_based? }
    Returns full 12-month simulation using the real model.
    """
    data      = request.get_json(force=True)
    use_rules = bool(data.get("rule_based", False))

    state = {
        "loyalty":    float(data["loyalty"]),
        "spend":      float(data["spend"]),
        "recency":    float(data["recency"]),
        "accepted":   int(data.get("accepted", 0)),
        "last_offer": int(data.get("last_offer", -1)),
    }

    rng = np.random.default_rng()
    loyalty_history = [state["loyalty"]]
    revenue_history = [0.0]
    cumulative_rev  = 0.0
    timeline        = []
    offer_counts    = [0] * ACTION_SIZE

    for month in range(1, 13):
        if use_rules:
            action = _rule_based(state["loyalty"], state["recency"])
        else:
            with torch.no_grad():
                q_tensor = network(state_to_tensor(state))
            action = int(np.argmax(q_tensor.squeeze().tolist()))

        offer  = OFFER_META[action]
        prob   = acceptance_prob(state["loyalty"], state["spend"], state["recency"], action)
        accepted = bool(rng.random() < prob)
        revenue  = state["spend"] * (1 - offer["cost"]) if accepted else 0.0

        cumulative_rev += revenue
        offer_counts[action] += 1
        q_vals = [] if use_rules else q_tensor.squeeze().tolist()

        timeline.append({
            "month":    month,
            "action":   action,
            "offer":    offer,
            "accepted": accepted,
            "revenue":  round(revenue, 2),
            "prob":     round(prob, 4),
            "loyalty":  round(state["loyalty"], 4),
            "q_values": [round(v, 3) for v in q_vals],
        })

        # State transition (matches customer_env.py)
        if accepted:
            state["loyalty"] = float(np.clip(state["loyalty"] + cfg.loyalty_gain, 0.0, 1.0))
            state["recency"] = 0
            state["accepted"] = 1
        else:
            state["loyalty"] = float(np.clip(state["loyalty"] - cfg.loyalty_decay, 0.0, 1.0))
            state["recency"] = min(state["recency"] + 1, cfg.max_recency)
            state["accepted"] = 0
        state["last_offer"] = action

        loyalty_history.append(round(state["loyalty"], 4))
        revenue_history.append(round(cumulative_rev, 2))

    total_accepted = sum(1 for t in timeline if t["accepted"])
    dominant       = int(np.argmax(offer_counts))

    return jsonify({
        "timeline":        timeline,
        "loyalty_history": loyalty_history,
        "revenue_history": revenue_history,
        "total_revenue":   round(cumulative_rev, 2),
        "total_accepted":  total_accepted,
        "offer_counts":    offer_counts,
        "final_loyalty":   round(state["loyalty"], 4),
        "dominant_offer":  dominant,
        "source":          "model",
    })


def _rule_based(loyalty, recency):
    if recency > 14:
        return 3
    if loyalty > 0.7:
        return 4
    if loyalty > 0.4:
        return 2
    return 3


if __name__ == "__main__":
    print("[API] Starting on http://localhost:5000")
    app.run(debug=False, port=5000)
