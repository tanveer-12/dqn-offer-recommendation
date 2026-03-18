# DQN Offer Engine - Demo Guide

## How to Run

```bash
# 1. Install dependencies (once)
pip install flask flask-cors torch numpy

# 2. Start the API (auto-opens browser)
python api/app.py
```

The browser opens `demo/index.html` automatically. The header badge switches from
**⚠ JS Simulation** (yellow) to **✓ Real Model** (green) when the Flask server is detected.

> You can also open `demo/index.html` directly without the server - it falls back
> to a JavaScript approximation of the trained policy.

---

## Layout at a Glance

```
┌─────────────────────────────────────────────────────────┐
│  Header: logo · status badges · theme toggle            │
├──────────────┬──────────────────────────────────────────┤
│              │  Tabs: Recommendation | 12-Month Journey │
│   Sidebar    │        Rule-Based vs DQN | Insights      │
│  (controls)  ├──────────────────────────────────────────┤
│              │  Main panel (changes per tab)            │
└──────────────┴──────────────────────────────────────────┘
```

---

## Sidebar - Customer Profile

These five sliders define the customer state sent to the model.

| Slider | Range | What it means |
|--------|-------|---------------|
| **Loyalty Score** | 0 – 1 | How attached the customer is. Grows when offers are accepted (+0.06), shrinks when rejected (−0.02). |
| **Avg. Spend** | $10 – $200 | Typical purchase amount per visit. Used to compute revenue. |
| **Days Since Last Visit** | 0 – 30 | Recency. Higher = more lapsed. Agent's acceptance probability decays exponentially with recency. |
| **Accepted Last Offer?** | No / Yes | Whether the previous offer was taken. Affects diversity penalty inside Q-value calculation. |
| **Last Offer Given** | None / 0–4 | Which offer was shown last. Prevents the agent from spamming the same offer repeatedly. |

**Customer Segment** label (below the sliders) is computed from loyalty + recency:

| Segment | Condition | Agent Tendency |
|---------|-----------|----------------|
| Champion | loyalty ≥ 0.7, recency ≤ 5d | No offer or Premium |
| At-Risk Loyal | loyalty ≥ 0.7, recency > 5d | Premium to re-lock |
| Developing | loyalty 0.4–0.7 | Discount to grow |
| Churned | loyalty < 0.4, recency > 10d | BOGO to win back |
| Occasional | everything else | Small nudge |

**Quick Presets** load representative customer profiles instantly.

---

## Buttons

| Button | What happens |
|--------|-------------|
| **⚡ Get Recommendation** | Calls `/predict` (or JS fallback). Populates the Recommendation tab. |
| **▶ Run 12-Month Simulation** | Calls `/simulate`. Runs 12 sequential months, updating loyalty and recency after each step. Switches to the Journey tab. |
| **🔀 Random Customer** | Randomises all sliders. |

---

## Tab 1 - Recommendation

Shows the single-step decision for the current customer state.

### Agent Decision card
- **Offer icon + name** - what the DQN chose.
- **Acceptance Prob.** - probability the customer buys given this offer. Formula:
  ```
  P = base(0.08) + uplift(offer) + 0.35×loyalty + 0.15×exp(−0.35×recency) + spend_bonus
  ```
- **Expected Revenue** - `spend × (1 − offer_cost) × acceptance_prob`.
- **Agent Reasoning** - plain-English explanation of *why* the agent picked this action,
  derived from the Q-value shaping signals.

### Q-Value Distribution card
The neural network outputs one Q-value per action.
**Q-value ≈ expected cumulative discounted reward** if you take that action now and act
optimally afterwards (γ = 0.99).
The bar chart normalises them so the best action fills 100%.

### All-Offer Comparison card
Table of all five offers with acceptance probability and expected revenue side by side.

---

## Tab 2 - 12-Month Journey (Simulation)

### How the simulation works step by step

```
Month 1:
  state = { loyalty, spend, recency, accepted, last_offer }
      ↓
  Q-network → argmax → action (offer)
      ↓
  acceptance_prob(state, action)
      ↓
  random draw → accepted? (yes/no)
      ↓
  revenue = accepted ? spend × (1 − cost) : 0
      ↓
  state update:
    if accepted:  loyalty += 0.06, recency = 0
    if rejected:  loyalty −= 0.02, recency += 1

Repeat for months 2–12 using updated state.
```

The simulation is **stochastic** - same inputs can produce different outcomes across runs
because acceptance is a random draw against the probability.

### Summary stats (top row)
| Stat | Meaning |
|------|---------|
| Total Revenue | Sum of all 12 months |
| Offers Accepted | How many of 12 were taken |
| Final Loyalty | Loyalty at end of month 12 |
| Offer Mix | Which offer type dominated |

### Charts
- **Loyalty Trajectory** - how loyalty evolves month by month.
- **Cumulative Revenue** - running total revenue, shows growth rate.

### Decision Log
Month-by-month table: offer sent, accepted/rejected (green/red), revenue earned,
loyalty at that point.

---

## Tab 3 - Rule-Based vs DQN

### Rule-based policy (left card)
Static if/else logic - the kind most businesses currently use:
```
if recency > 14  → BOGO
elif loyalty > 0.7 → Premium
elif loyalty > 0.4 → 10% Discount
else              → BOGO
```

### DQN policy (right card)
Emergent behaviours the agent discovered that rules cannot express:
- **No offer** for champion customers (saves margin - they'll buy anyway).
- **Premium** for at-risk loyals (locks in lifetime value).
- **Small discount** right after a rejection (avoids pushing harder = churn risk).

### Head-to-Head Comparison
Runs the same starting customer through **both** policies for 12 months and plots
cumulative revenue side by side. The "DQN Advantage" stat is `DQN revenue − rule revenue`.

---

## Tab 4 - Emergent Insights

Static analysis panels showing what the trained agent has learned across thousands
of customer profiles:

| Panel | What it shows |
|-------|--------------|
| **Key Behaviours** | Three top-level insights discovered by the agent |
| **Policy Heatmap** | Dominant action per (loyalty × recency) grid cell |
| **Offer Usage Distribution** | How often each of the 5 actions is chosen overall |
| **Acceptance Rate by Offer** | Which offers get accepted most (BOGO wins on acceptance, Premium wins on LTV) |

---

## Header Badges

| Badge | Meaning |
|-------|---------|
| ⚠ JS Simulation (yellow) | Flask API is offline; Q-values approximated in browser JS |
| ✓ Real Model (green) | Connected to `api/app.py`; real PyTorch weights in use |
| Running simulation… (green dot) | A `/simulate` call is in flight |
| PyTorch · DQN | Tech stack label |
| 10K Customers Trained | Training scale label |
| ☀️ Light / 🌙 Dark | Theme toggle (preference saved in localStorage) |

---

## Offer Reference

| ID | Name | Cost | Purchase Uplift |
|----|------|------|----------------|
| 0 | No Offer | 0% | 0% |
| 1 | 5% Discount | 5% | +5% |
| 2 | 10% Discount | 10% | +15% |
| 3 | BOGO | 25% | +30% |
| 4 | Premium Membership | 22% | +10% |

**Cost** is deducted from revenue. **Uplift** is added to baseline acceptance probability.

---

## API Endpoints (when server is running)

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/health` | GET | - | `{ status, model }` |
| `/predict` | POST | `{ loyalty, spend, recency, accepted?, last_offer? }` | `{ action, offer, q_values[5], accept_probs[5], exp_revenues[5] }` |
| `/simulate` | POST | same + `{ rule_based?: bool }` | `{ timeline[12], loyalty_history, revenue_history, total_revenue, total_accepted, offer_counts, final_loyalty }` |

---

## Suggested Next Feature - Offer Budget Cap

> **Real-world feature businesses actually need:**
>
> **Daily / Monthly Offer Budget Cap**
> Companies set a marketing spend ceiling (e.g., $500/day in discounts). Once
> the budget is consumed, the API automatically falls back to "No Offer" regardless
> of Q-values. This prevents the model from aggressively discounting beyond what
> the business can afford, and is the single most-asked-for constraint in
> production recommendation systems.
>
> **Implementation sketch:**
> Add a `budget_remaining` field to the simulation state. After each accepted offer,
> deduct `offer_cost × spend` from the budget. If `budget_remaining < min_cost`,
> force `action = 0`. The demo would show a "Budget Used" progress bar in the
> sidebar.
