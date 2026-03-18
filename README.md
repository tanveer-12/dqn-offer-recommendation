# ⚡ DQN Offer Engine

> *"What if instead of a tired marketing intern blindly sending everyone a 10% coupon, a neural network did it — but smarter, cheaper, and without needing coffee?"*

A Deep Reinforcement Learning system that learns **which offer to send to which customer, at the right time** — trained on 10,000 simulated café customers and served through a live interactive dashboard.

Built by **Tanveer Kaur & Davud Azizov**

---

## 🤔 Wait, Why Does This Exist?

Picture your local café's marketing strategy right now:

- Customer visited 3 weeks ago? **BOGO coupon.**
- Customer visited yesterday? **BOGO coupon.**
- Customer just bought a lifetime membership? **BOGO coupon.**

Yeah. That's it. That's the whole strategy.

This project replaces that with a **Deep Q-Network (DQN)** — an AI agent that actually *thinks* about whether you deserve a coupon, or whether you'd have bought that latte anyway (in which case, why waste the budget?).

---

## 🎬 See It in Action

> **Recommendation Demo**

<!-- PLACEHOLDER: Video showing recommendation in action -->
<!-- Replace the block below with your actual video embed or GIF -->
<!-- Suggested: screen recording of dashboard → adjust sliders → Get Recommendation → Run 12-Month Simulation -->

```
[ 📹 VIDEO PLACEHOLDER ]
Drop a screen recording of the demo dashboard here.
```

---

## 📸 Screenshots

### The Dashboard — Recommendation Mode

> Adjust customer sliders, hit the button, watch the AI decide whether you deserve a coupon.

<!-- PLACEHOLDER: Screenshot of the main recommendation window -->
<!-- Suggested filename: screenshots/recommendation_panel.png -->

```
[ 🖼️ IMAGE PLACEHOLDER — Main Recommendation Window ]
```

---

### The Showdown — Rule-Based vs. DQN

> Spoiler: the AI wins. But seeing *why* it wins is the interesting part.

<!-- PLACEHOLDER: Screenshot of the Rule-Based vs DQN comparison tab -->
<!-- Suggested filename: screenshots/comparison_panel.png -->

```
[ 🖼️ IMAGE PLACEHOLDER — Rule-Based vs DQN Comparison ]
```

---

## 🧠 What's Actually Going On Here?

The agent learns by playing a game — repeatedly.

**The game:**
1. A customer shows up with a loyalty score, spending habit, and days since their last visit
2. The agent picks one of 5 offers to send them
3. The customer either buys or doesn't (probability-based, not rigged)
4. The agent gets a reward or a penalty
5. Repeat 10,000 times until the agent stops doing dumb things like sending a BOGO to a customer who visits every single day

After training, the agent has learned things no human bothered to write down:

| Situation | What a human writes in the rulebook | What the AI figures out |
|-----------|-------------------------------------|------------------------|
| Super loyal, visits daily | "Give them Premium!" | "They'll buy anyway. Send nothing. Save the money." |
| Loyal but hasn't visited in 3 weeks | "Give them Premium!" | "They're drifting. Premium now locks them back in." |
| Low loyalty, hasn't visited in ages | "Give them BOGO!" | Also BOGO — but for the *right* reason, with the *right* timing |
| Customer just rejected your last offer | "Try a bigger discount!" | "Calm down. Small nudge. Don't push harder." |

---

## 🎰 The 5 Offers

| # | Offer | Your Cost | Their Motivation to Buy |
|---|-------|-----------|------------------------|
| 0 | 🚫 No Offer | 0% | Whatever they were already going to do |
| 1 | 🏷️ 5% Discount | 5% | A little nudge |
| 2 | 💰 10% Discount | 10% | A solid nudge |
| 3 | 🎁 BOGO | 25% | A very aggressive nudge |
| 4 | ⭐ Premium Membership | 22% | "We're in a relationship now" |

The agent's job is to pick the one that **maximizes money over 12 months** — not just today. This is the key difference from a rule-based system, which optimizes for "did they click the coupon" and calls it a day.

---

## 🏗️ Project Structure

```
dqn-offer-recommendation/
│
├── src/
│   ├── agents/
│   │   └── dqn_agent.py          ← The brain (minimal, clean)
│   └── offer_env/
│       ├── customer_env.py       ← The fake café world
│       ├── dqn_agent.py          ← The brain (full version, with comments)
│       ├── q_network.py          ← 3-layer neural net: 5 inputs → 64 → 64 → 5 outputs
│       ├── replay_buffer.py      ← Where the AI remembers its past mistakes
│       └── config.py             ← All the tunable knobs in one place
│
├── scripts/
│   ├── train_dqn.py              ← Start training (grab a coffee, it'll take a bit)
│   ├── train_cafe_system.py      ← Bigger training run (grab two coffees)
│   ├── eval_dqn.py               ← See how good the model got
│   ├── plot_training.py          ← Draw the "it's learning!" graphs
│   └── random_rollout.py         ← Sanity check: does the environment even work?
│
├── api/
│   └── app.py                    ← Flask server: asks the model what to do
│
├── demo/
│   ├── index.html                ← The interactive dashboard (open this)
│   └── DEMO_GUIDE.md             ← "What does this button do?" — answered
│
├── models/
│   ├── cafe_recommendation_model.pth  ← The trained model (the star)
│   ├── dqn_vanilla_256.pth            ← The baseline (good, not great)
│   └── dqn_double_256.pth             ← Double DQN (better, less overconfident)
│
└── logs/
    └── train_double.csv          ← Proof that the training actually improved
```

---

## 🚀 Quick Start

**Step 1 — Install dependencies**
```bash
pip install torch gymnasium numpy matplotlib flask flask-cors
```

**Step 2 — Start the demo** *(browser opens automatically)*
```bash
python api/app.py
```

That's it. The dashboard opens in your browser. The header badge turns green when it's talking to the real model.

> No server running? The demo still works — it falls back to a JavaScript approximation of the trained policy. Slightly less accurate, equally fun to play with.

---

## 🏋️ Training Your Own Model

```bash
# Quick training run (400 episodes, good for testing)
python scripts/train_dqn.py --episodes 400 --double-dqn --reward-scale 0.01

# Full café system training (10,000 customers)
python scripts/train_cafe_system.py --episodes 10000

# Evaluate a trained model
python scripts/eval_dqn.py --model-path models/dqn_double_256.pth

# Plot training curves
python scripts/plot_training.py logs/train_metrics.csv
```

---

## 📐 The Math (Simplified Dramatically)

The agent uses the **Bellman equation** to figure out what a decision is worth:

```
Q(state, action) = immediate reward + 0.99 × (best possible future reward)
```

Translation: *"Is this coupon worth sending right now, considering what might happen over the next 11 months?"*

The `0.99` (gamma) means the agent cares a lot about the future — not as much as the present, but enough to plan ahead. An agent with gamma = `0.0` would be a YOLO machine. An agent with gamma = `0.99` thinks like a business owner.

**Double DQN** is used to stop the agent from getting overconfident — it uses two networks: one to pick the best action, another to calmly evaluate how good that action actually is. It's the AI equivalent of asking for a second opinion before sending everyone a BOGO.

---

## 📊 How the Simulation Works

When you hit **"Run 12-Month Simulation"**, here's what actually happens:

```
Month 1
  → Read customer state: { loyalty: 0.45, spend: $60, recency: 3d, ... }
  → Feed into Q-network → get 5 Q-values
  → Pick action with highest Q-value (e.g., "10% Discount")
  → Roll the dice: does the customer accept? (probability-based)
  → If yes: loyalty +0.06, recency resets to 0, revenue += $54
  → If no:  loyalty −0.02, recency +1, revenue += $0

Month 2
  → Repeat with updated state
  → ...

Month 12
  → Total up revenue, final loyalty, offer breakdown
  → Show everything on the dashboard
```

Each run is slightly different because customer acceptance is stochastic — same inputs, different luck. Run it a few times on the same customer to see the variance.

---

## 🤺 DQN vs. Rule-Based

The rule-based system is honest about what it is: a few `if` statements written in 10 minutes by someone who had a reasonable intuition about customer behaviour.

```python
if recency > 14:     return BOGO      # lapsed customer
if loyalty > 0.7:    return Premium   # reward loyalty
if loyalty > 0.4:    return 10% Off   # encourage growth
else:                return BOGO      # low loyalty fallback
```

It's not *wrong*. It's just *static*. It doesn't know that:
- A highly loyal customer with high recency doesn't need BOGO — they need nothing
- A customer who just rejected your last offer is not impressed by a bigger discount
- Some customers are only worth a 5% nudge — sending BOGO to them is just giving money away

The DQN figured all of this out by playing the simulation 10,000 times.

---

## 🧩 Customer State (What the Model Sees)

| Feature | Range | What It Represents |
|---------|-------|--------------------|
| `loyalty_score` | 0.0 – 1.0 | How attached is this customer? Grows when they accept offers, shrinks when they don't |
| `avg_spend` | $10 – $200 | How much do they typically spend per visit |
| `recency` | 0 – 30 days | Days since last visit — the "are they drifting?" signal |
| `accepted_last_offer` | 0 or 1 | Did they take the last coupon? Context for the next decision |
| `last_offer` | -1 to 4 | What was the last offer sent? Prevents the agent from being repetitive |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| RL Framework | PyTorch (manual DQN implementation) |
| Environment | Gymnasium (custom `CustomerOfferEnv`) |
| API | Flask + Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Numerics | NumPy |
| Visualization | Matplotlib |

No overengineering. No Kubernetes. No blockchain. Just a neural network, a fake café, and a dashboard.

---

## 📈 Results

After training on 10,000 customer episodes with Double DQN:

- The agent consistently outperforms the rule-based baseline in head-to-head 12-month simulations
- It discovered the **"no offer"** strategy for champion customers entirely on its own — this was never explicitly programmed
- It learned to send smaller offers after rejections rather than escalating (which is what the rules do)
- Final loyalty scores are higher on average compared to rule-based, because the agent invests in the relationship rather than just chasing the immediate transaction

---

## 💡 Suggested Next Feature

> **Offer Budget Capping** — the thing every real business actually needs.
>
> Set a daily spend ceiling (e.g. $500/day in discount value). Once the budget is hit, the API forces `action = 0` (No Offer) regardless of what the model wants. Prevents the agent from being generous with someone else's money.
>
> See [demo/DEMO_GUIDE.md](demo/DEMO_GUIDE.md) for the full implementation sketch.

---

## 📄 License

MIT — take it, break it, make it sell more coffee.

---

*Built with PyTorch, Flask, too much caffeine, and a genuine curiosity about whether an AI could outsmart a loyalty card program.*
