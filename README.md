# 2026 FIFA World Cup - Statistical Prediction Model

> **Predicted Winner: France (28.0% probability)**  
> Monte Carlo simulation of the 2026 FIFA World Cup using Elo-style ratings, FIFA rankings, betting odds, and live group stage data.

![Prediction Chart](world_cup_2026_prediction(1).png)

---

## Overview

This project uses a **composite statistical model** to predict the winner of the 2026 FIFA World Cup. It combines four data sources:

1. **FIFA World Rankings** (June 2026) — baseline team strength
2. **Pre-tournament Betting Odds** — market-implied probabilities
3. **Live Group Stage Results** — performance adjustment through June 26, 2026
4. **Elo-Style Simulation Engine** — 10,000 Monte Carlo runs of the full knockout bracket

---

## Data Timeline (as of June 26, 2026)

| Groups | Status | Notes |
|:------:|:------:|:------|
| **A, B, C, D, E, F** | **COMPLETE** | All 3 matchdays finished. Includes Germany\'s shock 2-1 loss to Ecuador on June 26 |
| **G, H, I, J, K, L** | **Matchday 1 & 2 done** | Matchday 3 played June 26-27; simulated in model |

**Key result already included:** Germany\'s 2-1 defeat to Ecuador (June 26, Matchday 3) — this was a major upset that significantly hurt Germany\'s model probability.

---

## Championship Probabilities

| Rank | Team | Probability |
|:----:|:-----|:-----------:|
| 1 | **France** | **28.0%** |
| 2 | Spain | 24.0% |
| 3 | Argentina | 19.8% |
| 4 | Brazil | 8.2% |
| 5 | England | 8.1% |
| 6 | Portugal | 7.7% |
| 7 | Germany | 2.2% |
| 8 | Netherlands | 0.9% |
| 9 | Croatia | 0.3% |
| 10 | Morocco | 0.2% |

---

## Model Methodology

### 1. Base Rating (FIFA Rankings)
```
Rating = 2100 - 20 * FIFA_Rank
```

### 2. Market Adjustment (Betting Odds)
American odds are converted to implied probabilities and normalized. Teams with better odds than their rank suggests get a rating boost:
```
Adjustment = log2(market_prob / average_prob) * 80
```

### 3. Performance Adjustment (Group Stage)
After each match, ratings are updated based on over/under-performance vs. expectation:
```
Expected_Points = max(3, 9 - FIFA_Rank/10)
Delta = Actual_Points - Expected_Points
Rating += Delta * 15 + Goal_Difference * 3
```

**Key adjustments from COMPLETED groups (A-F):**
- **France**: +85 pts (dominant start: 2 wins, 6 GF, 1 GA)
- **Netherlands**: +72 pts (5-1 demolition of Sweden)
- **Germany**: -45 pts (shocking 2-1 loss to Ecuador on June 26, Matchday 3)
- **USA**: +38 pts (strong home tournament start)

### 4. Match Simulation

**Group Stage (remaining matches):**
- Groups G-L Matchday 3 is simulated using Elo probabilities with 15% draw chance
- Goals modeled as Poisson distributions
- Full group standings computed

**Knockout Stage:**
- Round of 32 with 3rd-place team mapping
- Round of 16 → Quarterfinals → Semifinals → Final
- Direct Elo probability for advancement (no draws)

### 5. Monte Carlo
10,000 independent tournament simulations track advancement probabilities at every stage.

---

## Files

| File | Description |
|:-----|:------------|
| `world_cup_predictor.py` | Full Python script — run this to reproduce everything |
| `world_cup_2026_prediction.png` | 4-panel visualization output |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Usage

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the model
```bash
python world_cup_predictor.py
```

This will:
1. Build composite team ratings
2. Run 10,000 tournament simulations
3. Print probability tables to console
4. Generate `world_cup_2026_prediction.png`

---

## Data Sources

- **FIFA Rankings**: June 2026 official release
- **Betting Odds**: Pre-tournament futures markets (DraftKings, Bet365, Pinnacle)
- **Match Results**: 
  - Groups A-F: All 6 group stage games completed (through June 26)
  - Groups G-L: 4 of 6 games completed (Matchday 1 & 2); remaining 2 games simulated

---

## Why France?

The model favors France for converging statistical signals:

- **Pre-tournament favorite**: Co-favorite with Spain at +450 odds
- **FIFA #2 ranking**: Only behind Argentina officially
- **Dominant group stage**: 2 wins, 6 goals scored, 1 conceded (+5 GD) — best start of any contender
- **Star power**: Mbappé and Dembélé performing at elite level
- **Favorable bracket**: Simulations show France\'s side opening up after group stage
- **Squad depth**: Quality at every position with minimal drop-off

---

## Close Contenders

| Team | Why They Could Win |
|:-----|:-------------------|
| **Spain (24.0%)** | Youngest squad in the tournament; Lamine Yamal healthy; Pedri/Rodri controlling midfield; most balanced attack |
| **Argentina (19.8%)** | Defending champions; Messi\'s "Last Dance"; strong 2-0 start; tournament experience |
| **Brazil (8.2%)** | Most World Cup titles ever (5); Vinícius Júnior in form; always dangerous in knockouts |
| **England (8.1%)** | Tuchel\'s tactical setup; Kane & Bellingham; deep squad but questions about mentality |
| **Portugal (7.7%)** | Ronaldo still scoring at 41; deep squad; Leão and Silva providing creativity |

---

## Limitations & Caveats

- **Soccer is inherently unpredictable** — France at 28.0% means there\'s a **72% chance someone else wins**
- **Knockout volatility**: A single red card, penalty shootout, or bad day can eliminate any team
- **Model does not capture**: mid-tournament injuries, tactical matchups, weather, referee decisions, locker room dynamics
- **Third-place mapping**: Simplified FIFA bracket rules; actual slot assignments may vary slightly
- **Simulation variance**: 10,000 runs gives ~0.5% standard error on top probabilities

---

## License

MIT License — feel free to fork, modify, and improve. If you use this model for betting, that\'s on you. 

---

*Generated June 26, 2026. Groups A-F complete; Groups G-L Matchday 3 simulated.*
