# FX Risk Regime Analysis

Regime-based FX risk monitoring and hedging framework using USD/KRW exchange rate data.

## Overview

This project analyzes USD/KRW exchange rate risk through volatility modeling, regime detection, and risk quantification.

Rather than attempting to predict future exchange rates directly, the framework focuses on identifying hidden market regimes and translating them into practical risk management decisions.

The project combines Hidden Markov Models (HMM), Value-at-Risk (VaR), volatility persistence analysis, and macro-financial signals to evaluate FX exposure under different market conditions.

Results below are computed on USD/KRW daily data from 2000-01-01 to 2026-05-12 (5,761 trading days), cached locally for reproducibility.

---

## Research Question

How can firms manage USD/KRW exchange rate risk when accurate exchange rate forecasting is inherently difficult?

This study approaches the problem from a state-detection perspective rather than a forecasting perspective.

---

## Methodology

### 1. Market Structure Analysis

* USD/KRW trend analysis
* Rolling annualized volatility
* Volatility persistence (ACF)

### 2. Risk Quantification

* Historical Value-at-Risk (VaR)
* Return distribution analysis
* Jarque-Bera normality test

### 3. Regime Detection

A Hidden Markov Model (HMM) is used to classify the FX market into three hidden states:

* Stable
* Transition
* Crisis

### 4. Transition Analysis

* Theoretical transition matrix
* Empirical transition matrix
* Regime persistence analysis

### 5. Regime-Specific Risk Measurement

* Conditional VaR by regime
* Crisis loss estimation
* Detection delay analysis

### 6. Global Signal Analysis

Lead-lag relationships between USD/KRW and:

* VIX
* DXY
* USD/CNY

### 7. Hedging Framework

Development of a regime-aware, rule-based hedging framework:

* Hedge ratio adjustment by regime
* Firm-type adjustment rules
* Hedge budget allocation framework

---

## Key Findings

### Regime Characteristics

| Regime     | Share | Mean Volatility | Self-persistence P(ii) | Daily 95% VaR | Skewness |
| ---------- | :---: | :-------------: | :--------------------: | :-----------: | :------: |
| Stable     | 28.6% | 0.0052          | 47.1%                  | −1.14%        | +0.84    |
| Transition | 38.4% | 0.0053          | 26.0%                  | —             | −0.74    |
| Crisis     | 33.0% | 0.0121          | 98.2%                  | −2.19%        | +0.13    |

Stable and Transition share nearly identical volatility (0.0052 vs 0.0053); they are separated by direction, not magnitude. Stable drifts toward KRW strength with a right-tail (skew +0.84), while Transition drifts toward KRW weakness with a left-tail (skew −0.74).

### Risk Concentration

* Crisis-state 95% VaR (−2.19%) was roughly 1.9× the Stable-state VaR (−1.14%).
* A single whole-sample VaR (−1.16%) masks this gap by averaging across regimes.
* On a $100M position (at 1,450 KRW/USD), the daily 95% loss rises from ≈ ₩1.65B (Stable) to ≈ ₩3.18B (Crisis).
* Jarque-Bera rejects normality (p ≈ 0); tail risk concentrates in a limited number of high-volatility periods.

### Regime Persistence

* Crisis is highly persistent: P(Crisis → Crisis) = 98.2%, an expected duration of over 56 days.
* Volatility autocorrelation (ACF) stays outside the 95% confidence band up to ~150 trading days (about six months).
* Stable and Transition churn between each other, while Crisis, once entered, tends to stay.

### Detection Delay

Regime identification is not instantaneous, and the delay depends on the shock type.

* Acute shocks (e.g., Lehman, COVID): ~0-day detection lag.
* Gradual shocks (e.g., Fed 75bp hike): up to ~147 days before Crisis is confirmed.
* Average detection lag: ~40 days.

With a 5-day signal lead, ~40-day average lag, and ~3-day execution time, the effective action window is negative on average. Preemptive hedging is structurally viable for acute shocks but requires pre-committed forwards for gradual ones.

### Global Signals

* VIX lead-lag correlation peaked at +0.536 (lag +5d), but the profile was nearly flat (~0.52) across lags — consistent with co-movement rather than a true lead.
* USD/CNY showed weak linear correlation (−0.034) but a proxy-selling channel during Asian risk-off episodes.
* External indicators are best read as real-time monitoring signals, not predictive leads.

---

## Hedging Framework

Hedge ratios are computed additively rather than from a single fixed value:

```text
Final HR = clip( base(regime) + Δexposure + Δsize + Δskew, 0, 1 )
```

`base` is a policy-fixed value per regime (Stable 30% / Transition 60% / Crisis 90%), adjusted by firm type, firm size, and regime skewness. The hedge budget links daily VaR and regime persistence:

```text
Daily budget      = λ × VaR_t
Expected duration = 1 / (1 - P(ii))
Cumulative cost   = Daily budget × Expected duration
```

Because Crisis persistence is high, its expected duration — and therefore cumulative hedge cost — is large, which is the quantitative case for preemptive over reactive hedging.

---

## Limitations

This framework is designed for risk monitoring rather than prediction.

* Annualized VaR (daily × √252) assumes i.i.d. normality and underestimates tail loss in a fat-tailed series.
* HMM is backward-looking and cannot eliminate detection delay.
* VIX–volatility lead-lag is a level-to-level correlation; common trends inflate it, so the flat profile is read conservatively as co-movement.
* Regime classifications can shift with model specification and observation window.
* Hedge ratios are rule-based policy parameters, not statistically optimized estimates.

---

## Repository Structure

```text
coremodel_fxrisk.py
    Core analytical library

main_research_for_fxrisk.py
    End-to-end research pipeline

README.md
    Project documentation
```

---

## Technologies

* Python
* Pandas
* NumPy
* SciPy
* Statsmodels
* Matplotlib
* hmmlearn

---

## Project Scope

This project is not a return forecasting model. Its primary objective is to provide a practical framework for monitoring FX risk, identifying market regimes, and supporting hedging decisions under changing market conditions.

## Data

* USD/KRW, VIX, DXY (UUP), USD/CNY — Yahoo Finance (cached to CSV for reproducibility)
* Foreign investor net flow (planned extension) — KRX Data System

## Author

장효원 (JangHyowon-biz)
