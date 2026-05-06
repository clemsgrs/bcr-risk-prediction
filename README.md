# BCR Risk Prediction

Predict biochemical recurrence (BCR) risk from H&E prostatectomy whole slide images (WSIs) using deep learning.

## Overview

This project combines:
1. **HIPT (Hierarchical Image Pyramid Transformer)**: A MIL-based aggregator that learns slide-level representations from pre-extracted tile features
2. **Survival modeling**: Time-to-event prediction with censoring support

The pipeline takes pre-extracted tile-level features (from a pathology foundation model) and predicts time to biochemical recurrence.

## Survival Modeling Approaches

### 1. Discrete-Time Survival (Default)

The continuous time scale is partitioned into `num_classes` discrete intervals (default: quartiles of observed recurrence times). The model outputs per-bin hazard probabilities, optimized with a censoring-aware negative log-likelihood loss.

**Limitations:**
- Sensitive to number of discrete bins
- Bin boundaries depend on training cohort's time distribution
- May not generalize well across cohorts with different follow-up patterns

### 2. DCTM - Deep Conditional Transformation Models

DCTM replaces discrete bins with a continuous-time hazard parameterization using Bernstein polynomials. This provides:
- Flexible non-proportional hazards
- No arbitrary binning decisions
- Smooth baseline hazard estimation

**DCTM Parameters:**
- `dctm.basis_features`: Number of Bernstein basis functions (default: 6)
- `dctm.family`: Distribution family - `"logistic"` or `"gompertz"` (default: `"logistic"`)

## Risk Score Computation

For the concordance index (c-index), we need a single scalar to rank patients by risk.

### Discrete-Time Risk Score: Negative Sum of Survival Probabilities

The discrete-time model outputs logits for each time bin. Risk is computed as:

```python
hazards = sigmoid(logits)           # h(t) = P(event at t | survived to t)
surv = cumprod(1 - hazards, dim=1)  # S(t) = P(survive past t)
risk = -sum(surv)                   # negative area under survival curve
```

**Intuition:** The sum of survival probabilities is the expected number of time bins the patient survives. Higher sum → longer expected survival → lower risk. We negate to get higher values = higher risk.

This is equivalent to negative restricted mean survival time (RMST) over the discrete bins.

### DCTM Risk Score: Log Cumulative Hazard at t=1.0

DCTM outputs the log cumulative hazard:
```
z = H(t|X) = baseline(t) + shift(X)
```

We evaluate at `t=1.0` (normalized maximum follow-up time) to get total accumulated risk:
- **Higher z** → higher cumulative hazard → higher risk
- This is analogous to Cox models using the linear predictor

**Why t=1.0?** Times are normalized to `[0, 1]` where `1.0 = max_time` from training. This captures risk over the entire observation window.

**Alternative risk scores** (all give identical c-index since they're monotonic transformations):

| Risk Score | Formula | Notes |
|------------|---------|-------|
| Log cumulative hazard | `H(t=1.0)` | Direct from forward pass, fast |
| Negative TTE | `-model.predict_tte(x)` | Requires root-finding |
| Negative survival | `-exp(-H(t=1.0))` | Bounded [0,1] |

For time-specific risk (e.g., 5-year recurrence):
```python
t_5yr = 5.0 / max_time  # normalize to [0,1]
risk_5yr = model(x, np.array([t_5yr]))
```

## Installation

```bash
# clone with submodules
git clone --recursive https://github.com/clemsgrs/bcr-risk-prediction.git

# install dependencies
pip install -r hipt/requirements.txt
pip install -e hipt/
```

## Usage

### Single-fold training

```bash
# discrete-bin survival
python main.py --config-file config.yaml

# DCTM survival
python main.py --config-file config.yaml dctm.enable=true
```

### Multi-fold cross-validation

Set `data.fold_dir` in your config to point to a directory with `fold-0/`, `fold-1/`, etc., each containing `train.csv`, `tune.csv`, and optionally `test.csv`.

```bash
python main.py --config-file config.yaml data.fold_dir=/path/to/folds/
```

## Data Format

**Features:** We expect a pre-extracted `.pt` files for each case with shape `(num_tiles, embed_dim)`. Set `features_dir` in your config to point to the directory with `patient_001.pt`, `patient_002.pt`, etc.

**Labels CSV:**
```csv
case_id,time_to_bcr,censored,discrete_label
patient_001,24.5,0,2
patient_002,60.0,1,3
```

- `time_to_bcr`: time to event (months/years)
- `censored`: 1 = no event observed, 0 = event occurred
- `discrete_label`: bin index for discrete-time approach

## References

- **HIPT**: [Hierarchical Image Pyramid Transformer](https://github.com/mahmoodlab/HIPT)
- **DCTM**: [Deep Conditional Transformation Models](https://arxiv.org/abs/2210.11366) - "A flexible deep learning framework for survival analysis with medical data" (MICCAI)
