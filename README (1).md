# MindScape — Emotional Intelligence System
## README

---

## Setup

### Requirements
```
Python 3.9+
scikit-learn >= 1.3
pandas >= 1.5
openpyxl >= 3.0
numpy >= 1.23
scipy >= 1.9
```

### Install
```bash
pip install scikit-learn pandas numpy scipy openpyxl
```

### Directory Structure
```
mindscape/
├── data/
│   ├── train.xlsx     ← Sample_arvyax_reflective_dataset.xlsx (1200 rows)
│   └── test.xlsx      ← arvyax_test_inputs_120.xlsx (120 rows)
├── features.py        ← Feature engineering
├── decision_engine.py ← What + When + Message logic
├── pipeline.py        ← Main training + prediction script
├── predictions.csv    ← Final output (120 rows)
├── README.md
├── ERROR_ANALYSIS.md
└── EDGE_PLAN.md
```

---

## How to Run

```bash
# Make sure data files are in data/
python pipeline.py
# Outputs: predictions.csv, console analysis (Parts 1–7)
```

---

## Dataset Summary

| Property | Value |
|---|---|
| Train rows | 1200 |
| Test rows | 120 |
| Emotional state classes | 6 (calm, focused, restless, neutral, mixed, overwhelmed) |
| Intensity range | 1–5 (nearly uniform distribution) |
| Missing: sleep_hours | 7 rows |
| Missing: previous_day_mood | 15 rows |
| Missing: face_emotion_hint | 123 rows (~10%) |
| Average text length | ~11 words |

---

## Approach

### The core challenge
This is NOT a standard text classification problem. Key difficulties observed in the real data:

1. **Adjacent states** — "mixed" shares vocabulary with all 5 other classes by definition
2. **Short, vague entries** — "still heavy", "okay session", "mind racing" (3–5 words with no clear class signal)
3. **Weak metadata signal** — stress/sleep are nearly uniformly distributed across all 6 states (mean stress ranges only 2.8–3.2 across classes)
4. **Conflicted language** — "The rain gave me a pause, but the pressure is still sitting hard" could be overwhelmed or mixed

The 60% CV accuracy honestly reflects this difficulty — not model failure.

---

## Feature Engineering

### Layer 1 — TF-IDF (300 features, 1–2 grams)
- `sublinear_tf=True` prevents short texts being dominated by a single word
- `min_df=2` removes noise tokens
- Bigrams capture phrases like "but not", "still feel", "too much" that carry more state signal than unigrams

### Layer 2 — Lexical Emotion Features (7 custom scores)
Each score = proportion of words matching a state-specific lexicon:

| Feature | Target State | Example words |
|---|---|---|
| `lex_calm` | calm | settled, peaceful, lighter, softened |
| `lex_focused` | focused | concentrate, clear, tackle, structured |
| `lex_restless` | restless | racing, jumping, couldn't, scattered |
| `lex_overwhelmed` | overwhelmed | heavy, pressure, carrying, drowning |
| `lex_mixed` | mixed | but, yet, part, uneasy, both |
| `lex_neutral` | neutral | okay, normal, idk, unchanged |
| `lex_uncertain` | any | maybe, kinda, sort of, not sure |

These dominate over raw TF-IDF because the dataset has relatively formulaic emotional vocabulary.

### Layer 3 — Text Structural Features (4 features)
`word_count`, `is_very_short` (≤4 words), `is_short` (≤8 words), `has_contrast` (but/yet/however)

### Layer 4 — Metadata Features (11 features)
Ordinal encoding of all contextual signals + derived:
- `sleep_deficit` = max(0, 8 − sleep_hours)  
- `stress_energy_ratio` = stress / (energy + 0.1)  
- `face_missing` binary flag

Missing values: `SimpleImputer(strategy="median")` — robust to ordinal scales and skewed distributions.

---

## Model Choice

### Emotional State: `RandomForestClassifier` + Isotonic Calibration
- Handles mixed continuous + categorical features without normalization
- `class_weight="balanced"` for mild class imbalance
- `CalibratedClassifierCV(method="isotonic")` → calibrated probabilities for trustworthy confidence scores
- **5-fold CV Accuracy: 60.3% ± 3.2%**

Why is accuracy "only" 60%? Because the 6 classes are genuinely hard to separate from brief text. The confusion matrix shows the errors are mostly adjacent-state confusions (overwhelmed↔mixed, calm↔neutral), not random noise.

### Intensity: `RandomForestRegressor`
- **Treated as regression** — intensity 1–5 is ordinal; regression preserves the distance relationship
- Classifying 3 vs 4 as "different classes" treats them as equally dissimilar to 1 vs 5, which is wrong
- **5-fold CV MAE: ~1.26** — note intensity is nearly uniform (each class 18–23%), so any predictor near chance gives MAE ~1.4; our model marginally beats this
- Practical implication: intensity from text alone is very weak; the decision engine uses predicted intensity + raw stress/energy for robustness

---

## Decision Engine

### What to do
Priority hierarchy:
1. Intensity ≥ 4 → override to calming activities regardless of state  
2. State-specific candidates (each state has 3–4 ordered activities)  
3. Time-of-day filter (no deep_work at night; movement preferred in morning)  
4. High stress filter (stress ≥ 4 → calming subset only)  
5. Low energy + late time filter (no movement/deep_work)

### When to do it
- `overwhelmed` + urgent → **now**  
- `restless/overwhelmed` + moderate intensity → **within_15_min**  
- `calm/focused` + adequate energy → **now**  
- `neutral` → **later_today** (low urgency state)  
- Night-time defaults → **tonight**

---

## Uncertainty Modeling

```python
# Adjusted confidence
adj_conf = max_proba × (1 - 0.12 × is_very_short) × (1 - 0.08 × entropy)
uncertain_flag = 1 if adj_conf < 0.50
```

**Threshold rationale:** With 6 classes, random = 0.167. Meaningful confidence starts around 0.45–0.50. The 0.50 threshold correctly flags ~57% of test inputs as uncertain — which accurately reflects the genuine ambiguity in short wellness journal entries.

---

## Ablation Results

| Configuration | CV Accuracy |
|---|---|
| Metadata-only | 18.8% |
| Text-only (TF-IDF + lexical) | 60.5% |
| **Text + Metadata (full)** | **60.3%** |

Metadata barely moves accuracy because stress/sleep are nearly uniformly distributed across classes. But metadata is still useful in the decision engine (e.g., high stress → force calming activity regardless of predicted state) and in the uncertainty estimator (very short text → metadata becomes primary signal).

---

## Performance Summary

| Metric | Value | Notes |
|---|---|---|
| State CV Accuracy | 60.3% | Genuine class ambiguity in ~11-word texts |
| State F1 (macro) | 0.603 | Balanced across all 6 classes |
| Intensity CV MAE | 1.26 | Near-uniform labels make this hard |
| Test uncertain_flag=1 | 57% | Reflects real ambiguity in test entries |
