# EDGE_PLAN.md — On-Device / Edge Deployment Plan

---

## Part 8: Running MindScape On-Device

### Why On-Device Matters Here
MindScape handles personal wellness journal text — intimate, sensitive data. Sending "felt heavy today, couldn't stop thinking about losing my job" to a cloud API is a real privacy risk users notice. On-device inference is not just a performance optimization; it's a trust feature.

Additionally, immersive sessions (forest, ocean, rain) are often in offline environments — parks, beaches, mountain cabins. The system must work without connectivity.

---

## Architecture for Mobile

```
User completes immersion session
           │
           ▼
   ┌─────────────────────┐
   │  Text Preprocessing │  ← Tokenize, lowercase, strip
   │  (Kotlin / Swift)   │  ~1ms
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   TF-IDF Transform  │  ← Pre-built vocabulary hash (~15 KB)
   │                     │  ~3ms
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Lexical Scorer     │  ← 6 state lexicons, hardcoded (~5 KB)
   │                     │  ~1ms
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   RF Classifier     │  ← ONNX / TFLite (~2–4 MB)
   │   (50 trees, d=8)   │  ~10–20ms
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Decision Engine    │  ← Pure rule logic, compiled
   │                     │  <1ms
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Message Templates  │  ← Pre-written strings (~50 KB)
   │                     │  <1ms
   └─────────────────────┘

Total: ~25ms end-to-end on a mid-range Android phone (2022)
```

---

## Model Size Optimization

### Current model: RandomForest (300 trees, depth=15)
- Serialized: ~25–40 MB (too large for mobile)

### Optimization path to production:

**Step 1 — Reduce trees and depth**
```python
# From 300/15 → 50/8
# Accuracy drop: ~60.3% → ~57.5% (acceptable)
model_small = RandomForestClassifier(n_estimators=50, max_depth=8, ...)
```

**Step 2 — Export to ONNX**
```python
from skl2onnx import convert_sklearn, Int64TensorType
onnx_model = convert_sklearn(model_small, "rf", [("input", Int64TensorType([None, 322]))])
# Size: ~3–5 MB
```

**Step 3 — Convert to TFLite or Core ML**
- Android: TFLite FlatBuffer (~1.5–3 MB)
- iOS: Core ML via coremltools (~2–4 MB)

**Step 4 — Quantize to INT8**
```
Post-training quantization: ~4x size reduction
Final: ~0.5–1 MB model
Accuracy drop: ~1–2% (negligible)
```

**Total on-device footprint:**
| Component | Size |
|---|---|
| Model (INT8 quantized) | ~1 MB |
| TF-IDF vocabulary | ~15 KB |
| Lexicons | ~5 KB |
| Message templates | ~50 KB |
| Runtime (TFLite) | ~300 KB (shared library) |
| **Total** | **~1.5 MB** |

---

## Latency Budget

| Step | Time (mid-range phone) |
|---|---|
| Text preprocessing | 1 ms |
| TF-IDF vectorization | 3 ms |
| Lexical scoring | 1 ms |
| RF inference (TFLite) | 15–20 ms |
| Decision engine | <1 ms |
| Message generation | <1 ms |
| **Total** | **~22 ms** |

This is well under 100ms (imperceptible to users) and even under the 50ms "fast" threshold.

---

## Feature Reduction for Edge

The current model uses 322 features. On device, we can reduce this:

1. **Drop low-importance TF-IDF tokens** (keep top-100 by importance score): 300 → 100 features
2. **Keep all 11 hand-crafted + 11 metadata features**: these are cheap to compute and high signal
3. Final feature count: **~122 features** (vs 322)
4. Retrain on this reduced set: accuracy drop ~1%

This also speeds up TF-IDF at inference time (smaller vocabulary lookup).

---

## Privacy Architecture

- Journal text is **processed locally, never transmitted**
- Model weights are bundled at install time (no network call for inference)
- Metadata (sleep, stress) comes from HealthKit/Google Health — user controls permissions
- Optional: encrypt journal entries with user's device key before storing locally
- If user opts into anonymized telemetry: send only `{predicted_state, confidence, uncertain_flag}` — never the journal text

---

## Offline Storage

```sql
-- SQLite schema
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    timestamp INTEGER,
    ambience_type TEXT,
    predicted_state TEXT,
    predicted_intensity INTEGER,
    confidence REAL,
    uncertain_flag INTEGER,
    what_to_do TEXT,
    when_to_do TEXT,
    completed INTEGER DEFAULT 0
);

CREATE TABLE sync_queue (
    session_id TEXT,
    pending_since INTEGER
);
```

When internet available: sync aggregated stats (no raw text) to a backend for model improvement with user consent.

---

## Model Update Strategy

```
v1.0 model on device
      │
      ▼
Background download when on WiFi + charging
(delta model weights only, ~500 KB)
      │
      ▼
Shadow mode: run both v1.0 + v1.1 silently
Compare confidence distributions over 7 days
      │
      ▼
If v1.1 mean confidence > v1.0 → auto-promote
If v1.1 confidence lower → rollback, alert team
```

---

## Part 9: Robustness Details

### Very short text ("ok", "fine", "still heavy")
The dataset has real examples of 2–4 word entries. Strategy:

1. `is_very_short` flag → confidence adjusted down by 12%
2. If adjusted confidence < 0.50 → `uncertain_flag = 1`
3. **UI action:** show soft recommendation with caveat: *"We're not fully sure how you're feeling — here's a gentle suggestion based on your session context"*
4. **Metadata fallback:** for word_count ≤ 3, increase weight of stress/energy/time features in the decision engine (even if text model is uncertain)
5. **Clarifying prompt:** optionally ask "Would you like to add a few more words about how you felt?" to improve next-session predictions

### Missing values
- 10% of `face_emotion_hint` and ~1.2% of `sleep_hours` missing in training
- `SimpleImputer(strategy="median")` fills at inference time with training-set medians
- `face_missing` binary flag explicitly signals to the model that this channel is absent
- The model was trained on data with real missingness — it learned to handle it

Worst case: all metadata missing → model falls back to text-only features, which provide ~60% of the signal anyway.

### Contradictory inputs
Example: Text = "felt so energized and ready to go!" but stress=5, sleep=3.0, energy_level=1

Handling strategy:
1. If top-2 class probabilities are close (|p1 - p2| < 0.15) → `uncertain_flag = 1`
2. Decision engine uses **both** predicted state AND raw metadata:
   - If predicted state = focused/energized BUT stress ≥ 4 → override activity to box_breathing/grounding
   - This respects "what the user wrote" for state labeling but protects against harm in the decision
3. Supportive message acknowledges the conflict: *"Sometimes our words don't capture the full picture. Your stress level today suggests you could use something grounding before diving in."*
4. For model improvement: log contradictory cases (text-state vs metadata-expectation gap > 2 sigma) as training signal for a future ensemble that explicitly models text-metadata disagreement

---

## Tradeoffs Summary

| Factor | On-Device | Cloud (GPT-class) |
|---|---|---|
| Latency | ~22 ms | ~800–2000 ms |
| Privacy | Full | Requires trust |
| Offline | Yes | No |
| Accuracy | ~58–60% | ~75–85% (with longer prompts) |
| Model size | ~1.5 MB | ~100B params |
| Cost/inference | ~$0 | ~$0.01–0.05 |
| Update cycle | Weekly (background) | Instant |

**Verdict:** For a personal wellness app, on-device is the right default. The accuracy gap (~15–20% vs a hosted LLM) can be partially recovered by:
1. Better data collection (longer entries, structured prompts)
2. User-specific fine-tuning (personalized model after 30+ sessions)
3. Hybrid: on-device for real-time, optional cloud sync for model improvement with consent
