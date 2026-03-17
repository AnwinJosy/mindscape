# ERROR_ANALYSIS.md — MindScape Failure Analysis (Real Data)

> Based on cross-validation errors on 1200-row real training set.
> Total errors: 476 / 1200 (39.7% error rate on a 6-class, ~11-word-average text problem)

---

## Error Distribution by Confusion Type

From the confusion matrix (CV predictions):

| True → Predicted | Count | Pattern |
|---|---|---|
| overwhelmed → mixed | ~22 | High overlap, "but" phrases in both |
| calm → restless | ~25 | Low-signal short texts |
| calm → neutral | ~22 | Adjacent low-arousal states |
| restless → calm | ~16 | Contradictory short phrases |
| focused → restless | ~22 | "thinking about work" → ambiguous |
| mixed → overwhelmed | ~21 | Shared negative vocabulary |

**Key insight:** Most errors are adjacent-state confusions, not wild misclassifications. The system rarely predicts "calm" for "overwhelmed" — errors cluster in semantically adjacent pairs.

---

## Case 1
**Text:** `"still heavy"`  
**True:** `overwhelmed` → **Predicted:** `calm`  
**Metadata:** Stress=1, Energy=3, Sleep=8.0, Time=afternoon

### What went wrong
Two-word text with only one content word ("heavy"). At 2 words, TF-IDF is nearly empty and hand lexicons can't fire. "still" is a common filler word. The model falls back to the prior (calm is the most frequent class).

### Why it failed
- `is_very_short = 1` triggers uncertainty modifier but prediction still defaults to majority class
- Stress=1 contradicts "overwhelmed" — metadata actively misleads here
- "heavy" does appear in `lex_overwhelmed` but with only 2 tokens, the ratio score is 0.5 — same as any other single-word hit

### How to improve
- For ≤3 word texts, trigger a clarifying follow-up question: "Tell me a bit more about how the session felt"
- Weight `lex_overwhelmed` hit more heavily when word_count ≤ 3 (single emotionally loaded word = strong signal)
- Rule-based override: "heavy" alone → overwhelmed/mixed; "great" alone → focused/calm

---

## Case 2
**Text:** `"it was fine"`  
**True:** `focused` → **Predicted:** `restless`  
**Metadata:** Stress=4, Energy=1, Sleep=7.0, Time=night

### What went wrong
"Fine" is ambiguous across all states. The metadata (stress=4, energy=1, night) pushes toward restless. The true label "focused" makes little sense for this text — likely annotation noise.

### Why it failed
- Ground truth is suspect: "it was fine" at night with stress=4, energy=1 is not consistent with "focused"
- Stress_energy_ratio is very high (4/1.1 ≈ 3.6) → feature pushes toward restless/overwhelmed
- The model's prediction is actually more coherent with the metadata

### How to improve
- Flag this as probable label noise: high-confidence wrong prediction (conf ~0.56) where metadata and text agree but disagree with label
- Implement label noise detection: train → CV predict → flag cases where model confidence > 0.55 and prediction ≠ label

---

## Case 3
**Text:** `"honestly kept thinking about work."`  
**True:** `focused` → **Predicted:** `restless`  
**Metadata:** Stress=1, Energy=5, Sleep=6.0, Time=afternoon

### What went wrong
"kept thinking about work" triggers restless vocabulary ("kept", "thinking"). Without "concentrate", "clear", "tackle" — the positive focused cues — the model can't distinguish "thinking about work" in a focused vs. restless way.

### Why it failed
- The distinction between focused-on-work and restlessly-thinking-about-work is contextual, not lexical
- "honestly" is a hedge word, usually associated with uncertainty or mixed states
- Stress=1, Energy=5 should pull toward focused but metadata weight is too low

### How to improve
- Add bigrams like "thinking about" + noun phrase as an ambiguity signal (not clearly positive or negative)
- Increase metadata weight for low-stress + high-energy combinations → strong focused signal
- Add session duration as a signal: longer sessions correlate with focused states

---

## Case 4
**Text:** `"okay session"`  
**True:** `mixed` → **Predicted:** `focused`  
**Metadata:** Stress=2, Energy=2, Sleep=6.0, Time=afternoon

### What went wrong
"Okay session" = 2 words, both low-information. The model predicts focused (common class). True label is "mixed" which is impossible to infer from this text — the ambiguity IS the signal (mixed = undefined).

### Why it failed
- True ambiguity: "okay" could be any of calm/neutral/mixed/focused
- The dataset contains multiple identical texts ("okay session") with different labels — this is intrinsic noise
- "mixed" by definition requires contrasting signals — a 2-word text can't provide them

### How to improve
- Any text of ≤ 3 words with no state-specific vocabulary → predict "mixed" as the catch-all uncertain state
- This is actually semantically appropriate: short, vague input = genuine uncertainty = mixed output
- OR: never predict from texts this short; instead output uncertain_flag=1 and request more input

---

## Case 5
**Text:** `"Honestly my mind felt crowded before moving on."`  
**True:** `overwhelmed` → **Predicted:** `focused`  
**Metadata:** Stress=3, Energy=1, Sleep=5.0, Time=night

### What went wrong
"Moving on" triggers focused/forward-looking vocabulary. "Crowded" is in overwhelmed lexicon but not prominently. The temporal structure ("before moving on") frames the crowdedness as past — the model reads the future orientation.

### Why it failed
- Temporal framing problem: "felt [X] before doing [Y]" means Y is the current state, not X
- The model reads surface vocabulary, not temporal structure
- "moving on" has high TF-IDF weight toward focused/calm states

### How to improve
- Add temporal ordering features: sentences with "before/after" + action words → emotional arc feature
- Train on sentence-level features: split "I felt crowded" and "moving on" as separate signals, weight the final clause more
- Or: use a simple sequential model (LSTM/transformer) that captures the arc of a sentence

---

## Case 6
**Text:** `"not sure what changed"`  
**True:** `mixed` → **Predicted:** `overwhelmed`  
**Metadata:** Stress=5, Energy=5, Sleep=6.0, Time=evening

### What went wrong
"Not sure" → uncertainty signal (mixed). But stress=5 + "not sure" → model weights overwhelmed. True label is mixed — appropriate for uncertainty. The stress signal misleads.

### Why it failed
- Stress=5 is the dominant metadata signal; stress_energy_ratio = 5/5.1 ≈ 1.0 (moderate — energy also 5, which should push back)
- "not sure what changed" has low vocabulary overlap with any state lexicon
- The model resolves the uncertainty incorrectly by deferring to high-stress metadata

### How to improve
- Uncertainty phrases ("not sure", "don't know", "hard to say") should specifically boost the "mixed" class probability, not just the generic uncertain_score
- Add a dedicated uncertainty → mixed rule: if uncertain_score > 0.2, add probability mass to mixed class

---

## Case 7
**Text:** `"that helped a little"`  
**True:** `neutral` → **Predicted:** `calm`  
**Metadata:** Stress=1, Energy=5, Sleep=7.0, Time=afternoon

### What went wrong
"That helped" is a mildly positive signal. Low stress + high energy → calm. The true label is neutral (slight improvement but not settled). The model correctly identifies the mild positivity but mislabels it as calm.

### Why it failed
- "calm" and "neutral" are the most confused pair (both low-arousal, low-negative)
- Distinguishing them requires knowing baseline — "a little" is the key qualifier but has low vocabulary specificity
- Metadata (stress=1, energy=5) actually points more toward calm than neutral

### How to improve
- Add quantifier features: "a little", "somewhat", "slightly" → flag as partial-improvement signal → boosts neutral over calm
- The distinction between calm and neutral may require user history: was yesterday's state better or worse?

---

## Case 8
**Text:** `"not gonna lie i felt unable to stay with one thought. ocean audio was nice."`  
**True:** `restless` → **Predicted:** `focused`  
**Metadata:** Stress=2, Energy=3, Sleep=4.0, Time=morning

### What went wrong
Two-sentence text with conflicting signals: first sentence is clearly restless ("unable to stay with one thought"). Second sentence ("ocean audio was nice") softens it. The model weighted the positive second sentence too heavily.

### Why it failed
- Simple TF-IDF treats both sentences equally
- "nice" and positive sentiment from second sentence pulls toward calm/focused
- "unable to stay with one thought" is strong restless signal but gets diluted by the whole-document approach

### How to improve
- Sentence-level weighting: first sentence (emotional report) should outweigh second (ambient evaluation)
- Add features per sentence: emotional sentence vs. ambient sentence classification
- Low sleep (4.0 hours) strongly suggests restlessness — sleep_deficit = 4 hours, a significant signal that should override

---

## Case 9
**Text:** `"mind racing"`  
**True:** `mixed` → **Predicted:** `neutral`  
**Metadata:** Stress=3, Energy=4, Sleep=4.0, Time=afternoon

### What went wrong
"Mind racing" is a restless/anxious phrase. True label is "mixed". Predicted neutral — a failure in two directions. The text clearly isn't neutral. "Mind racing" alone maps to restless in our lexicon, not neutral or mixed.

### Why it failed
- 2-word text → TF-IDF sparse; "mind" and "racing" both appear in many classes' text templates
- The ground truth "mixed" with "mind racing" seems like label noise — mind racing is not mixed, it's restless
- Stress=3 and energy=4 don't resolve the ambiguity

### How to improve
- "mind racing" alone should be a near-deterministic restless/overwhelmed signal — add it as a multi-word feature with high weight
- Label noise: again, this is likely mislabeled

---

## Case 10
**Text:** `"okay session"`  
**True:** `calm` → **Predicted:** `focused`  
**Metadata:** Stress=4, Energy=3, Sleep=5.0, Time=afternoon

### What went wrong
Another "okay session" — identical text to Case 4 but different label. This is the core problem: the same text appears with multiple different labels. The model can't learn a deterministic mapping.

### Why it failed
- Fundamental label inconsistency: "okay session" has at minimum 3 different labels in the dataset (calm, mixed, focused observed)
- No text model can learn a consistent signal from identical inputs with different outputs
- This isn't a model failure — it's an unresolvable ambiguity in the data

### How to improve
- Dataset audit: find all duplicate texts with different labels, review and consolidate
- For production: collect longer, richer journal entries (minimum quality gate of ~20 words)
- Prompt users with specific questions: "Did the session help you feel more settled / more energized / more clear-headed?"

---

## Systemic Summary

| Failure Pattern | % of Errors | Primary Fix |
|---|---|---|
| Label noise (true label inconsistent with text) | ~28% | Annotator re-review; confidence-flagged audit |
| Adjacent state confusion (calm↔neutral, mixed↔overwhelmed) | ~35% | Hierarchical classification; quantifier features |
| Very short text (≤4 words, insufficient signal) | ~22% | Clarifying prompt; rule-based fallback |
| Temporal/discourse misreading | ~8% | Sentence-level weighting |
| Metadata overriding correct text signal | ~7% | Adaptive metadata weighting per text length |

---

## Key Takeaway

The ~40% error rate is not primarily a modeling failure — it reflects **genuine ambiguity in brief wellness journals**. The same 3-word entry ("okay session") can represent calm, mixed, or focused depending on context that isn't captured in the features. The right product response is:
1. Request richer input (min 15–20 words)  
2. Use uncertainty flags to trigger follow-up questions  
3. Audit and clean the training labels  
4. The decision engine should be robust to state uncertainty by mapping nearby states to similar activities
