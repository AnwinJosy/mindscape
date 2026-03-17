"""
decision_engine.py — What + When + Supportive Message
Tuned for the 6 real classes: calm, focused, restless, neutral, mixed, overwhelmed
"""

# ── Activity → State mapping ─────────────────────────────────────────────────
STATE_ACTIVITIES = {
    "calm":        ["deep_work", "light_planning", "journaling", "yoga"],
    "focused":     ["deep_work", "light_planning", "journaling", "movement"],
    "restless":    ["box_breathing", "movement", "grounding", "yoga"],
    "overwhelmed": ["box_breathing", "grounding", "rest", "sound_therapy"],
    "mixed":       ["journaling", "grounding", "box_breathing", "light_planning"],
    "neutral":     ["light_planning", "journaling", "movement", "deep_work"],
}

INTENSITY_OVERRIDE = {
    5: ["box_breathing", "grounding", "rest"],
    4: ["box_breathing", "grounding", "sound_therapy"],
}

TIME_FILTER = {
    "night":        ["rest", "sound_therapy", "journaling", "box_breathing"],
    "evening":      ["journaling", "sound_therapy", "yoga", "light_planning", "rest", "box_breathing"],
    "early_morning":["movement", "yoga", "box_breathing", "light_planning"],
    "morning":      ["movement", "yoga", "deep_work", "light_planning", "box_breathing", "journaling"],
    "afternoon":    None,
}


def decide_what(state, intensity, stress, energy, time_of_day):
    candidates = STATE_ACTIVITIES.get(state, ["light_planning"])

    # Intensity override
    if intensity in INTENSITY_OVERRIDE:
        candidates = INTENSITY_OVERRIDE[intensity]

    # Time filter
    tf = TIME_FILTER.get(str(time_of_day))
    if tf:
        filtered = [a for a in candidates if a in tf]
        if filtered:
            candidates = filtered

    # High stress → calming
    if stress is not None and not _isnan(stress) and float(stress) >= 4:
        calming = [a for a in candidates
                   if a in ["box_breathing", "grounding", "rest", "sound_therapy", "journaling"]]
        if calming:
            candidates = calming

    # Low energy + late → no movement/deep_work
    if energy is not None and not _isnan(energy) and float(energy) <= 2:
        if str(time_of_day) in ["evening", "night"]:
            candidates = [a for a in candidates if a not in ["movement", "deep_work"]]
            if not candidates:
                candidates = ["rest"]

    return candidates[0]


def decide_when(state, intensity, stress, energy, time_of_day):
    stress_f  = None if (_isnan(stress)  or stress  is None) else float(stress)
    energy_f  = None if (_isnan(energy)  or energy  is None) else float(energy)
    intensity = int(intensity)
    urgent = (intensity >= 4) or (stress_f is not None and stress_f >= 4)

    if state == "overwhelmed" and urgent:
        return "now"
    if state in ["restless", "overwhelmed"] and intensity >= 3:
        return "within_15_min"
    if state == "restless" and urgent:
        return "now"
    if state == "mixed":
        return "within_15_min" if intensity >= 3 else "later_today"
    if state in ["calm", "focused"]:
        if energy_f is not None and energy_f >= 3:
            return "now"
        return "within_15_min"
    if state == "neutral":
        return "later_today"
    if str(time_of_day) == "night":
        return "tonight"
    if urgent:
        return "within_15_min"
    return "later_today"


def supportive_message(state, intensity, what, when):
    urgency_phrase = {
        "now":             "right now",
        "within_15_min":  "in the next few minutes",
        "later_today":    "a little later today",
        "tonight":        "this evening before bed",
        "tomorrow_morning": "tomorrow morning",
    }.get(when, "soon")

    act_desc = {
        "box_breathing":  "a short box-breathing exercise",
        "grounding":      "a quick grounding exercise (5-4-3-2-1 senses)",
        "journaling":     "a few minutes of journaling",
        "sound_therapy":  "a calming sound session",
        "deep_work":      "a focused deep-work session",
        "light_planning": "some light planning or a to-do list",
        "rest":           "genuine rest — no screens",
        "movement":       "light movement or a short walk",
        "yoga":           "a gentle yoga or stretch session",
        "pause":          "a mindful pause",
    }.get(what, what.replace("_", " "))

    opener = {
        "overwhelmed": "You're carrying a lot right now.",
        "restless":    "You seem to have restless energy that needs somewhere to go.",
        "mixed":       "Your feelings seem a bit tangled — that's completely okay.",
        "calm":        "You're in a calm, centered place.",
        "focused":     "Your mind is clear and ready.",
        "neutral":     "You seem in a steady, balanced place.",
    }.get(state, "You seem to have a lot on your mind.")

    intensity_note = ""
    if intensity >= 4:
        intensity_note = " The intensity feels high — let's address it."
    elif intensity <= 2:
        intensity_note = " It seems mild, so gentle action is enough."

    closer = "settle and reset" if state in ["overwhelmed", "restless", "mixed"] else "make the most of this moment"

    return f"{opener}{intensity_note} Try {act_desc} {urgency_phrase} — it'll help you {closer}."


def run_decision_engine(row):
    state     = str(row.get("predicted_state", "neutral"))
    intensity = int(row.get("predicted_intensity", 3))
    stress    = row.get("stress_level")
    energy    = row.get("energy_level")
    time_od   = str(row.get("time_of_day", "morning"))

    what = decide_what(state, intensity, stress, energy, time_od)
    when = decide_when(state, intensity, stress, energy, time_od)
    msg  = supportive_message(state, intensity, what, when)
    return what, when, msg


def _isnan(v):
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return True


if __name__ == "__main__":
    tests = [
        {"predicted_state": "overwhelmed", "predicted_intensity": 5, "stress_level": 5, "energy_level": 1, "time_of_day": "afternoon"},
        {"predicted_state": "focused",     "predicted_intensity": 2, "stress_level": 2, "energy_level": 4, "time_of_day": "morning"},
        {"predicted_state": "restless",    "predicted_intensity": 4, "stress_level": 4, "energy_level": 3, "time_of_day": "evening"},
        {"predicted_state": "mixed",       "predicted_intensity": 3, "stress_level": 3, "energy_level": 2, "time_of_day": "night"},
        {"predicted_state": "calm",        "predicted_intensity": 2, "stress_level": 1, "energy_level": 4, "time_of_day": "morning"},
        {"predicted_state": "neutral",     "predicted_intensity": 2, "stress_level": 2, "energy_level": 3, "time_of_day": "afternoon"},
    ]
    for t in tests:
        w, wh, msg = run_decision_engine(t)
        print(f"[{t['predicted_state']:12s} I={t['predicted_intensity']} @{t['time_of_day']:10s}] → {w:20s} / {wh}")
        print(f"  {msg}\n")
