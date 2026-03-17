"""
features.py — Feature engineering for MindScape (real dataset version)
Two-layer text + metadata features, with proper imputation and encoding.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
import re

# ── Emotion-aware lexicon (derived from real data vocabulary) ────────────────
CALM_WORDS = {
    "calm", "settled", "peaceful", "lighter", "softened", "gentle", "quiet",
    "still", "relaxed", "grounded", "slowed", "breathe", "ease", "restful",
    "shoulders", "tension", "release", "serene", "flow", "mellow"
}
FOCUSED_WORDS = {
    "concentrate", "focused", "clear", "tackle", "steps", "task", "plan",
    "organized", "productive", "sharp", "ready", "direction", "next",
    "to-do", "list", "structured", "precision", "goal", "work"
}
RESTLESS_WORDS = {
    "racing", "jumping", "buzz", "fidgety", "unsettled", "distracted",
    "keep", "kept", "couldn't", "still", "everywhere", "antsy", "restless",
    "scattered", "can't", "cannot", "all", "once", "switching"
}
OVERWHELMED_WORDS = {
    "heavy", "pressure", "too", "much", "carrying", "hard", "drowning",
    "buried", "exhausted", "swamped", "tight", "chest", "breath", "stuck",
    "piling", "overload", "breaking", "collapse", "weight", "suffocating"
}
MIXED_WORDS = {
    "but", "yet", "however", "though", "part", "uneasy", "better", "worse",
    "and", "still", "both", "feel", "don't", "not", "unsure", "half",
    "mixed", "conflict", "complicated", "ambivalent"
}
NEUTRAL_WORDS = {
    "okay", "okay", "normal", "alright", "fine", "neutral", "average",
    "idk", "maybe", "shifted", "much", "same", "unchanged", "ordinary",
    "whatever", "nothing", "different", "just", "bit", "aware"
}
UNCERTAINTY_PHRASES = [
    "i don't know", "not sure", "kind of", "sort of", "a bit", "hard to say",
    "maybe", "i guess", "kinda", "idk", "i think", "feels like", "somewhat"
]


def hand_text_features(texts):
    """11 hand-crafted features capturing sentiment, uncertainty, length signals."""
    out = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            text = "empty"
        t = text.lower()
        words = re.findall(r"\b\w+\b", t)
        wset = set(words)
        n = max(len(words), 1)

        calm_s     = len(wset & CALM_WORDS) / n
        focused_s  = len(wset & FOCUSED_WORDS) / n
        restless_s = len(wset & RESTLESS_WORDS) / n
        overwhelm_s= len(wset & OVERWHELMED_WORDS) / n
        mixed_s    = len(wset & MIXED_WORDS) / n
        neutral_s  = len(wset & NEUTRAL_WORDS) / n
        uncertain_s= sum(1 for p in UNCERTAINTY_PHRASES if p in t) / n

        wc = len(words)
        very_short = int(wc <= 4)
        short      = int(wc <= 8)
        has_contrast = int(any(w in t for w in ["but", "yet", "however", "though", "still"]))

        out.append([
            calm_s, focused_s, restless_s, overwhelm_s,
            mixed_s, neutral_s, uncertain_s,
            wc, very_short, short, has_contrast
        ])
    return np.array(out, dtype=float)


HAND_FEATURE_NAMES = [
    "lex_calm", "lex_focused", "lex_restless", "lex_overwhelmed",
    "lex_mixed", "lex_neutral", "lex_uncertain",
    "word_count", "is_very_short", "is_short", "has_contrast"
]

# ── Metadata encoding ────────────────────────────────────────────────────────
TIME_MAP = {"early_morning": 0, "morning": 1, "afternoon": 2, "evening": 3, "night": 4}
MOOD_MAP = {"calm": 0, "focused": 1, "neutral": 2, "mixed": 3, "restless": 4, "overwhelmed": 5}
FACE_MAP = {"calm_face": 0, "happy_face": 1, "neutral_face": 2,
            "tired_face": 3, "tense_face": 4, "none": 2}  # none → neutral
QUALITY_MAP = {"clear": 2, "vague": 0, "conflicted": 1}


def metadata_features(df):
    rows = []
    for _, r in df.iterrows():
        sleep   = r.get("sleep_hours", np.nan)
        energy  = r.get("energy_level", np.nan)
        stress  = r.get("stress_level", np.nan)
        dur     = r.get("duration_min", np.nan)
        tod     = TIME_MAP.get(str(r.get("time_of_day", "")), np.nan)
        mood    = MOOD_MAP.get(str(r.get("previous_day_mood", "")), np.nan)
        face    = FACE_MAP.get(str(r.get("face_emotion_hint", "")), np.nan)
        quality = QUALITY_MAP.get(str(r.get("reflection_quality", "")), np.nan)

        sleep_def = (8.0 - float(sleep)) if not np.isnan(float(sleep) if sleep == sleep else np.nan) else np.nan
        try:
            s_e_ratio = float(stress) / (float(energy) + 0.1)
        except Exception:
            s_e_ratio = np.nan

        face_missing = int(pd.isna(r.get("face_emotion_hint")) or
                          str(r.get("face_emotion_hint", "")) in ["nan", "none", ""])

        rows.append([
            float(sleep) if sleep == sleep else np.nan,
            float(energy) if energy == energy else np.nan,
            float(stress) if stress == stress else np.nan,
            float(dur)    if dur == dur else np.nan,
            tod, mood, face, quality,
            sleep_def if not (sleep_def != sleep_def) else np.nan,
            s_e_ratio if not (s_e_ratio != s_e_ratio) else np.nan,
            face_missing,
        ])

    arr = np.array(rows, dtype=float)
    # Fix NaN from failed conversions
    arr = np.where(np.isinf(arr), np.nan, arr)

    imputer = SimpleImputer(strategy="median")
    arr_imp = imputer.fit_transform(arr)
    return arr_imp, imputer


META_FEATURE_NAMES = [
    "sleep_hours", "energy_level", "stress_level", "duration_min",
    "time_of_day", "prev_mood", "face_hint", "reflection_quality",
    "sleep_deficit", "stress_energy_ratio", "face_missing"
]


# ── Combined feature builder ──────────────────────────────────────────────────
def build_features(train_df, test_df=None, max_tfidf=300):
    """
    Fits on train_df. If test_df given, transforms it with same fitted objects.
    Returns: X_train, X_test (or None), feature_names, tfidf_vec, meta_imputer
    """
    texts_train = train_df["journal_text"].fillna("").tolist()

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_tfidf,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    X_tfidf_train = tfidf.fit_transform(texts_train).toarray()

    # Hand features
    X_hand_train = hand_text_features(texts_train)

    # Metadata
    X_meta_train, meta_imp = metadata_features(train_df)

    X_train = np.hstack([X_tfidf_train, X_hand_train, X_meta_train])

    tfidf_names = [f"tfidf_{t}" for t in tfidf.get_feature_names_out()]
    feature_names = tfidf_names + HAND_FEATURE_NAMES + META_FEATURE_NAMES

    X_test = None
    if test_df is not None:
        texts_test = test_df["journal_text"].fillna("").tolist()
        X_tfidf_test = tfidf.transform(texts_test).toarray()
        X_hand_test  = hand_text_features(texts_test)

        # Metadata for test — reuse imputer fitted on train
        rows_test = []
        for _, r in test_df.iterrows():
            sleep   = r.get("sleep_hours", np.nan)
            energy  = r.get("energy_level", np.nan)
            stress  = r.get("stress_level", np.nan)
            dur     = r.get("duration_min", np.nan)
            tod     = TIME_MAP.get(str(r.get("time_of_day", "")), np.nan)
            mood    = MOOD_MAP.get(str(r.get("previous_day_mood", "")), np.nan)
            face    = FACE_MAP.get(str(r.get("face_emotion_hint", "")), np.nan)
            quality = QUALITY_MAP.get(str(r.get("reflection_quality", "")), np.nan)
            try:
                sleep_def = 8.0 - float(sleep)
            except Exception:
                sleep_def = np.nan
            try:
                s_e_ratio = float(stress) / (float(energy) + 0.1)
            except Exception:
                s_e_ratio = np.nan
            face_missing = int(pd.isna(r.get("face_emotion_hint")) or
                              str(r.get("face_emotion_hint","")) in ["nan","none",""])
            rows_test.append([
                float(sleep) if sleep==sleep else np.nan,
                float(energy) if energy==energy else np.nan,
                float(stress) if stress==stress else np.nan,
                float(dur) if dur==dur else np.nan,
                tod, mood, face, quality,
                sleep_def, s_e_ratio, face_missing,
            ])
        arr_test = np.array(rows_test, dtype=float)
        arr_test = np.where(np.isinf(arr_test), np.nan, arr_test)
        X_meta_test = meta_imp.transform(arr_test)
        X_test = np.hstack([X_tfidf_test, X_hand_test, X_meta_test])

    return X_train, X_test, feature_names, tfidf, meta_imp


if __name__ == "__main__":
    import pandas as pd
    train = pd.read_excel(r"C:\Users\HP\Downloads\mindscape\data\Sample_arvyax_reflective_dataset.xlsx")
    test  = pd.read_excel(r"C:\Users\HP\Downloads\mindscape\data\arvyax_test_inputs_120.xlsx")
    X_tr, X_te, names, vec, imp = build_features(train, test)
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
    print(f"Sample feature names: {names[:5]} ... {names[-5:]}")
