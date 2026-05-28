from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai_classifier import TextClassifier, TextInferConfig
from review_classifier.config import env_bool, get_hf_token, normalize_subfolder
from review_classifier.inference import ReviewScamClassifier


_SCAM_CLASSIFIER_CACHE: Dict[Tuple[Any, ...], ReviewScamClassifier] = {}
_AI_CLASSIFIER_CACHE: Dict[Tuple[Any, ...], TextClassifier] = {}


def _env_scam_model_id() -> str:
    return os.getenv(
        "SCAM_REVIEW_HF_REPO_ID",
        os.getenv("REVIEW_CLASSIFIER_HF_REPO_ID", "Trinotrotolueno/review-scam-adapters"),
    )


def _env_scam_subfolder() -> Optional[str]:
    return normalize_subfolder(
        os.getenv(
            "SCAM_REVIEW_SUBFOLDER",
            os.getenv("REVIEW_CLASSIFIER_SUBFOLDER", "scam_reviews"),
        )
    )


def _env_scam_token() -> Optional[str]:
    return (
        os.getenv("SCAM_REVIEW_HF_TOKEN")
        or os.getenv("REVIEW_CLASSIFIER_HF_TOKEN")
        or get_hf_token()
        or None
    )


def _env_bool_any(*names: str, default: str = "0") -> bool:
    for name in names:
        if os.getenv(name) is not None:
            return env_bool(name, default)
    return str(default).strip().lower() in {"1", "true", "yes", "y"}


def get_scam_classifier() -> ReviewScamClassifier:
    model_id = _env_scam_model_id()
    subfolder = _env_scam_subfolder()
    token = _env_scam_token()
    max_length = int(os.getenv("SCAM_REVIEW_MAX_LENGTH", os.getenv("REVIEW_CLASSIFIER_MAX_LENGTH", "512")))
    load_in_4bit = not _env_bool_any("SCAM_REVIEW_NO_4BIT", "REVIEW_CLASSIFIER_NO_4BIT", default = "0")
    apply_one_word_mapping = _env_bool_any("SCAM_REVIEW_APPLY_ONE_WORD_MAPPING", default = "1")
    force_unknown_single_word_clean = _env_bool_any("SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN", default = "1")

    key = (
        model_id,
        subfolder,
        bool(token),
        max_length,
        load_in_4bit,
        apply_one_word_mapping,
        force_unknown_single_word_clean,
    )

    if key not in _SCAM_CLASSIFIER_CACHE:
        _SCAM_CLASSIFIER_CACHE[key] = ReviewScamClassifier.from_hf(
            model_id = model_id,
            subfolder = subfolder,
            token = token,
            max_length = max_length,
            load_in_4bit = load_in_4bit,
            apply_one_word_mapping = apply_one_word_mapping,
            force_unknown_single_word_clean = force_unknown_single_word_clean,
        )

    return _SCAM_CLASSIFIER_CACHE[key]


def _env_ai_repo_id() -> str:
    return os.getenv("AI_REVIEW_HF_REPO_ID", os.getenv("AIGT_REPO_ID", "Trinotrotolueno/aigt-loras"))


def _env_ai_token() -> Optional[str]:
    return (
        os.getenv("AI_REVIEW_HF_TOKEN")
        or os.getenv("AIGT_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def get_ai_classifier() -> TextClassifier:
    cfg = TextInferConfig(
        repo_id = _env_ai_repo_id(),
        subdir_by_lang_json = os.getenv(
            "AI_REVIEW_SUBDIR_BY_LANG_JSON",
            os.getenv("AIGT_SUBDIR_BY_LANG_JSON", '{"en":"reviews/best"}'),
        ),
        revision = os.getenv("AI_REVIEW_REVISION", os.getenv("AIGT_REVISION", "")) or None,
        hf_token = _env_ai_token(),
        device = os.getenv("AI_REVIEW_DEVICE", os.getenv("AIGT_DEVICE", "cuda")),
        cache_policy = os.getenv("AI_REVIEW_CACHE_POLICY", os.getenv("AIGT_CACHE_POLICY", "keep")),
        max_len = int(os.getenv("AI_REVIEW_MAX_LEN", os.getenv("AIGT_MAX_LEN", "500"))),
        batch_size = int(os.getenv("AI_REVIEW_BATCH_SIZE", os.getenv("AIGT_BATCH_SIZE", "16"))),
        window_ai_threshold = float(os.getenv("AI_REVIEW_WINDOW_AI_THRESHOLD", os.getenv("AIGT_WINDOW_AI_THRESHOLD", "0.5"))),
        prefer_bf16 = os.getenv("AI_REVIEW_PREFER_BF16", os.getenv("AIGT_PREFER_BF16", "1")) not in {"0", "false", "False"},
        min_words = int(os.getenv("AI_REVIEW_MIN_WORDS", os.getenv("AIGT_MIN_WORDS", "10"))),
    )

    key = (
        cfg.repo_id,
        cfg.subdir_by_lang_json,
        cfg.revision,
        bool(cfg.hf_token),
        cfg.device,
        cfg.cache_policy,
        cfg.max_len,
        cfg.batch_size,
        cfg.window_ai_threshold,
        cfg.prefer_bf16,
        cfg.min_words,
    )

    if key not in _AI_CLASSIFIER_CACHE:
        clf = TextClassifier(cfg = cfg)
        clf.load_models()
        _AI_CLASSIFIER_CACHE[key] = clf

    return _AI_CLASSIFIER_CACHE[key]


def unload_cached_classifiers():
    for clf in _SCAM_CLASSIFIER_CACHE.values():
        clf.unload()
    _SCAM_CLASSIFIER_CACHE.clear()

    _AI_CLASSIFIER_CACHE.clear()


def cache_info() -> Dict[str, Any]:
    return {
        "n_cached_scam_classifiers": len(_SCAM_CLASSIFIER_CACHE),
        "n_cached_ai_classifiers": len(_AI_CLASSIFIER_CACHE),
        "scam_classifiers": [clf.get_model_info() for clf in _SCAM_CLASSIFIER_CACHE.values()],
        "ai_classifiers": [clf.get_model_info() for clf in _AI_CLASSIFIER_CACHE.values()],
    }


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_lang(value: Any) -> str:
    if value is None or pd.isna(value):
        return "en"

    lang = str(value).lower().strip()
    if not lang or lang in {"nan", "none", "null"}:
        return "en"

    return lang.split("-")[0].split("_")[0] or "en"


def _ai_label_and_confidence(ai_pred: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    status = str(ai_pred.get("status") or "empty_or_failed")

    if status == "skipped_short_text":
        return "skipped_short_text", None

    if status == "error":
        return "error", None

    ai_probability_raw = ai_pred.get("ai_probability")

    if ai_probability_raw is None or pd.isna(ai_probability_raw):
        return "empty_or_failed", None

    ai_probability = float(ai_probability_raw)
    ai_probability = max(0.0, min(1.0, ai_probability))

    pred_short = ai_pred.get("prediction_short")
    if pred_short is None or str(pred_short).strip() in {"", "None", "nan"}:
        pred_label = "AI" if ai_probability >= 0.5 else "Human"
    else:
        pred_label = str(pred_short).strip()

    if pred_label.lower() == "human":
        return pred_label, float(1.0 - ai_probability)

    return pred_label, float(ai_probability)


def predict_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []

    df = pd.DataFrame(records).copy()

    if "text" not in df.columns:
        df["text"] = ""

    if "prediction_id" not in df.columns:
        df["prediction_id"] = [uuid.uuid4().hex for _ in range(len(df))]

    df["text"] = df["text"].map(_coerce_text)

    scam_clf = get_scam_classifier()
    scam_batch_size = int(os.getenv("SCAM_REVIEW_BATCH_SIZE", os.getenv("BATCH_SIZE", "8")))

    scam_scored = scam_clf.predict_df(
        df = df,
        text_col = "text",
        batch_size = scam_batch_size,
        prediction_id_col = "prediction_id",
        overwrite_prediction_id = False,
        include_probabilities = False,
        include_label_ids = False,
    )

    ai_clf = get_ai_classifier()
    langs = []
    for _, row in df.iterrows():
        langs.append(
            _coerce_lang(
                row.get("lang")
                or row.get("language")
                or row.get("language_code")
            )
        )

    ai_preds = ai_clf.classify_texts_batch(
        texts = df["text"].tolist(),
        langs = langs,
        prediction_ids = df["prediction_id"].astype(str).tolist(),
    )

    out: List[Dict[str, Any]] = []

    for i, record in enumerate(records):
        scam_row = scam_scored.iloc[i].to_dict()
        ai_pred = ai_preds[i] if i < len(ai_preds) else {"status": "empty_or_failed"}
        ai_pred_label, ai_pred_confidence = _ai_label_and_confidence(ai_pred)

        ai_probability = ai_pred.get("ai_probability")
        fraction_ai = ai_pred.get("fraction_ai")
        n_tokens = ai_pred.get("n_tokens")

        out.append({
            "store": _coerce_text(record.get("store")).lower(),
            "bundle_id": _coerce_text(record.get("bundle_id")),
            "review_id": _coerce_text(record.get("review_id")),
            "text": _coerce_text(record.get("text")),
            "scam_pred_label": scam_row.get("pred_label"),
            "scam_pred_confidence": scam_row.get("pred_confidence"),
            "ai_pred_label": ai_pred_label,
            "ai_pred_confidence": ai_pred_confidence,
            "ai_probability": None if ai_probability is None or pd.isna(ai_probability) else float(ai_probability),
            "fraction_ai": None if fraction_ai is None or pd.isna(fraction_ai) else float(fraction_ai),
            "n_tokens": None if n_tokens is None or pd.isna(n_tokens) else int(n_tokens),
        })

    if os.getenv("REVIEW_DETECTORS_CACHE_POLICY", "keep") == "unload_after_call":
        unload_cached_classifiers()

    return out
