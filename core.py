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
        os.getenv("REVIEW_CLASSIFIER_HF_REPO_ID", "DeepSee-io/app-high-risk-signals-classifiers"),
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
    return os.getenv("AI_REVIEW_HF_REPO_ID", os.getenv("AIGT_REPO_ID", "DeepSee-io/qwen_adapters_aigt"))


def _env_ai_token() -> Optional[str]:
    return (
        os.getenv("AI_REVIEW_HF_TOKEN")
        or os.getenv("AIGT_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def _env_scam_detector_name() -> str:
    return os.getenv("SCAM_REVIEW_DETECTOR_NAME", "scam").strip() or "scam"


def _env_ai_detector_name() -> str:
    return os.getenv("AI_REVIEW_DETECTOR_NAME", "ai").strip() or "ai"


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


VALID_AI_PRED_LABELS = {
    "human": "Human",
    "mixed": "Mixed",
    "ai": "AI",
}


def _ai_label_and_confidence(ai_pred: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Return only public AI categories or null values.

    Internal statuses such as skipped_short_text, error, or empty_or_failed are
    intentionally not exposed as label categories.
    """
    status = str(ai_pred.get("status") or "empty_or_failed").strip().lower()

    if status in {"skipped_short_text", "error", "empty_or_failed"}:
        return None, None

    ai_probability_raw = ai_pred.get("ai_probability")

    if ai_probability_raw is None or pd.isna(ai_probability_raw):
        return None, None

    ai_probability = float(ai_probability_raw)
    ai_probability = max(0.0, min(1.0, ai_probability))

    pred_short = ai_pred.get("prediction_short")
    if pred_short is None or str(pred_short).strip().lower() in {"", "none", "nan"}:
        pred_label = "AI" if ai_probability >= 0.5 else "Human"
    else:
        pred_label = VALID_AI_PRED_LABELS.get(str(pred_short).strip().lower())

    if pred_label is None:
        return None, None

    if pred_label == "Human":
        return pred_label, float(1.0 - ai_probability)

    return pred_label, float(ai_probability)


WIDE_OUTPUT_COLS = [
    "store",
    "bundle_id",
    "review_id",
    "text",
    "scam_pred_label",
    "scam_pred_confidence",
    "ai_pred_label",
    "ai_pred_confidence",
    "ai_probability",
    "fraction_ai",
    "n_tokens",
]

LONG_OUTPUT_COLS = [
    "store",
    "bundle_id",
    "review_id",
    "text",
    "detector",
    "label",
    "confidence",
    "score",
    "fraction_ai",
    "n_tokens",
]

VALID_OUTPUT_FORMATS = {"long", "wide"}


def normalize_output_format(output_format: str = "long") -> str:
    value = str(output_format or "long").strip().lower()

    if value not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}; got {output_format!r}."
        )

    return value


def _as_float_or_none(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _build_wide_result(
    record: Dict[str, Any],
    scam_row: Dict[str, Any],
    ai_pred: Dict[str, Any],
) -> Dict[str, Any]:
    ai_pred_label, ai_pred_confidence = _ai_label_and_confidence(ai_pred)

    return {
        "store": _coerce_text(record.get("store")).lower(),
        "bundle_id": _coerce_text(record.get("bundle_id")),
        "review_id": _coerce_text(record.get("review_id")),
        "text": _coerce_text(record.get("text")),
        "scam_pred_label": scam_row.get("pred_label"),
        "scam_pred_confidence": _as_float_or_none(scam_row.get("pred_confidence")),
        "ai_pred_label": ai_pred_label,
        "ai_pred_confidence": ai_pred_confidence,
        "ai_probability": _as_float_or_none(ai_pred.get("ai_probability")),
        "fraction_ai": _as_float_or_none(ai_pred.get("fraction_ai")),
        "n_tokens": _as_int_or_none(ai_pred.get("n_tokens")),
    }


def _wide_result_to_long_rows(wide_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    identity = {
        "store": wide_row.get("store"),
        "bundle_id": wide_row.get("bundle_id"),
        "review_id": wide_row.get("review_id"),
        "text": wide_row.get("text"),
    }

    return [
        {
            **identity,
            "detector": _env_scam_detector_name(),
            "label": wide_row.get("scam_pred_label"),
            "confidence": wide_row.get("scam_pred_confidence"),
            "score": None,
            "fraction_ai": None,
            "n_tokens": None,
        },
        {
            **identity,
            "detector": _env_ai_detector_name(),
            "label": wide_row.get("ai_pred_label"),
            "confidence": wide_row.get("ai_pred_confidence"),
            "score": wide_row.get("ai_probability"),
            "fraction_ai": wide_row.get("fraction_ai"),
            "n_tokens": wide_row.get("n_tokens"),
        },
    ]


def predict_records(
    records: List[Dict[str, Any]],
    output_format: str = "long",
) -> List[Dict[str, Any]]:
    """Score review records with both detectors.

    Args:
        records: Input review records. Each record should include store,
            bundle_id, review_id, and text; language fields are optional.
        output_format: ``"long"`` by default, yielding one row per detector
            per review. Use ``"wide"`` for the former one-row-per-review
            compatibility output.

    Returns:
        For ``output_format="long"``, returns exactly two rows per input review:
        one scam-detector row and one AI-detector row. Detector identifiers are
        configured through SCAM_REVIEW_DETECTOR_NAME and AI_REVIEW_DETECTOR_NAME
        and default to ``scam`` and ``ai``. For reviews that are too short or
        otherwise not scorable by the AI detector, the AI row is retained with
        null label/confidence/score fields.
    """
    output_format = normalize_output_format(output_format)

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

    wide_rows: List[Dict[str, Any]] = []

    for i, record in enumerate(records):
        scam_row = scam_scored.iloc[i].to_dict()
        ai_pred = ai_preds[i] if i < len(ai_preds) else {"status": "empty_or_failed"}
        wide_rows.append(_build_wide_result(record, scam_row, ai_pred))

    if os.getenv("REVIEW_DETECTORS_CACHE_POLICY", "keep") == "unload_after_call":
        unload_cached_classifiers()

    if output_format == "wide":
        return wide_rows

    long_rows: List[Dict[str, Any]] = []
    for wide_row in wide_rows:
        long_rows.extend(_wide_result_to_long_rows(wide_row))

    return long_rows
