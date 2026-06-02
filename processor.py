import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List

from core import LONG_OUTPUT_COLS, WIDE_OUTPUT_COLS, normalize_output_format, predict_records
from processor_config import (
    AI_REVIEW_BATCH_SIZE,
    AI_REVIEW_CACHE_POLICY,
    AI_REVIEW_DEVICE,
    AI_REVIEW_HF_REPO_ID,
    AI_REVIEW_HF_TOKEN,
    AI_REVIEW_MAX_LEN,
    AI_REVIEW_MIN_WORDS,
    AI_REVIEW_PREFER_BF16,
    AI_REVIEW_REVISION,
    AI_REVIEW_SUBDIR_BY_LANG_JSON,
    AI_REVIEW_WINDOW_AI_THRESHOLD,
    BATCH_SIZE,
    EMPTY_QUEUE_SLEEP_SECONDS,
    REVIEW_DETECTORS_DUPLICATE_REVIEW_ID_POLICY,
    REVIEW_DETECTORS_REQUIRE_REVIEW_ID,
    REVIEW_DETECTORS_OUTPUT_FORMAT,
    SCAM_REVIEW_APPLY_ONE_WORD_MAPPING,
    SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN,
    SCAM_REVIEW_HF_REPO_ID,
    SCAM_REVIEW_HF_TOKEN,
    SCAM_REVIEW_MAX_LENGTH,
    SCAM_REVIEW_NO_4BIT,
    SCAM_REVIEW_SUBFOLDER,
)
from processor_utils import pop, push


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


QUEUE_OUTPUT_COLS_BY_FORMAT = {
    "long": LONG_OUTPUT_COLS,
    "wide": WIDE_OUTPUT_COLS,
}



def _env_bool(value: str) -> bool:
    return str(value).strip() in {"1", "true", "TRUE", "yes", "YES", "y", "Y"}


def _set_model_env():
    os.environ["SCAM_REVIEW_HF_REPO_ID"] = SCAM_REVIEW_HF_REPO_ID
    os.environ["SCAM_REVIEW_SUBFOLDER"] = SCAM_REVIEW_SUBFOLDER
    os.environ["SCAM_REVIEW_MAX_LENGTH"] = str(SCAM_REVIEW_MAX_LENGTH)
    os.environ["SCAM_REVIEW_NO_4BIT"] = SCAM_REVIEW_NO_4BIT
    os.environ["SCAM_REVIEW_APPLY_ONE_WORD_MAPPING"] = SCAM_REVIEW_APPLY_ONE_WORD_MAPPING
    os.environ["SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN"] = SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN

    if SCAM_REVIEW_HF_TOKEN:
        os.environ["SCAM_REVIEW_HF_TOKEN"] = SCAM_REVIEW_HF_TOKEN

    os.environ["AI_REVIEW_HF_REPO_ID"] = AI_REVIEW_HF_REPO_ID
    os.environ["AI_REVIEW_SUBDIR_BY_LANG_JSON"] = AI_REVIEW_SUBDIR_BY_LANG_JSON
    os.environ["AI_REVIEW_CACHE_POLICY"] = AI_REVIEW_CACHE_POLICY
    os.environ["AI_REVIEW_DEVICE"] = AI_REVIEW_DEVICE
    os.environ["AI_REVIEW_MAX_LEN"] = str(AI_REVIEW_MAX_LEN)
    os.environ["AI_REVIEW_BATCH_SIZE"] = str(AI_REVIEW_BATCH_SIZE)
    os.environ["AI_REVIEW_WINDOW_AI_THRESHOLD"] = str(AI_REVIEW_WINDOW_AI_THRESHOLD)
    os.environ["AI_REVIEW_PREFER_BF16"] = "1" if AI_REVIEW_PREFER_BF16 else "0"
    os.environ["AI_REVIEW_MIN_WORDS"] = str(AI_REVIEW_MIN_WORDS)

    if AI_REVIEW_REVISION:
        os.environ["AI_REVIEW_REVISION"] = AI_REVIEW_REVISION

    if AI_REVIEW_HF_TOKEN:
        os.environ["AI_REVIEW_HF_TOKEN"] = AI_REVIEW_HF_TOKEN


def _extract_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in job and isinstance(job["payload"], dict):
        return job["payload"]

    if "data" in job and isinstance(job["data"], dict):
        return job["data"]

    return job


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _normalize_store(value: Any) -> str:
    return _coerce_text(value).lower()


def _review_key(payload: Dict[str, Any]) -> str:
    store = _normalize_store(payload.get("store"))
    bundle_id = _coerce_text(payload.get("bundle_id"))
    review_id = _coerce_text(payload.get("review_id"))

    if store or bundle_id:
        return f"{store}:{bundle_id}:{review_id}"

    return review_id


def _validate_review_keys(payloads: List[Dict[str, Any]]):
    require_review_id = _env_bool(REVIEW_DETECTORS_REQUIRE_REVIEW_ID)
    duplicate_policy = REVIEW_DETECTORS_DUPLICATE_REVIEW_ID_POLICY.strip().lower()

    missing_review_positions = []
    missing_text_positions = []
    review_keys = []

    for i, payload in enumerate(payloads):
        review_id = _coerce_text(payload.get("review_id"))
        text = _coerce_text(payload.get("text"))

        if require_review_id and not review_id:
            missing_review_positions.append(i)

        if not text:
            missing_text_positions.append(i)

        key = _review_key(payload)

        if key:
            review_keys.append(key)

    if missing_review_positions:
        raise ValueError(
            "Missing review_id in queue payload(s) at batch positions: "
            f"{missing_review_positions[:20]}"
        )

    if missing_text_positions:
        raise ValueError(
            "Missing text in queue payload(s) at batch positions: "
            f"{missing_text_positions[:20]}"
        )

    if duplicate_policy not in {"error", "allow"}:
        raise ValueError("REVIEW_DETECTORS_DUPLICATE_REVIEW_ID_POLICY must be error or allow.")

    if duplicate_policy == "error":
        counts = Counter(review_keys)
        duplicates = {key: n for key, n in counts.items() if n > 1}

        if duplicates:
            duplicate_preview = dict(list(duplicates.items())[:20])
            raise ValueError(
                "Duplicate review identity values found in the same queue batch. "
                "This would make output alignment ambiguous. "
                f"Duplicate preview: {duplicate_preview}"
            )


def _build_records(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []

    for payload in payloads:
        records.append({
            "prediction_id": uuid.uuid4().hex,
            "store": _normalize_store(payload.get("store")),
            "bundle_id": _coerce_text(payload.get("bundle_id")),
            "review_id": _coerce_text(payload.get("review_id")),
            "lang": _coerce_text(payload.get("lang") or payload.get("language") or payload.get("language_code")),
            "text": _coerce_text(payload.get("text")),
        })

    return records


def _clean_prediction_result(prediction: Dict[str, Any], output_format: str = "long") -> Dict[str, Any]:
    output_format = normalize_output_format(output_format)
    return {col: prediction.get(col) for col in QUEUE_OUTPUT_COLS_BY_FORMAT[output_format]}


def _build_job_ref(job: Dict[str, Any]) -> Dict[str, Any]:
    out = {"id": job.get("id")}

    if "token" in job:
        out["token"] = job.get("token")

    return out


def process_jobs(
    jobs: List[Dict[str, Any]],
    output_format: str = "long",
) -> List[Dict[str, Any]]:
    output_format = normalize_output_format(output_format)

    if not jobs:
        return []

    payloads = [_extract_payload(job) for job in jobs]
    _validate_review_keys(payloads)

    records = _build_records(payloads)
    predictions = predict_records(records, output_format = output_format)

    expected_count = len(records) * 2 if output_format == "long" else len(records)
    if len(predictions) != expected_count:
        raise RuntimeError(
            f"Prediction count mismatch for output_format={output_format!r}: "
            f"got {len(predictions)} predictions for {len(records)} records; "
            f"expected {expected_count}."
        )

    cleaned_predictions = [
        _clean_prediction_result(pred, output_format = output_format)
        for pred in predictions
    ]

    return [
        {
            "jobs": [_build_job_ref(job) for job in jobs],
            "results": cleaned_predictions,
        }
    ]


def run_once(output_format: str = REVIEW_DETECTORS_OUTPUT_FORMAT) -> int:
    output_format = normalize_output_format(output_format)
    _set_model_env()

    jobs = pop(batch_size = BATCH_SIZE)

    if not jobs:
        logger.info("No jobs received.")
        return 0

    logger.info("Pulled %s jobs.", len(jobs))

    processed_jobs = process_jobs(jobs, output_format = output_format)
    response = push(processed_jobs)

    logger.info("Pushed results for %s processed jobs.", len(jobs))
    logger.info("Queue response: %s", json.dumps(response, ensure_ascii = False)[:1000])

    return len(jobs)


def main():
    _set_model_env()

    logger.info(
        "Starting combined review detectors processor: batch_size=%s, output_format=%s, scam_hf=%s/%s, ai_hf=%s, ai_min_words=%s.",
        BATCH_SIZE,
        REVIEW_DETECTORS_OUTPUT_FORMAT,
        SCAM_REVIEW_HF_REPO_ID,
        SCAM_REVIEW_SUBFOLDER,
        AI_REVIEW_HF_REPO_ID,
        AI_REVIEW_MIN_WORDS,
    )

    while True:
        try:
            n = run_once(output_format = REVIEW_DETECTORS_OUTPUT_FORMAT)

            if n == 0:
                time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)

        except KeyboardInterrupt:
            logger.info("Stopping processor.")
            break

        except Exception as e:
            logger.exception("Processor error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
