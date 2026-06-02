from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import logging
import json

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TextInferConfig:
    # HuggingFace / LoRA assets. Each mapped subdir is a checkpoint root;
    # the AIGT loader retrieves LoRA files from <subdir>/lora_adapter/.
    repo_id: str = "DeepSee-io/qwen_adapters_aigt"
    subdir_by_lang_json: str = '{"en":"reviews/best"}'
    revision: Optional[str] = None
    hf_token: Optional[str] = None

    # Runtime knobs
    device: str = "cuda"
    cache_policy: str = "unload_after_call"
    max_len: int = 500
    batch_size: int = 16
    window_ai_threshold: float = 0.5
    prefer_bf16: bool = True
    min_words: int = 10


class TextClassifier:
    """
    Review-level AI text detection via `aigt.detect_batch`.

    Public API mirrors the image worker style:
      - load_models()
      - are_models_loaded()
      - classify_texts_batch(...)

    Returns exactly one dict per input text.
    Output order always matches input order.
    Missing / dropped docs are emitted as empty_or_failed rows.
    """

    def __init__(self, cfg: Optional[TextInferConfig] = None):
        self.cfg = cfg or TextInferConfig()
        self._loaded = False
        self._detector = None
        self._subdir_by_lang = json.loads(self.cfg.subdir_by_lang_json)

    def _build_detector(self):
        from aigt import Detector, RuntimeConfig

        runtime = RuntimeConfig(
            device = self.cfg.device,
            cache_policy = self.cfg.cache_policy,
            model_name_fallback = "Qwen/Qwen2.5-3B-Instruct",
            max_length_fallback = 512,
            window_ai_threshold = float(self.cfg.window_ai_threshold),
            prefer_bf16 = bool(self.cfg.prefer_bf16),
        )

        return Detector.from_hf(
            repo_id = self.cfg.repo_id,
            subdir_by_lang = self._subdir_by_lang,
            revision = self.cfg.revision,
            token = self.cfg.hf_token,
            runtime = runtime,
        )

    def load_models(self):
        from aigt import WindowConfig, BatchConfig

        self._detector = self._build_detector()

        for lang in self._subdir_by_lang:
            logger.info("Pre-loading language model: %s", lang)
            self._detector.load_language(lang)

        self._detector.predict(
            texts = ["warmup"],
            doc_ids = ["warmup"],
            lang = "en",
            window = WindowConfig(token_length = int(self.cfg.max_len)),
            batch = BatchConfig(batch_size = 1, show_progress = False),
            window_ai_threshold = float(self.cfg.window_ai_threshold),
        )

        self._loaded = True
        logger.info("Text classifier initialized successfully.")

    def are_models_loaded(self) -> bool:
        return self._loaded and self._detector is not None

    def _coerce_text(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def _count_words(self, text: Any) -> int:
        text = self._coerce_text(text)
        if not text:
            return 0
        return len(text.split())

    def _coerce_lang(self, x: Any) -> str:
        if x is None or pd.isna(x):
            return "en"

        lang = str(x).lower().strip()

        if not lang:
            return "en"

        lang = lang.split("-")[0].split("_")[0]

        if lang not in self._subdir_by_lang:
            return "en"

        return lang

    def classify_texts_batch(
        self,
        texts: Sequence[Any],
        langs: Optional[Sequence[str]] = None,
        prediction_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns one dict per input text, aligned to input order.
        """

        if not self.are_models_loaded():
            raise RuntimeError("Models not loaded. Call load_models() first.")

        from aigt import WindowConfig, BatchConfig

        clean_texts = [self._coerce_text(t) for t in texts]
        word_counts = [self._count_words(t) for t in clean_texts]
        n = len(clean_texts)

        if langs is None:
            langs_list = ["en"] * n
        else:
            langs_list = [self._coerce_lang(x) for x in langs]

        if prediction_ids is None:
            prediction_ids_list = [str(i) for i in range(n)]
        else:
            prediction_ids_list = [str(x) for x in prediction_ids]

        if len(langs_list) != n:
            raise ValueError(
                f"langs must have same length as texts: got len(langs)={len(langs_list)}, len(texts)={n}"
            )

        if len(prediction_ids_list) != n:
            raise ValueError(
                f"prediction_ids must have same length as texts: got len(prediction_ids)={len(prediction_ids_list)}, len(texts)={n}"
            )

        try:
            min_words = int(self.cfg.min_words)
            eligible_mask = [wc >= min_words for wc in word_counts]

            model_texts = [
                text for text, keep in zip(clean_texts, eligible_mask)
                if keep
            ]

            model_prediction_ids = [
                pid for pid, keep in zip(prediction_ids_list, eligible_mask)
                if keep
            ]

            model_langs = [
                lang for lang, keep in zip(langs_list, eligible_mask)
                if keep
            ]

            if model_texts:
                window_cfg = WindowConfig(token_length = int(self.cfg.max_len))
                batch_cfg = BatchConfig(
                    batch_size = int(self.cfg.batch_size),
                    show_progress = False,
                )

                articles_df, windows_df = self._detector.predict(
                    texts = model_texts,
                    doc_ids = list(model_prediction_ids),
                    lang = list(model_langs),
                    window = window_cfg,
                    batch = batch_cfg,
                    window_ai_threshold = float(self.cfg.window_ai_threshold),
                )
            else:
                articles_df = pd.DataFrame()
                windows_df = pd.DataFrame()

            if articles_df is None:
                articles_df = pd.DataFrame()

            if windows_df is None:
                windows_df = pd.DataFrame()

            if not isinstance(articles_df, pd.DataFrame):
                articles_df = pd.DataFrame(articles_df)

            if not isinstance(windows_df, pd.DataFrame):
                windows_df = pd.DataFrame(windows_df)

            if (
                not windows_df.empty
                and "prediction_id" in windows_df.columns
                and "token_count" in windows_df.columns
            ):
                token_counts = (
                    windows_df.assign(
                        prediction_id = windows_df["prediction_id"].astype(str)
                    )
                    .groupby("prediction_id", as_index = True)["token_count"]
                    .sum()
                    .to_dict()
                )
            else:
                token_counts = {}

            articles_by_pid: Dict[str, pd.Series] = {}
            duplicate_pids: List[str] = []

            if not articles_df.empty and "prediction_id" in articles_df.columns:
                for _, r in articles_df.iterrows():
                    pid = str(r.get("prediction_id"))
                    if pid in articles_by_pid:
                        duplicate_pids.append(pid)
                        continue
                    articles_by_pid[pid] = r
            elif not articles_df.empty:
                logger.warning(
                    "articles_df returned by detector has no prediction_id column; all outputs will be marked empty_or_failed"
                )

            if duplicate_pids:
                dup_preview = duplicate_pids[:10]
                logger.warning(
                    "Duplicate prediction_id rows found in articles_df; keeping first occurrence. Examples: %s",
                    dup_preview,
                )

            out: List[Dict[str, Any]] = []

            for i, pid in enumerate(prediction_ids_list):
                requested_lang = langs_list[i]

                if not eligible_mask[i]:
                    out.append({
                        "status": "skipped_short_text",
                        "prediction_id": pid,
                        "lang": requested_lang,
                        "prediction_short": None,
                        "prediction_long": None,
                        "fraction_ai": None,
                        "ai_probability": None,
                        "human_probability": None,
                        "n_windows": 0,
                        "n_ai_segments": 0,
                        "n_human_segments": 0,
                        "n_tokens": 0,
                    })
                    continue

                r = articles_by_pid.get(pid)

                if r is None:
                    out.append({
                        "status": "empty_or_failed",
                        "prediction_id": pid,
                        "lang": requested_lang,
                        "prediction_short": None,
                        "prediction_long": None,
                        "fraction_ai": None,
                        "ai_probability": None,
                        "human_probability": None,
                        "n_windows": 0,
                        "n_ai_segments": 0,
                        "n_human_segments": 0,
                        "n_tokens": int(token_counts.get(pid, 0) or 0),
                    })
                    continue

                row_lang = str(r.get("lang") or requested_lang)

                n_windows_raw = r.get("num_windows")
                try:
                    n_windows = int(n_windows_raw or 0)
                except Exception:
                    n_windows = 0

                ai_text_probability_raw = r.get("ai_text_probability")
                fraction_ai_raw = r.get("fraction_ai")

                is_missing = (
                    n_windows == 0
                    or ai_text_probability_raw is None
                    or pd.isna(ai_text_probability_raw)
                    or fraction_ai_raw is None
                    or pd.isna(fraction_ai_raw)
                )

                if is_missing:
                    out.append({
                        "status": "empty_or_failed",
                        "prediction_id": pid,
                        "lang": row_lang,
                        "prediction_short": None,
                        "prediction_long": None,
                        "fraction_ai": None,
                        "ai_probability": None,
                        "human_probability": None,
                        "n_windows": 0,
                        "n_ai_segments": 0,
                        "n_human_segments": 0,
                        "n_tokens": int(token_counts.get(pid, 0) or 0),
                    })
                    continue

                ai_probability = float(ai_text_probability_raw)
                fraction_ai = float(fraction_ai_raw)

                ai_probability = max(0.0, min(1.0, ai_probability))
                fraction_ai = max(0.0, min(1.0, fraction_ai))

                try:
                    n_ai_segments = int(r.get("num_ai_segments") or 0)
                except Exception:
                    n_ai_segments = 0

                try:
                    n_human_segments = int(r.get("num_human_segments") or 0)
                except Exception:
                    n_human_segments = 0

                out.append({
                    "status": "success",
                    "prediction_id": pid,
                    "lang": row_lang,
                    "prediction_short": str(r.get("prediction_short")),
                    "prediction_long": str(r.get("prediction")),
                    "fraction_ai": fraction_ai,
                    "ai_probability": ai_probability,
                    "human_probability": float(1.0 - ai_probability),
                    "n_windows": n_windows,
                    "n_ai_segments": n_ai_segments,
                    "n_human_segments": n_human_segments,
                    "n_tokens": int(token_counts.get(pid, 0) or 0),
                })

            if len(out) != n:
                logger.warning(
                    "Output length mismatch after reconstruction: len(out)=%s, len(texts)=%s",
                    len(out),
                    n,
                )

            return out

        except Exception as e:
            msg = str(e)
            logger.exception("Batch text processing failed: %s", msg)

            return [{
                "status": "error",
                "error": msg,
                "prediction_id": prediction_ids_list[i],
                "lang": langs_list[i],
                "prediction_short": None,
                "prediction_long": None,
                "fraction_ai": None,
                "ai_probability": None,
                "human_probability": None,
                "n_windows": 0,
                "n_ai_segments": 0,
                "n_human_segments": 0,
                "n_tokens": 0,
            } for i in range(n)]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "status": "loaded" if self.are_models_loaded() else "not_loaded",
            "models_loaded": self.are_models_loaded(),
            "repo_id": self.cfg.repo_id,
            "subdir_by_lang": self._subdir_by_lang,
            "revision": self.cfg.revision,
            "device": self.cfg.device,
            "cache_policy": self.cfg.cache_policy,
            "max_len": self.cfg.max_len,
            "batch_size": self.cfg.batch_size,
            "window_ai_threshold": self.cfg.window_ai_threshold,
            "prefer_bf16": self.cfg.prefer_bf16,
            "min_words": self.cfg.min_words,
        }
