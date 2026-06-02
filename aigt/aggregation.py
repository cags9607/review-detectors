from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def prediction_bins_from_fraction(ai_ratio: float) -> Tuple[str, str]:
    if ai_ratio != ai_ratio:
        return ("Unknown", "Unknown")
    p = float(ai_ratio)
    if p <= 0.0:
        return ("Fully human-written", "Human")
    if p >= 1.0:
        return ("Fully AI-generated", "AI")
    if p < 0.10:
        return ("Primarily human; small AI detected", "Human")
    if p < 0.30:
        return ("Primarily human; some AI detected", "Human")
    if p < 0.80:
        return ("Mix of AI and human", "Mixed")
    return ("Primarily AI; some human detected", "AI")


def aggregate_token_weighted(
    df_windows: pd.DataFrame,
    *,
    threshold: float,
    tail_min_tokens: int = 75,
) -> pd.DataFrame:
    """
    Same output contract as before.

    Changes:
      - ai_text_probability = token-weighted mean of window probabilities
      - fraction_ai = previous thresholded token-weighted fraction
      - tail_min_tokens is kept for compatibility but not used
      - no tail exception is applied
    """
    thr = float(threshold)

    required = {"prediction_id", "ai_assistance_score", "token_count"}
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"df_windows missing required columns: {sorted(missing)}")

    output_cols = [
        "prediction_id",
        "ai_text_probability",
        "fraction_ai",
        "fraction_human",
        "num_ai_segments",
        "num_human_segments",
        "prediction",
        "prediction_short",
    ]

    if df_windows.empty:
        return pd.DataFrame(columns = output_cols)

    df0 = df_windows.copy()
    df0["prediction_id"] = df0["prediction_id"].astype(str)

    has_window_index = "window_index" in df0.columns

    def _agg(g: pd.DataFrame) -> pd.Series:
        if has_window_index:
            g0 = g.sort_values("window_index", kind = "mergesort").reset_index(drop = True)
        else:
            g0 = g.reset_index(drop = True)

        scores = pd.to_numeric(g0["ai_assistance_score"], errors = "coerce").to_numpy(dtype = float)
        tw = pd.to_numeric(g0["token_count"], errors = "coerce").to_numpy(dtype = float)

        valid = np.isfinite(scores)

        if valid.sum() == 0:
            return pd.Series(
                {
                    "ai_text_probability": np.nan,
                    "fraction_ai": np.nan,
                    "fraction_human": np.nan,
                    "num_ai_segments": 0,
                    "num_human_segments": 0,
                    "prediction": "Unknown",
                    "prediction_short": "Unknown",
                }
            )

        scores_v = scores[valid]
        tw_v = tw[valid]

        is_ai = (scores_v >= thr).astype(int)
        num_ai = int(is_ai.sum())
        num_human = int(valid.sum() - num_ai)

        if np.isfinite(tw_v).all() and tw_v.sum() > 0:
            ai_text_probability = float(np.average(scores_v, weights = tw_v))
            fraction_ai = float(tw_v[is_ai == 1].sum() / tw_v.sum())
        else:
            ai_text_probability = float(np.nanmean(scores_v))
            fraction_ai = float(is_ai.mean())

        ai_text_probability = max(0.0, min(1.0, ai_text_probability))
        fraction_ai = max(0.0, min(1.0, fraction_ai))

        prediction, prediction_short = prediction_bins_from_fraction(fraction_ai)

        return pd.Series(
            {
                "ai_text_probability": ai_text_probability,
                "fraction_ai": fraction_ai,
                "fraction_human": float(1.0 - fraction_ai),
                "num_ai_segments": num_ai,
                "num_human_segments": num_human,
                "prediction": prediction,
                "prediction_short": prediction_short,
            }
        )

    out = (
        df0
        .groupby("prediction_id", sort = False, group_keys = False)
        .apply(_agg, include_groups = False)
        .reset_index()
    )

    return out
