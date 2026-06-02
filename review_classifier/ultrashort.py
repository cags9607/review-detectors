import re
import unicodedata
from typing import Any, Dict, Optional

import pandas as pd


single_word_scam_mapping = {
    # broad scam / fraud
    "scam": "explicit_scam_or_fraud",
    "scams": "explicit_scam_or_fraud",
    "fraud": "explicit_scam_or_fraud",
    "frauds": "explicit_scam_or_fraud",
    "fraudulent": "explicit_scam_or_fraud",
    "fake": "explicit_scam_or_fraud",
    "phony": "explicit_scam_or_fraud",
    "bogus": "explicit_scam_or_fraud",
    "deceptive": "explicit_scam_or_fraud",
    "deception": "explicit_scam_or_fraud",
    "misleading": "explicit_scam_or_fraud",
    "ripoff": "explicit_scam_or_fraud",
    "scammer": "explicit_scam_or_fraud",
    "scammers": "explicit_scam_or_fraud",
    "cheater": "explicit_scam_or_fraud",
    "cheaters": "explicit_scam_or_fraud",
    "cheating": "explicit_scam_or_fraud",

    # theft / stealing
    "theft": "explicit_scam_or_fraud",
    "thief": "explicit_scam_or_fraud",
    "thieves": "explicit_scam_or_fraud",
    "steal": "explicit_scam_or_fraud",
    "stealing": "explicit_scam_or_fraud",
    "stole": "explicit_scam_or_fraud",
    "robbery": "explicit_scam_or_fraud",
    "robber": "explicit_scam_or_fraud",
    "robbers": "explicit_scam_or_fraud",
    "stolen": "explicit_scam_or_fraud",

    # cyber / malicious behavior
    "malware": "malware_or_spyware_claims",
    "spyware": "malware_or_spyware_claims",
    "virus": "malware_or_spyware_claims",
    "trojan": "malware_or_spyware_claims",
    "phishing": "impersonation_phishing",
    "hacked": "malware_or_spyware_claims",
    "hacker": "malware_or_spyware_claims",
    "hackers": "malware_or_spyware_claims",

    # adware
    "adware": "adware_aggressive_ads",

    # romance / fake chat
    "catfish": "romance_or_fake_chat_scam",
    "catfishing": "romance_or_fake_chat_scam",
    "bots": "romance_or_fake_chat_scam",

    # billing / money abuse
    "overcharged": "subscription_billing_abuse",
    "overcharge": "subscription_billing_abuse",
    "charged": "subscription_billing_abuse",

    # payout / rewards
    "nonpayment": "reward_payout_scam",
    "unpaid": "reward_payout_scam",

    # loan abuse
    "loan-shark": "predatory_loan",
    "loanshark": "predatory_loan",
    "extortion": "predatory_loan",
    "blackmail": "predatory_loan",
}


emoji_like_clean_mapping = {
    # negative emoji / symbols: negative sentiment, but not scam evidence
    "😡": "clean",
    "🤬": "clean",
    "😠": "clean",
    "😤": "clean",
    "👎": "clean",
    "💩": "clean",
    "🤮": "clean",
    "😒": "clean",
    "😑": "clean",
    "🙄": "clean",
    "😞": "clean",
    "😔": "clean",
    "😭": "clean",
    "😢": "clean",
    "😱": "clean",
    "🚫": "clean",
    "❌": "clean",
    "⛔": "clean",
    "🖕": "clean",

    # punctuation-only / symbolic frustration: negative, but not scam evidence
    "!": "clean",
    "!!": "clean",
    "!!!": "clean",
    "!!!!": "clean",
    "!!!!!": "clean",
    "?": "clean",
    "??": "clean",
    "???": "clean",
    "????": "clean",
    "?????": "clean",
    "...": "clean",
    "..": "clean",
    ".": "clean",
    "-": "clean",
    "--": "clean",
    "---": "clean",
}


def normalize_short_review_text(text: Any) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("–", "-")
        .replace("—", "-")
    )

    return text.strip()


def strip_outer_punctuation(text: str) -> str:
    return re.sub(r"^[^\w#@+-]+|[^\w#@+-]+$", "", text, flags = re.UNICODE)


def count_whitespace_words(text: Any) -> int:
    normalized = normalize_short_review_text(text)

    if normalized == "":
        return 0

    return len(normalized.split())


def is_emoji_or_symbol_only(text: Any) -> bool:
    normalized = normalize_short_review_text(text)

    if normalized == "":
        return False

    has_alnum = any(char.isalnum() for char in normalized)
    has_nonspace = any(not char.isspace() for char in normalized)

    return has_nonspace and not has_alnum


def is_repeated_punctuation_or_symbol(text: Any) -> bool:
    normalized = normalize_short_review_text(text)

    if normalized == "":
        return False

    compact = re.sub(r"\s+", "", normalized)

    if compact == "":
        return False

    return (
        is_emoji_or_symbol_only(compact)
        and len(set(compact)) <= 2
        and len(compact) <= 12
    )


def map_ultrashort_review(
    text: Any,
    scam_mapping: Dict[str, str] = single_word_scam_mapping,
    emoji_mapping: Dict[str, str] = emoji_like_clean_mapping,
) -> Dict[str, Any]:
    """
    Conservative pre-labeler for reviews with <= 1 whitespace-delimited word.

    Explicit scam/fraud terms become scam subcategories. Emoji/punctuation-only
    texts become clean because they are sentiment, not scam evidence. Unknown
    one-word text is left as clean but not marked as heuristic_applied, matching
    the original helper contract.
    """
    normalized = normalize_short_review_text(text)
    n_words = count_whitespace_words(normalized)

    empty_result = {
        "heuristic_label": "clean",
        "heuristic_confidence": None,
        "heuristic_reason": None,
        "heuristic_evidence": None,
        "heuristic_applied": False,
    }

    if n_words > 1:
        return empty_result

    if normalized == "":
        return {
            "heuristic_label": "clean",
            "heuristic_confidence": 0.70,
            "heuristic_reason": "Empty or whitespace-only review has no scam evidence.",
            "heuristic_evidence": "",
            "heuristic_applied": True,
        }

    stripped = strip_outer_punctuation(normalized)

    if stripped in scam_mapping:
        return {
            "heuristic_label": scam_mapping[stripped],
            "heuristic_confidence": 0.95,
            "heuristic_reason": "Single-token review contains an explicit scam/fraud term.",
            "heuristic_evidence": normalized,
            "heuristic_applied": True,
        }

    if normalized in emoji_mapping:
        return {
            "heuristic_label": emoji_mapping[normalized],
            "heuristic_confidence": 0.75,
            "heuristic_reason": "Emoji/symbol-only review is negative but does not provide scam evidence.",
            "heuristic_evidence": normalized,
            "heuristic_applied": True,
        }

    if is_repeated_punctuation_or_symbol(normalized):
        return {
            "heuristic_label": "clean",
            "heuristic_confidence": 0.70,
            "heuristic_reason": "Symbol-only review does not provide scam evidence.",
            "heuristic_evidence": normalized,
            "heuristic_applied": True,
        }

    return empty_result


def add_ultrashort_review_labels(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"df must contain column: {text_col}")

    out = df.copy()

    heuristic_df = pd.DataFrame(
        out[text_col].map(map_ultrashort_review).tolist()
    )

    return pd.concat(
        [
            out.reset_index(drop = True),
            heuristic_df.reset_index(drop = True),
        ],
        axis = 1,
    )


def replace_single_word(
    df: pd.DataFrame,
    pred_col: str = "pred_label",
    n_words_col: str = "n_words",
    heuristic_col: str = "heuristic_label",
    only_non_null_heuristic: bool = True,
) -> pd.DataFrame:
    required_cols = [pred_col, n_words_col, heuristic_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out = df.copy()

    mask = out[n_words_col].eq(1)

    if only_non_null_heuristic:
        mask = mask & out[heuristic_col].notna()

    out[pred_col] = out[pred_col].mask(mask, out[heuristic_col])

    return out


def apply_ultrashort_overrides(
    df: pd.DataFrame,
    label_col: str = "pred_label",
    confidence_col: str = "pred_confidence",
    text_col: str = "text",
    force_unknown_single_word_clean: bool = True,
) -> pd.DataFrame:
    """
    Apply the one-word review heuristic after model inference.

    Default behavior intentionally mirrors the supplied replace_single_word helper:
    every one-word review is assigned the heuristic label. Unknown one-word tokens
    therefore become clean because the original helper returns clean as the empty
    default label.
    """
    out = add_ultrashort_review_labels(df, text_col = text_col)

    if "n_words" not in out.columns:
        out["n_words"] = out[text_col].map(count_whitespace_words)

    one_word_mask = out["n_words"].eq(1)

    if not force_unknown_single_word_clean:
        one_word_mask = one_word_mask & out["heuristic_applied"].eq(True)

    out[label_col] = out[label_col].mask(one_word_mask, out["heuristic_label"])

    if confidence_col in out.columns:
        out[confidence_col] = out[confidence_col].mask(
            one_word_mask & out["heuristic_confidence"].notna(),
            out["heuristic_confidence"],
        )

    return out
