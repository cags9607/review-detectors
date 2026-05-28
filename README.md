# Combined Review Detectors

Inference-only queue worker that combines two review-level detectors:

1. Scam review classifier
2. AI-generated review detector

The scam detector uses `Trinotrotolueno/review-scam-adapters` with the default subfolder `scam_reviews`.
The AI-review detector uses `Trinotrotolueno/aigt-loras` with a configurable language-to-subdir mapping.

## Queue input contract

Each queue job should contain one review, either directly or under `payload` / `data`:

```json
{
  "store": "google",
  "bundle_id": "com.example.app",
  "review_id": "review_001",
  "text": "This app is a scam",
  "lang": "en"
}
```

`lang` is optional and defaults to `en`. `language` and `language_code` are also accepted.

## Queue output contract

The processor pushes exactly one result row per input review with these columns:

```text
store
bundle_id
review_id
text
scam_pred_label
scam_pred_confidence
ai_pred_label
ai_pred_confidence
ai_probability
fraction_ai
n_tokens
```

Internal fields such as `prediction_id`, heuristic audit columns, label IDs, and per-class probabilities are not pushed.

## Short-review behavior

The scam detector applies the 1-word mapping internally. For example, `scam`, `fraud`, `fake`, `malware`, and similar one-token reviews can be mapped to scam subcategories.

The AI-review detector keeps the original minimum word policy. By default, reviews with fewer than `AI_REVIEW_MIN_WORDS=10` are returned as:

```text
ai_pred_label = skipped_short_text
ai_pred_confidence = null
ai_probability = null
fraction_ai = null
```

## Environment variables

See `env.example` for the full set. Key defaults:

```bash
SCAM_REVIEW_HF_REPO_ID=Trinotrotolueno/review-scam-adapters
SCAM_REVIEW_SUBFOLDER=scam_reviews
AI_REVIEW_HF_REPO_ID=Trinotrotolueno/aigt-loras
AI_REVIEW_SUBDIR_BY_LANG_JSON='{"en":"reviews/best"}'
AI_REVIEW_MIN_WORDS=10
```

If `Trinotrotolueno/aigt-loras` uses a different folder than `reviews/best`, update `AI_REVIEW_SUBDIR_BY_LANG_JSON`.

## Local CSV scoring

```bash
pip install -e .
python scripts/predict_csv.py input.csv output.csv
```

## Queue worker

```bash
pip install -e .
python processor.py
```
