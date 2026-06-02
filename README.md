# Review Detectors

Inference-only queue worker that combines two review-level detectors:

1. Scam review classifier
2. AI-generated review detector

The scam detector uses `Trinotrotolueno/review-scam-adapters` with the default subfolder `scam_reviews`.
The AI-review detector uses `Trinotrotolueno/aigt-loras` with a configurable language-to-subdir mapping.

## Output format

The primary API is:

```python
predict_records(records, output_format = "long")
```

`output_format = "long"` is the default. It emits one result row per detector per input review, so two enabled detectors produce two rows per review. The former one-row-per-review layout remains available with `output_format = "wide"` for compatibility and migration checks.

The queue processor uses `REVIEW_DETECTORS_OUTPUT_FORMAT=long` by default. It can be temporarily set to `wide` during a migration if needed.

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

## Default long queue output contract

For each review, the processor pushes two result rows: one with `detector = "scam_review"` and one with `detector = "ai_review"`.

```text
store
bundle_id
review_id
text
detector
label
confidence
score
fraction_ai
n_tokens
```

Field definitions:

| Field | Description |
|---|---|
| `detector` | Stable detector identifier: currently `scam_review` or `ai_review`. |
| `label` | Detector-specific label. Scam subcategories for `scam_review`; `Human`, `Mixed`, `AI`, or null for `ai_review`. |
| `confidence` | Confidence in the emitted `label`; null when no AI label is emitted. |
| `score` | Detector-specific primary score. For `ai_review`, this is `ai_probability`; currently null for `scam_review`. |
| `fraction_ai` | AI segment fraction emitted only for `ai_review`; null for `scam_review`. |
| `n_tokens` | Number of tokens evaluated by the AI detector; null for `scam_review`. |

Example for one review:

```json
[
  {
    "store": "google",
    "bundle_id": "com.example.app",
    "review_id": "review_001",
    "text": "This app is a scam",
    "detector": "scam_review",
    "label": "explicit_scam_or_fraud",
    "confidence": 0.91,
    "score": null,
    "fraction_ai": null,
    "n_tokens": null
  },
  {
    "store": "google",
    "bundle_id": "com.example.app",
    "review_id": "review_001",
    "text": "This app is a scam",
    "detector": "ai_review",
    "label": null,
    "confidence": null,
    "score": null,
    "fraction_ai": null,
    "n_tokens": 0
  }
]
```

Internal fields such as `prediction_id`, heuristic audit columns, label IDs, and per-class probabilities are not pushed.

## Compatibility wide output

For callers that still need the previous schema:

```python
predict_records(records, output_format = "wide")
```

or set:

```bash
REVIEW_DETECTORS_OUTPUT_FORMAT=wide
```

Wide output emits one row per review:

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

## Short-review behavior

The scam detector applies the 1-word mapping internally. For example, `scam`, `fraud`, `fake`, `malware`, and similar one-token reviews can be mapped to scam subcategories.

The AI-review detector keeps the original minimum word policy. By default, reviews with fewer than `AI_REVIEW_MIN_WORDS=10` still produce an `ai_review` row in long format, but with null public prediction fields:

```text
label = null
confidence = null
score = null
fraction_ai = null
```

The only non-null values emitted by the AI detector in `label` are:

```text
Human
Mixed
AI
```

Internal conditions such as short text, empty inference output, or an AI-detector error are not exposed as label categories.

## Environment variables

See `env.example` for the full set. Key defaults:

```bash
SCAM_REVIEW_HF_REPO_ID=Trinotrotolueno/review-scam-adapters
SCAM_REVIEW_SUBFOLDER=scam_reviews
AI_REVIEW_HF_REPO_ID=Trinotrotolueno/aigt-loras
AI_REVIEW_SUBDIR_BY_LANG_JSON='{"en":"reviews/best"}'
AI_REVIEW_MIN_WORDS=10
REVIEW_DETECTORS_OUTPUT_FORMAT=long
```

If `Trinotrotolueno/aigt-loras` uses a different folder than `reviews/best`, update `AI_REVIEW_SUBDIR_BY_LANG_JSON`.

## Local CSV scoring

Long output is the default:

```bash
pip install -e .
python scripts/predict_csv.py input.csv output_long.csv
```

For backward-compatible wide output:

```bash
python scripts/predict_csv.py input.csv output_wide.csv --output-format wide
```

## Queue worker

```bash
pip install -e .
python processor.py
```
