# Review Detectors

Inference-only queue worker that combines two review-level detectors:

1. Scam review classifier
2. AI-generated review detector

The scam detector uses `Trinotrotolueno/review-scam-adapters` with the default subfolder `scam_reviews`.
The AI-review detector uses `Trinotrotolueno/aigt-loras` with a configurable language-to-subdir mapping.

## Output format

The primary inference API is:

```python
predict_records(records, output_format = "long")
```

`output_format = "long"` is the default. It emits one result row per detector per input review, so two enabled detectors produce two rows per review. The former one-row-per-review layout remains available with `output_format = "wide"` for local compatibility and migration checks.

The queue processor uses `REVIEW_DETECTORS_OUTPUT_FORMAT=long` by default. Its long-format payload is intentionally smaller than the detailed local/core output because it is designed for ClickHouse storage.

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

For each review, the processor pushes two result rows: one with `detector = "scam"` and one with `detector = "ai"`.

```text
store
bundle_id
review_id
detector
label
confidence
```

The processor intentionally omits `text`, `score`, `fraction_ai`, and `n_tokens` from long-format queue results. Review text already exists in the source review table and duplicating it once per detector row would grow storage unnecessarily.

| Field | Description |
|---|---|
| `store` | Review store identifier. |
| `bundle_id` | App identifier. |
| `review_id` | Review identifier. |
| `detector` | Stable detector identifier, configurable and defaulting to `scam` or `ai`. |
| `label` | Detector-specific label. Scam subcategories for `scam`; `Human`, `Mixed`, `AI`, or null for `ai`. |
| `confidence` | Confidence in `label`; null when no AI prediction is emitted. |

Example processor result rows for one short review:

```json
[
  {
    "store": "google",
    "bundle_id": "com.example.app",
    "review_id": "review_001",
    "detector": "scam",
    "label": "explicit_scam_or_fraud",
    "confidence": 0.91
  },
  {
    "store": "google",
    "bundle_id": "com.example.app",
    "review_id": "review_001",
    "detector": "ai",
    "label": null,
    "confidence": null
  }
]
```

Missing results are serialized as JSON `null`, not `NaN`.

## Detailed local/core long output

The underlying `core.predict_records(records, output_format = "long")` function retains additional fields for local analysis and debugging:

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

For `detector = "ai"`, `score` is `ai_probability`. These detailed fields are not pushed by the queue processor in long mode.

## Compatibility wide output

For callers that still need the previous local schema:

```python
predict_records(records, output_format = "wide")
```

or temporarily set:

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

The scam detector applies the 1-word mapping internally. For example, `scam`, `fraud`, `fake`, and `malware` can map directly to scam subcategories.

The AI-review detector keeps the original minimum word policy. By default, reviews with fewer than `AI_REVIEW_MIN_WORDS=10` still produce an `ai` result row, but with:

```text
label = null
confidence = null
```

The only non-null values emitted in the AI detector's `label` field are:

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
SCAM_REVIEW_DETECTOR_NAME=scam
AI_REVIEW_HF_REPO_ID=Trinotrotolueno/aigt-loras
AI_REVIEW_SUBDIR_BY_LANG_JSON='{"en":"reviews/best"}'
AI_REVIEW_DETECTOR_NAME=ai
AI_REVIEW_MIN_WORDS=10
REVIEW_DETECTORS_OUTPUT_FORMAT=long
```

Tokens are not embedded in this repository. Supply `HF_TOKEN`, `SCAM_REVIEW_HF_TOKEN`, or `AI_REVIEW_HF_TOKEN` at runtime when required.

## Local CSV scoring

Detailed long output is the default for local scoring:

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
