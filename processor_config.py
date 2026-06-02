import os


QUEUE_URL = os.getenv(
    "QUEUE_URL",
    "https://deepsee-queue.herokuapp.com/exchange-batch",
)

QUEUE_API_KEY = os.getenv(
    "QUEUE_API_KEY",
    "PLACEHOLDER_QUEUE_API_KEY",
)

QUEUE_KEY = os.getenv(
    "QUEUE_KEY",
    "PLACEHOLDER_QUEUE_KEY",
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EMPTY_QUEUE_SLEEP_SECONDS = int(os.getenv("EMPTY_QUEUE_SLEEP_SECONDS", "60"))

# Scam review detector: HF / LoRA
SCAM_REVIEW_HF_REPO_ID = os.getenv(
    "SCAM_REVIEW_HF_REPO_ID",
    os.getenv("REVIEW_CLASSIFIER_HF_REPO_ID", "DeepSee-io/app-high-risk-signals-classifiers"),
)

SCAM_REVIEW_SUBFOLDER = os.getenv(
    "SCAM_REVIEW_SUBFOLDER",
    os.getenv("REVIEW_CLASSIFIER_SUBFOLDER", "scam_reviews"),
)

SCAM_REVIEW_HF_TOKEN = os.getenv(
    "SCAM_REVIEW_HF_TOKEN",
    os.getenv("REVIEW_CLASSIFIER_HF_TOKEN", os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", ""))),
)

SCAM_REVIEW_MAX_LENGTH = int(os.getenv("SCAM_REVIEW_MAX_LENGTH", os.getenv("REVIEW_CLASSIFIER_MAX_LENGTH", "512")))
SCAM_REVIEW_NO_4BIT = os.getenv("SCAM_REVIEW_NO_4BIT", os.getenv("REVIEW_CLASSIFIER_NO_4BIT", "0"))
SCAM_REVIEW_APPLY_ONE_WORD_MAPPING = os.getenv("SCAM_REVIEW_APPLY_ONE_WORD_MAPPING", "1")
SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN = os.getenv("SCAM_REVIEW_FORCE_UNKNOWN_SINGLE_WORD_CLEAN", "1")

# AI-generated review detector: HF / LoRA
# The language mapping points to the checkpoint root; aigt.model_loader appends
# /lora_adapter when retrieving adapter_config.json and adapter weights.
AI_REVIEW_HF_REPO_ID = os.getenv(
    "AI_REVIEW_HF_REPO_ID",
    os.getenv("AIGT_REPO_ID", "DeepSee-io/qwen_adapters_aigt"),
)

AI_REVIEW_SUBDIR_BY_LANG_JSON = os.getenv(
    "AI_REVIEW_SUBDIR_BY_LANG_JSON",
    os.getenv("AIGT_SUBDIR_BY_LANG_JSON", '{"en":"reviews/best"}'),
)

AI_REVIEW_REVISION = os.getenv("AI_REVIEW_REVISION", os.getenv("AIGT_REVISION", "")) or None
AI_REVIEW_HF_TOKEN = os.getenv(
    "AI_REVIEW_HF_TOKEN",
    os.getenv("AIGT_HF_TOKEN", os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", ""))),
) or None
AI_REVIEW_DEVICE = os.getenv("AI_REVIEW_DEVICE", os.getenv("AIGT_DEVICE", "cuda"))
AI_REVIEW_CACHE_POLICY = os.getenv("AI_REVIEW_CACHE_POLICY", os.getenv("AIGT_CACHE_POLICY", "keep"))
AI_REVIEW_MAX_LEN = int(os.getenv("AI_REVIEW_MAX_LEN", os.getenv("AIGT_MAX_LEN", "500")))
AI_REVIEW_BATCH_SIZE = int(os.getenv("AI_REVIEW_BATCH_SIZE", os.getenv("AIGT_BATCH_SIZE", "16")))
AI_REVIEW_WINDOW_AI_THRESHOLD = float(os.getenv("AI_REVIEW_WINDOW_AI_THRESHOLD", os.getenv("AIGT_WINDOW_AI_THRESHOLD", "0.5")))
AI_REVIEW_PREFER_BF16 = os.getenv("AI_REVIEW_PREFER_BF16", os.getenv("AIGT_PREFER_BF16", "1")) not in {"0", "false", "False"}
AI_REVIEW_MIN_WORDS = int(os.getenv("AI_REVIEW_MIN_WORDS", os.getenv("AIGT_MIN_WORDS", "10")))

# Combined worker behavior.
# Stable detector values written to long-format output rows.
SCAM_REVIEW_DETECTOR_NAME = os.getenv("SCAM_REVIEW_DETECTOR_NAME", "scam")
AI_REVIEW_DETECTOR_NAME = os.getenv("AI_REVIEW_DETECTOR_NAME", "ai")

REVIEW_DETECTORS_REQUIRE_REVIEW_ID = os.getenv("REVIEW_DETECTORS_REQUIRE_REVIEW_ID", "1")
REVIEW_DETECTORS_DUPLICATE_REVIEW_ID_POLICY = os.getenv("REVIEW_DETECTORS_DUPLICATE_REVIEW_ID_POLICY", "error")
REVIEW_DETECTORS_CACHE_POLICY = os.getenv("REVIEW_DETECTORS_CACHE_POLICY", "keep")

# Output shape: long (default; one row per review per detector) or wide (compatibility).
REVIEW_DETECTORS_OUTPUT_FORMAT = os.getenv("REVIEW_DETECTORS_OUTPUT_FORMAT", "long")
