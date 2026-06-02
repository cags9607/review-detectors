import os
from getpass import getpass
from typing import Optional, Tuple
from urllib.parse import urlparse


def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "y",
        "Y",
    }


def get_hf_token(token: Optional[str] = None) -> Optional[str]:
    return (
        token
        or os.getenv("REVIEW_CLASSIFIER_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def normalize_hf_repo_id(repo_id_or_url: str) -> str:
    value = str(repo_id_or_url).strip()

    if not value:
        raise ValueError("Hugging Face repo ID/URL cannot be empty.")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path_parts = [part for part in parsed.path.split("/") if part]

        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError(
                "Expected a Hugging Face URL like https://huggingface.co/org/repo."
            )

        if len(path_parts) < 2:
            raise ValueError(
                "Could not parse Hugging Face repo ID from URL. "
                "Expected https://huggingface.co/org/repo."
            )

        return "/".join(path_parts[:2])

    return value


def normalize_subfolder(subfolder: Optional[str] = None) -> Optional[str]:
    value = (
        subfolder
        if subfolder is not None
        else os.getenv("REVIEW_CLASSIFIER_SUBFOLDER", "")
    )

    value = str(value).strip()

    return value or None


def prompt_for_hf_repo_and_token(
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    force_prompt: bool = False,
) -> Tuple[str, Optional[str]]:
    repo_value = None if force_prompt else model_id

    while not repo_value:
        repo_value = input(
            "Hugging Face repo ID or URL "
            "(for example DeepSee-io/app-high-risk-signals-classifiers): "
        ).strip()

    repo_id = normalize_hf_repo_id(repo_value)

    hf_token = None if force_prompt else get_hf_token(token)

    if not hf_token:
        entered = getpass(
            "Hugging Face API key/token "
            "(press Enter if the repo is public): "
        ).strip()
        hf_token = entered or None

    return repo_id, hf_token
