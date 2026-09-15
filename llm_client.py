"""
llm_client.py — Central configuration for LLM providers.

Provides a single OpenAI-compatible client and the model names used by
agents, guardrails, and long-term memory. The active provider is selected
through LLM_PROVIDER in Apikey.env.

Supported providers: Groq, DeepSeek, and Gemini.

Usage:
    from llm_client import get_llm_client, model_names

    client = get_llm_client()
    models = model_names()
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / "Apikey.env")


def configure_tls_certificates():
    """Repair a stale certificate-file override without disabling TLS checks."""
    configured = os.environ.get("SSL_CERT_FILE")
    if configured is not None and not Path(configured).is_file():
        import certifi
        bundle = Path(certifi.where())
        if not bundle.is_file():
            raise RuntimeError("The certifi CA bundle is missing; reinstall certifi")
        os.environ["SSL_CERT_FILE"] = str(bundle)


configure_tls_certificates()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")

if not LLM_PROVIDER:
    raise RuntimeError("LLM_PROVIDER Missing in Apikey.env")

LLM_PROVIDER = LLM_PROVIDER.strip().lower()

# Keep model selection centralized for generation, citation checks, and summaries.
_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": {
            "main": "llama-3.3-70b-versatile",
            "check": "llama-3.3-70b-versatile",
            "light": "llama-3.1-8b-instant",
        },
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": {
            "main": "deepseek-chat",
            "check": "deepseek-chat",
            "light": "deepseek-chat",
        },
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "models": {
            "main": "gemini-3.5-flash",
            "check": "gemini-3.5-flash",
            "light": "gemini-3.1-flash-lite",
        },
    },
}


def _config() -> dict:
    cfg = _PROVIDERS.get(LLM_PROVIDER)
    if cfg is None:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r} in Apikey.env — "
            f"expected one of {list(_PROVIDERS)}"
        )
    return cfg


def get_llm_client() -> OpenAI:
    """OpenAI-compatible client for whichever provider LLM_PROVIDER selects."""
    configure_tls_certificates()
    cfg = _config()
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing {cfg['api_key_env']} — add it to Apikey.env")
    return OpenAI(base_url=cfg["base_url"], api_key=api_key)


def model_names() -> dict:
    """{'main': ..., 'check': ..., 'light': ...} for the active provider."""
    return _config()["models"]


