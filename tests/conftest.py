"""Тесты не используют реальные ключи или серверный датасет из окружения."""

import pytest


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch):
    for name in ("OPENAI_API_KEY", "NVIDIA_API_KEY", "LLM_PROVIDER", "OPENAI_MODEL",
                 "NVIDIA_MODEL", "OPENAI_BASE_URL", "NVIDIA_BASE_URL", "DATASET_PATH",
                 "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
