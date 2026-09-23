"""Проверка API -> FilterEngine -> LLMExplainer -> SDK -> HTTP-заглушка.

HTTP-заглушка заменяет внешнюю модель; платных запросов и настоящих ключей нет.
"""

import json
from pathlib import Path
import threading
import time

import httpx2
from fastapi.testclient import TestClient
from openai import OpenAI
import pytest

from api import create_app
from src import FilterEngine, load_profiles
from src.llm_explainer import LLMExplainer


ROOT = Path(__file__).resolve().parents[1]
QUERY = {"city": "Алматы", "date": "2026-10-10", "category": "Ведущий",
         "event_format": "корпоратив", "budget_kzt": 1500000}


def content_for(payload):
    return {"outcome": "SUCCESS", "explanations": {
        p["id"]: f"Начальная цена профиля {p['id']} — {p['price_from_kzt']} ₸; формат «корпоратив» указан в анкете."
        for p in reversed(payload["candidates"])
    }}


def completion(content):
    return httpx2.Response(200, json={
        "id": "mock-completion", "object": "chat.completion", "created": 0,
        "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
    })


def mock_explainer(handler, provider="openai"):
    explainer = LLMExplainer(provider=provider, api_key="test-key-not-real")
    assert explainer.client is not None
    explainer.client.close()
    base_url = explainer.base_url or "https://api.openai.com/v1"
    explainer.client = OpenAI(
        api_key="test-key-not-real", base_url=base_url, max_retries=0, timeout=8,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    return explainer


@pytest.mark.parametrize("provider", ["openai", "nvidia"])
def test_real_sdk_receives_selected_profiles_and_preserves_ranking(provider):
    sent = []

    def handler(request):
        data = json.loads(request.content)
        sent.append((request.url, data))
        payload = json.loads(data["messages"][1]["content"])
        return completion(content_for(payload))

    explainer = mock_explainer(handler, provider)
    try:
        with TestClient(create_app(explainer=explainer, env_file=None)) as client:
            result = client.post("/recommendations", json=QUERY).json()
            assert client.post("/recommendations", json=QUERY).json() == result
        expected = FilterEngine(load_profiles()).search(QUERY)
        assert result["explanation_status"]["source"] == "external_llm"
        assert len(sent) == 1  # Повтор использует кеш успешного ответа.
        url, data = sent[0]
        assert url.host == ("api.openai.com" if provider == "openai" else "integrate.api.nvidia.com")
        assert url.path == "/v1/chat/completions"
        assert data["temperature"] == 0 and data["max_tokens"] == 512
        payload = json.loads(data["messages"][1]["content"])
        assert payload["outcome"] == "SUCCESS"
        assert len(payload["candidates"]) == 3
        assert all(p["name"] == p["anon_name"] for p in payload["candidates"])
        assert payload["user_request"]["budget"] == QUERY["budget_kzt"]
        assert payload["user_request"]["date"] == QUERY["date"]
        assert result["candidates"] == expected["candidates"]
        for actual, original in zip(result["cards"], expected["cards"], strict=True):
            assert actual["explanation_source"] == "external_llm"
            assert {k: v for k, v in actual.items() if not k.startswith("explanation")} == {
                k: v for k, v in original.items() if not k.startswith("explanation")}
    finally:
        explainer.client.close()


@pytest.mark.parametrize("request_path", sorted((ROOT / "examples").glob("*_request.json")), ids=lambda p: p.stem)
def test_six_scenarios_through_complete_api_without_key(request_path):
    body = json.loads(request_path.read_text())
    expected = FilterEngine(load_profiles()).search(body)
    with TestClient(create_app(env_file=None)) as client:
        response = client.post("/recommendations", json=body)
    assert response.status_code == 200
    result = response.json()
    metadata = result.pop("explanation_status")
    assert result == expected
    assert metadata["reason"] == ("missing_api_key" if result["cards"] else "empty_result")


@pytest.mark.parametrize("failure", ["rate_limit", "network_timeout", "bad_json", "wrong_id", "wrong_outcome", "cliche"])
def test_failures_keep_factual_backend_cards_and_do_not_poison_cache(failure):
    calls = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if len(calls) > 1:
            return completion(content_for(payload))
        if failure == "rate_limit":
            return httpx2.Response(429, json={"error": {"message": "test", "type": "rate_limit_error"}})
        if failure == "network_timeout":
            raise httpx2.ReadTimeout("test timeout", request=request)
        if failure == "bad_json":
            response = completion({})
            data = response.json()
            data["choices"][0]["message"]["content"] = "this is not JSON"
            return httpx2.Response(200, json=data)
        answer = content_for(payload)
        if failure == "wrong_id":
            answer["explanations"] = {"unknown": "Другой подрядчик."}
        elif failure == "wrong_outcome":
            answer["outcome"] = "NO_AVAILABLE_CONTRACTORS"
        else:
            answer["explanations"][payload["candidates"][0]["id"]] = "Это отличный выбор."
        return completion(answer)

    explainer = mock_explainer(handler)
    try:
        with TestClient(create_app(explainer=explainer, env_file=None)) as client:
            first = client.post("/recommendations", json=QUERY).json()
            assert first["cards"] == FilterEngine(load_profiles()).search(QUERY)["cards"]
            assert first["explanation_status"]["source"] == "deterministic_template"
            assert len(calls) == 1  # Повторов платного запроса внутри одного поиска нет.
            second = client.post("/recommendations", json=QUERY).json()
            assert second["explanation_status"]["source"] == "external_llm"
            assert len(calls) == 2  # Следующий поиск снова пробует API, ошибка не кешируется.
    finally:
        explainer.client.close()


def test_empty_results_do_not_call_model():
    calls = []
    explainer = mock_explainer(lambda request: calls.append(request))
    try:
        with TestClient(create_app(explainer=explainer, env_file=None)) as client:
            for changes, outcome in [({"budget_kzt": 0}, "C"),
                                     ({"city": "Астана", "category": "Декоратор"}, "B")]:
                result = client.post("/recommendations", json={**QUERY, **changes}).json()
                assert result["outcome"] == outcome
                assert result["message"]
                assert result["explanation_status"]["reason"] == "empty_result"
        assert calls == []
    finally:
        explainer.client.close()


def test_overall_deadline_returns_while_model_is_still_running():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class SlowExplainer:
        provider = "openai"

        def generate_explanations(self, *_):
            started.set()
            release.wait(2)
            finished.set()
            return {"source": "fallback"}

    try:
        with TestClient(create_app(explainer=SlowExplainer(), env_file=None, timeout_seconds=0.1)) as client:
            start = time.monotonic()
            result = client.post("/recommendations", json=QUERY).json()
            assert started.is_set() and not finished.is_set()
            assert time.monotonic() - start < 1
            assert result["explanation_status"]["reason"] == "timeout"
            assert len(result["cards"]) == 3
    finally:
        release.set()
        assert finished.wait(2)


def test_languages_and_duration_are_forwarded_to_model():
    sent = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        sent.append(payload)
        return completion(content_for(payload))

    body = json.loads((ROOT / "examples/06_languages_duration_request.json").read_text())
    explainer = mock_explainer(handler)
    try:
        with TestClient(create_app(explainer=explainer, env_file=None)) as client:
            assert client.post("/recommendations", json=body).json()["explanation_status"]["source"] == "external_llm"
        assert sent[0]["user_request"]["language"] == sorted(body["languages"])
        assert sent[0]["user_request"]["duration"] == body["duration_hours"]
        assert len(sent[0]["candidates"]) == 1
    finally:
        explainer.client.close()


def test_dotenv_is_loaded_and_system_environment_takes_priority(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=nvidia\nNVIDIA_MODEL=test-file-model\n")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    # Restore the variable introduced by load_dotenv after the test as well.
    monkeypatch.setenv("NVIDIA_MODEL", "temporary")
    monkeypatch.delenv("NVIDIA_MODEL")
    with TestClient(create_app(env_file=env_file)) as client:
        assert client.app.state.service.explainer.provider == "openai"
        import os
        assert os.environ["NVIDIA_MODEL"] == "test-file-model"


def test_fallback_does_not_share_cache_with_configured_instance():
    local = LLMExplainer()
    expected = FilterEngine(load_profiles()).search(QUERY)
    selected = expected["candidates"][:3]
    assert local.generate_explanations(QUERY, selected, "SUCCESS")["source"] == "fallback"
    assert len(local._response_cache) == 0
    remote = mock_explainer(lambda request: completion(content_for(
        json.loads(json.loads(request.content)["messages"][1]["content"]))))
    try:
        assert remote.generate_explanations(QUERY, selected, "SUCCESS")["source"] == "llm"
        assert len(local._response_cache) == 0 and len(remote._response_cache) == 1
    finally:
        remote.client.close()
