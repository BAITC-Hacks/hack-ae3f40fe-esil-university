"""Проверка контракта с файлом участника 2 без реального вызова API."""

from src import FilterEngine, load_profiles
from src.llm_contract import apply_explanations
from src.llm_explainer import LLMExplainer


def test_existing_llm_explainer_accepts_backend_candidates_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    engine = FilterEngine(load_profiles())
    query = {"city": "Алматы", "date": "2026-10-10", "category": "Ведущий",
             "event_format": "корпоратив", "budget_kzt": 1500000}
    result = engine.search(query)
    selected = [{**p, "name": p["anon_name"]} for p in result["candidates"][:3]]
    response = LLMExplainer(provider="openai").generate_explanations(
        result["request"], selected, "SUCCESS"
    )
    updated = apply_explanations(result, response["explanations"])
    assert [p["id"] for p in updated["cards"]] == [p["id"] for p in result["cards"]]
    assert all(p["explanation"] for p in updated["cards"])


def test_existing_llm_explainer_accepts_empty_outcomes_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    engine = FilterEngine(load_profiles())
    for city, category, budget, expected in [
        ("Астана", "Декоратор", 1500000, "NO_CATEGORY_IN_CITY"),
        ("Алматы", "Ведущий", 0, "NO_AVAILABLE_CONTRACTORS"),
    ]:
        result = engine.search({"city": city, "date": "2026-10-10", "category": category,
                                "event_format": "корпоратив", "budget_kzt": budget})
        response = LLMExplainer(provider="openai").generate_explanations(
            result["request"], [], expected
        )
        assert response["outcome"] == expected
        assert response["message"]
