"""Общий путь: строгий отбор -> объяснения участника 2 -> неизменный порядок карточек."""

import math
import os

import anyio

from .filter_engine import FilterEngine
from .llm_contract import apply_explanations
from .llm_explainer import LLMExplainer
from .models import SearchRequest


class RecommendationService:
    def __init__(self, engine: FilterEngine, explainer=None, timeout_seconds: float = 8.0):
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 8:
            raise ValueError("Время ожидания LLM должно быть больше 0 и не больше 8 секунд")
        self.engine = engine
        self.explainer = explainer if explainer is not None else LLMExplainer(
            provider=os.getenv("LLM_PROVIDER", "openai")
        )
        self.timeout_seconds = timeout_seconds

    async def recommend(self, request: SearchRequest | dict) -> dict:
        result = self.engine.search(request)

        def finish(reason, source="deterministic_template"):
            result["explanation_status"] = {
                "source": source,
                "reason": reason,
                "provider": self.explainer.provider,
            }
            return result

        # Причины и числа уже вычислены точно; при B/C сетевой запрос не нужен.
        if not result["cards"]:
            return finish("empty_result")

        selected_ids = {card["id"] for card in result["cards"]}
        selected = [
            {**profile, "name": profile["anon_name"]}
            for profile in result["candidates"] if profile["id"] in selected_ids
        ]
        request_for_llm = {
            **result["request"],
            "budget": result["request"]["budget_kzt"],
            "language": result["request"]["languages"],
            "duration": result["request"]["duration_hours"],
        }
        try:
            # Ограничиваем общее ожидание, а не только отдельную сетевую операцию.
            with anyio.fail_after(self.timeout_seconds):
                answer = await anyio.to_thread.run_sync(
                    self.explainer.generate_explanations,
                    request_for_llm, selected, "SUCCESS", abandon_on_cancel=True,
                )
            if answer.get("source") != "llm":
                return finish(answer.get("fallback_reason", "unverified_source"))
            if answer.get("outcome") != "SUCCESS":
                return finish("invalid_response")
            result = apply_explanations(result, answer.get("explanations"))
            return finish("generated", source="external_llm")
        except TimeoutError:
            return finish("timeout")
        except (TypeError, ValueError, AttributeError):
            return finish("invalid_response")
        except Exception:
            return finish("request_failed")
