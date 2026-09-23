"""HTTP API участника 1. Запуск: python -m uvicorn api:app --reload."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from src import FilterEngine, QueryError, SearchRequest, load_profiles
from src.data_loader import DEFAULT_DATASET
from src.recommendation_service import RecommendationService


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    city: StrictStr = Field(min_length=1, examples=["Алматы"])
    date: StrictStr = Field(examples=["2026-10-10"])
    event_format: StrictStr = Field(min_length=1, examples=["корпоратив"])
    category: StrictStr = Field(min_length=1, examples=["Ведущий"])
    budget_kzt: StrictInt = Field(ge=0, examples=[1500000])
    languages: list[StrictStr] = Field(default_factory=list)
    duration_hours: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] | None = None


def create_app(dataset_path: str | Path | None = None, *, explainer=None,
               timeout_seconds: float = 8.0, env_file: str | Path | None = Path(__file__).with_name(".env")) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if env_file is not None:
            load_dotenv(env_file, override=False)
        path = dataset_path or os.environ.get("DATASET_PATH") or DEFAULT_DATASET
        # Ошибки датасета останавливают запуск: поврежденный каталог нельзя скрывать.
        application.state.engine = FilterEngine(load_profiles(path))
        application.state.service = RecommendationService(
            application.state.engine, explainer=explainer, timeout_seconds=timeout_seconds,
        )
        try:
            yield
        finally:
            if explainer is None:
                client = application.state.service.explainer.client
                if client is not None:
                    client.close()

    application = FastAPI(
        title="HackAlem — подбор подрядчиков", version="1.1.0",
        description="Строгий отбор до 5 кандидатов, до 3 карточек и объяснения OpenAI/NVIDIA.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request):
        engine = request.app.state.engine
        return {"status": "ok", "total_profiles": len(engine.profiles),
                "dataset_version": engine.dataset_version}

    @application.get("/options")
    def options(request: Request):
        """Значения для списков в Streamlit и границы календаря."""
        return request.app.state.engine.options()

    @application.post("/recommendations")
    async def recommendations(body: SearchBody, request: Request):
        """A/B/C возвращаются с HTTP 200. Неверный ввод — HTTP 422."""
        try:
            query = SearchRequest.from_mapping(body.model_dump())
            return await request.app.state.service.recommend(query)
        except QueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
