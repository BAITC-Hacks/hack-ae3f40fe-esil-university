"""HTTP API участника 1. Запуск: python -m uvicorn api:app --reload."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from src import FilterEngine, QueryError, SearchRequest, load_profiles
from src.data_loader import DEFAULT_DATASET


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    city: StrictStr = Field(min_length=1, examples=["Алматы"])
    date: StrictStr = Field(examples=["2026-10-10"])
    event_format: StrictStr = Field(min_length=1, examples=["корпоратив"])
    category: StrictStr = Field(min_length=1, examples=["Ведущий"])
    budget_kzt: StrictInt = Field(ge=0, examples=[1500000])
    languages: list[StrictStr] = Field(default_factory=list)
    duration_hours: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] | None = None


def create_app(dataset_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        path = dataset_path or os.environ.get("DATASET_PATH") or DEFAULT_DATASET
        # Ошибки датасета останавливают запуск: поврежденный каталог нельзя скрывать.
        application.state.engine = FilterEngine(load_profiles(path))
        yield

    application = FastAPI(
        title="HackAlem — Backend участника 1", version="1.0.0",
        description="Отбор до 5 кандидатов и до 3 карточек. Без бронирования и вызовов LLM.",
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
    def recommendations(body: SearchBody, request: Request):
        """A/B/C возвращаются с HTTP 200. Неверный ввод — HTTP 422."""
        try:
            query = SearchRequest.from_mapping(body.model_dump())
            return request.app.state.engine.search(query)
        except QueryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
