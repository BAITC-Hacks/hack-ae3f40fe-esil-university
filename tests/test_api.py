from fastapi.testclient import TestClient
import pytest
from api import create_app


@pytest.fixture
def client():
    with TestClient(create_app(env_file=None)) as connection:
        yield connection


def request(**changes):
    body = dict(city="Алматы", date="2026-10-10", category="Ведущий",
                event_format="корпоратив", budget_kzt=1500000)
    body.update(changes)
    return body


def test_health_options_and_openapi(client):
    assert client.get("/health").json()["total_profiles"] == 66
    assert client.get("/options").json()["synthetic_profiles"] == 13
    assert "/recommendations" in client.get("/openapi.json").json()["paths"]
    assert client.get("/docs").status_code == 200


def test_all_three_business_outcomes_are_http_200(client):
    for body, status in [(request(), "ok"),
                         (request(city="Астана", category="Декоратор"), "no_category_in_city"),
                         (request(budget_kzt=0), "no_matches")]:
        response = client.post("/recommendations", json=body)
        assert response.status_code == 200
        assert response.json()["status"] == status


@pytest.mark.parametrize("changes", [
    {"budget_kzt": -1}, {"budget_kzt": True}, {"budget_kzt": "500000"},
    {"date": "2027-01-01"}, {"date": "2026-02-30"}, {"city": ""},
    {"duration_hours": 0}, {"duration_hours": True}, {"languages": "русский"},
    {"unknown": 1},
])
def test_invalid_input_returns_422(client, changes):
    assert client.post("/recommendations", json=request(**changes)).status_code == 422


def test_missing_body_and_required_field(client):
    assert client.post("/recommendations").status_code == 422
    assert client.post("/recommendations", json={"city": "Алматы"}).status_code == 422


def test_http_repeatability_and_limits(client):
    a = client.post("/recommendations", json=request()).json()
    b = client.post("/recommendations", json=request()).json()
    assert a == b
    assert len(a["cards"]) == 3
    assert len(a["candidates"]) == 5
